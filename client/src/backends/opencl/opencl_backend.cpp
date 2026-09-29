#include "backends/opencl/opencl_backend.h"

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "backends/common/lowbits_filter.h"
#include "backends/common/row_batch.h"
#include "backends/opencl/search_kernel.h" // generated from search.cl - see CMakeLists.txt
#include "engine/hash_match.h"
#include "engine/limits.h"

// Like CUDA_CHECK: an OpenCL call failing mid-search means a broken driver
// or device, not something a search can recover from.
#define CL_CHECK(call)                                                                    \
    do {                                                                                  \
        cl_int err__ = (call);                                                            \
        if (err__ != CL_SUCCESS) {                                                        \
            fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", err__, __FILE__, __LINE__, #call); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

namespace {

// Work-items per work-group, unless the device or kernel allows fewer. Must
// be >= the alphabet size: the kernel fills its local tables one entry per
// work-item.
constexpr size_t kPreferredWorkGroupSize = 256;

std::string deviceInfoString(cl_device_id device, cl_device_info what) {
    size_t size = 0;
    clGetDeviceInfo(device, what, 0, nullptr, &size);
    std::string value(size, '\0');
    clGetDeviceInfo(device, what, size, value.data(), nullptr);
    while (!value.empty() && value.back() == '\0')
        value.pop_back();
    return value;
}

class OpenClBackend : public SearchBackend {
public:
    OpenClBackend(cl_device_id device, cl_context context, cl_command_queue queue)
        : device_(device), context_(context), queue_(queue) {
        cl_int err = CL_SUCCESS;
        matchCount_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
        CL_CHECK(err);
        matchIdx_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, MAX_MATCHES * sizeof(cl_ulong), nullptr, &err);
        CL_CHECK(err);
        for (cl_mem* table : {&alphabetKey_, &alphabetOrd_}) {
            *table = clCreateBuffer(context_, CL_MEM_READ_ONLY, MAX_ALPHABET_SIZE * sizeof(cl_uint), nullptr, &err);
            CL_CHECK(err);
        }
        for (cl_mem* table : {&suffixKey_, &suffixOrd_}) {
            *table = clCreateBuffer(context_, CL_MEM_READ_ONLY, kMaxSuffixSize * sizeof(cl_uint), nullptr, &err);
            CL_CHECK(err);
        }
        filterTable_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, kLowBitsFilterEntries * sizeof(cl_ulong), nullptr, &err);
        CL_CHECK(err);
    }

    ~OpenClBackend() override {
        for (auto& entry : kernels_) {
            clReleaseKernel(entry.second.kernel);
            clReleaseProgram(entry.second.program);
        }
        for (cl_mem buffer : {matchCount_, matchIdx_, alphabetKey_, alphabetOrd_, suffixKey_, suffixOrd_, filterTable_})
            clReleaseMemObject(buffer);
        clReleaseCommandQueue(queue_);
        clReleaseContext(context_);
    }

    const char* name() const override { return "opencl"; }
    // The kernel is compiled for the alphabet size at hand, so any size works.
    std::vector<int> supportedAlphabetSizes() const override { return {}; }
    int windowChars() const override { return windowCharsOr(5); }
    // Row indices are 32-bit in the kernel: 50^5 < 2^32 <= 50^6.
    int maxTrailingLen() const override { return 6; }
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * rowsPerBatchOr(1u << 23); }

    void beginSearch(const SearchConstants& constants) override;
    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;
    void endSearch() override {}

private:
    struct CompiledKernel {
        cl_program program;
        cl_kernel kernel;
        size_t workGroupSize;
    };

    const CompiledKernel& kernelFor(int trailingLen);

    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_mem matchCount_ = nullptr;
    cl_mem matchIdx_ = nullptr;
    cl_mem alphabetKey_ = nullptr, alphabetOrd_ = nullptr;
    cl_mem suffixKey_ = nullptr, suffixOrd_ = nullptr;
    // This search's lookup filter: kLowBitsFilterEntries entries, see
    // buildLowBitsFilterTable (backends/common/lowbits_filter.h).
    cl_mem filterTable_ = nullptr;
    // Compiled once per (alphabet size, suffix length, trailing length) and
    // kept for the backend's lifetime - a coordinator client searches range
    // after range of the same shape, and each compile takes a moment.
    std::map<std::tuple<int, int, int>, CompiledKernel> kernels_;
    bool announcedDevice_ = false;

    HitVerifier verifier_;
    int alphabetSize_ = 0;
    int suffixLen_ = 0;
    uint32_t targetA_ = 0;
};

// How many of the lookup filter's entries beginSearch checks against their
// definition before every search (see checkLowBitsFilterTable), each with two
// different random high bits - about a millisecond, as in the CUDA backend.
constexpr uint32_t kFilterEntriesCheckedPerSearch = 1024;

// A search must not start with a filter table that is wrong: it could drop a
// match without any other sign. Like CL_CHECK, this ends the process - a
// coordinator range is then reassigned when its lease runs out.
[[noreturn]] void refuseFilterTable(const char* what, const std::string& detail) {
    fprintf(stderr, "INTERNAL ERROR: the lookup filter table %s (%s) - refusing to search with it\n", what, detail.c_str());
    exit(1);
}

void OpenClBackend::beginSearch(const SearchConstants& constants) {
    if (!announcedDevice_) {
        printf("OpenCL device: %s (%s)\n", deviceInfoString(device_, CL_DEVICE_NAME).c_str(),
               deviceInfoString(device_, CL_DEVICE_VERSION).c_str());
        announcedDevice_ = true;
    }
    verifier_.begin(constants);
    alphabetSize_ = (int) constants.alphabet.size();
    suffixLen_ = (int) constants.suffix.size();
    targetA_ = constants.targetHashA;

    std::vector<cl_uint> key(MAX_ALPHABET_SIZE, 0), ord(MAX_ALPHABET_SIZE, 0);
    for (int k = 0; k < alphabetSize_; ++k) {
        ord[k] = (unsigned char) constants.alphabet[k];
        key[k] = constants.cryptTable[0x100 + ord[k]];
    }
    std::vector<cl_uint> suffixKey(kMaxSuffixSize, 0), suffixOrd(kMaxSuffixSize, 0);
    for (int i = 0; i < suffixLen_; ++i) {
        suffixOrd[i] = (unsigned char) constants.suffix[i];
        suffixKey[i] = constants.cryptTable[0x100 + suffixOrd[i]];
    }
    CL_CHECK(clEnqueueWriteBuffer(queue_, alphabetKey_, CL_TRUE, 0, key.size() * sizeof(cl_uint), key.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(queue_, alphabetOrd_, CL_TRUE, 0, ord.size() * sizeof(cl_uint), ord.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(queue_, suffixKey_, CL_TRUE, 0, suffixKey.size() * sizeof(cl_uint), suffixKey.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(queue_, suffixOrd_, CL_TRUE, 0, suffixOrd.size() * sizeof(cl_uint), suffixOrd.data(), 0, nullptr, nullptr));
    // Establishes the "matchCount is 0 at launch" invariant runBatch keeps.
    const cl_int zero = 0;
    CL_CHECK(clEnqueueWriteBuffer(queue_, matchCount_, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr));

    // This search's lookup filter - it depends on the alphabet, the suffix and
    // the target, so it's built for every search. Checked against its
    // definition before it's used, and read back after the upload to make
    // sure the device has exactly what was checked.
    const std::vector<uint64_t> table = buildLowBitsFilterTable(constants);
    std::string error;
    if (!checkLowBitsFilterTable(table, constants, kFilterEntriesCheckedPerSearch, 2, std::random_device{}(), error))
        refuseFilterTable("failed its check", error);
    static_assert(sizeof(cl_ulong) == sizeof(uint64_t), "the table's entries are 64-bit on both sides");
    const size_t tableBytes = table.size() * sizeof(uint64_t);
    CL_CHECK(clEnqueueWriteBuffer(queue_, filterTable_, CL_TRUE, 0, tableBytes, table.data(), 0, nullptr, nullptr));
    std::vector<uint64_t> readBack(table.size());
    CL_CHECK(clEnqueueReadBuffer(queue_, filterTable_, CL_TRUE, 0, tableBytes, readBack.data(), 0, nullptr, nullptr));
    if (readBack != table)
        refuseFilterTable("read back from the device differs from the one uploaded", std::to_string(tableBytes) + " bytes");
}

const OpenClBackend::CompiledKernel& OpenClBackend::kernelFor(int trailingLen) {
    const auto key = std::make_tuple(alphabetSize_, suffixLen_, trailingLen);
    auto found = kernels_.find(key);
    if (found != kernels_.end())
        return found->second;

    cl_int err = CL_SUCCESS;
    const char* source = kSearchKernelSource;
    cl_program program = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    CL_CHECK(err);
    const std::string options = "-cl-std=CL1.2 -DALPHABET_SIZE=" + std::to_string(alphabetSize_) + " -DSUFFIX_LEN=" + std::to_string(suffixLen_) +
                                " -DTRAILING_LEN=" + std::to_string(trailingLen) + " -DMAX_MATCHES=" + std::to_string(MAX_MATCHES) +
                                " -DHASHA_MATCH_MASK=" + std::to_string(kHashAMatchMask) + "u" +
                                " -DFILTER_BITS=" + std::to_string(kLowBitsFilterBits);
    if (clBuildProgram(program, 1, &device_, options.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        size_t size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::string log(size, '\0');
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
        fprintf(stderr, "OpenCL: compiling the search kernel (%s) failed:\n%s\n", options.c_str(), log.c_str());
        exit(1);
    }
    CompiledKernel compiled;
    compiled.program = program;
    compiled.kernel = clCreateKernel(program, "searchRows", &err);
    CL_CHECK(err);
    size_t kernelMax = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(compiled.kernel, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMax), &kernelMax, nullptr));
    compiled.workGroupSize = std::min(kPreferredWorkGroupSize, kernelMax);
    if (compiled.workGroupSize < (size_t) alphabetSize_) {
        fprintf(stderr, "OpenCL: the device allows only %zu work-items per group, fewer than the alphabet's %d characters\n",
                compiled.workGroupSize, alphabetSize_);
        exit(1);
    }
    return kernels_.emplace(key, compiled).first->second;
}

BatchOutcome OpenClBackend::runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) {
    const RowRange rows = rowRangeFor(start, count, alphabetSize_);
    if (rows.firstRow + rows.rowCount - 1 > UINT32_MAX || rows.rowCount > (1ull << 31)) {
        fprintf(stderr, "Batch too large for the kernel's 32-bit row index (rows %llu..%llu) - exiting\n",
                (unsigned long long) rows.firstRow, (unsigned long long) (rows.firstRow + rows.rowCount - 1));
        exit(1);
    }
    const CompiledKernel& compiled = kernelFor(trailingLen);
    cl_kernel kernel = compiled.kernel;

    const cl_uint firstRow = (cl_uint) rows.firstRow, rowCount = (cl_uint) rows.rowCount;
    const cl_int firstRowStartK = rows.firstRowStartK, lastRowEndK = rows.lastRowEndK;
    const cl_uint targetA = targetA_, seed1 = params.seed1Start, seed2 = params.seed2Start;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(firstRow), &firstRow));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(rowCount), &rowCount));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(firstRowStartK), &firstRowStartK));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(lastRowEndK), &lastRowEndK));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(targetA), &targetA));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(seed1), &seed1));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(seed2), &seed2));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &alphabetKey_));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &alphabetOrd_));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &suffixKey_));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem), &suffixOrd_));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_mem), &filterTable_));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem), &matchCount_));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_mem), &matchIdx_));

    const size_t local = compiled.workGroupSize;
    const size_t global = (rows.rowCount + local - 1) / local * local;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr));

    // The common batch costs exactly this one blocking read.
    cl_int hitCount = 0;
    CL_CHECK(clEnqueueReadBuffer(queue_, matchCount_, CL_TRUE, 0, sizeof(hitCount), &hitCount, 0, nullptr, nullptr));

    BatchOutcome outcome;
    outcome.hitCount = hitCount;
    if (hitCount == 0)
        return outcome;
    if (hitCount <= MAX_MATCHES) {
        std::vector<uint64_t> trailingIndices(hitCount);
        CL_CHECK(clEnqueueReadBuffer(queue_, matchIdx_, CL_TRUE, 0, hitCount * sizeof(cl_ulong), trailingIndices.data(), 0, nullptr, nullptr));
        verifier_.addHits(trailingIndices, trailingLen, params, outcome);
    }
    // Past MAX_MATCHES nothing was recorded completely - the engine searches
    // the range again in halves. Either way, restore "matchCount is 0".
    const cl_int zero = 0;
    CL_CHECK(clEnqueueWriteBuffer(queue_, matchCount_, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr));
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeOpenClBackend(std::string& error) {
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) {
        error = "no OpenCL platform (driver) installed";
        return nullptr;
    }
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    for (cl_platform_id platform : platforms) {
        cl_device_id device = nullptr;
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &deviceCount) != CL_SUCCESS || deviceCount == 0)
            continue;
        cl_int err = CL_SUCCESS;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            continue;
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            continue;
        }
        return std::make_unique<OpenClBackend>(device, context, queue);
    }
    error = "no OpenCL GPU found";
    return nullptr;
}
