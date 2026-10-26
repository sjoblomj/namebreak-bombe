#include "backends/opencl/opencl_backend.h"

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "backends/common/dictionary_batch.h"
#include "backends/common/meet_in_the_middle_launch.h"
#include "backends/common/launch_waiter.h"
#include "backends/common/lowbits_filter.h"
#include "backends/common/row_batch.h"
#include "backends/common/row_pruning.h"
#include "backends/opencl/dictionary_kernel.h"         // generated from dictionary.cl - see CMakeLists.txt
#include "backends/opencl/meet_in_the_middle_kernel.h" // generated from meet_in_the_middle.cl
#include "backends/opencl/search_kernel.h"             // generated from search.cl
#include "engine/dictionary_search.h"
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
// work-item. This and the other k*Default values below are what the backend
// uses where config.conf's [opencl] says nothing (GpuTuning, see
// backends/gpu_tuning.h).
constexpr int kWorkGroupSizeDefault = 256;

// About how many rows of a row group one work-item searches (ROWS_PER_THREAD
// in search.cl; see kChunksPerGroup in the CUDA backend). On the RTX 3080 Ti
// Laptop (search_bench --scale 20, three runs each), 1 row per work-item did
// about 880 G candidates/s, 7 about 1,300, 25 about 1,435, and 49 and 64 about
// 1,460 - a whole row group per work-item, for any alphabet size. (CUDA's
// kernel does best with two chunks per group; this one doesn't.)
constexpr int kRowsPerThreadDefault = rowsPerThreadOr(64);

// How many batches - usually each a whole leading value - one launch searches
// at most (SearchBackend::maxBatchesPerCall; see NAMEBREAK_BATCHES_PER_LAUNCH
// in the CUDA backend's tuning.h, whose 16 this takes over, measured there).
// A launch of one leading value is so short that the gaps between launches,
// and a CPU core spinning through all of them, cost more than they need to.
constexpr int kMaxBatchesPerLaunch = 32; // the batches buffer's size
constexpr int kBatchesPerLaunchDefault = batchesPerLaunchOr(16);
static_assert(kBatchesPerLaunchDefault >= 1 && kBatchesPerLaunchDefault <= kMaxBatchesPerLaunch, "NAMEBREAK_BATCHES_PER_LAUNCH must be 1-32");
// How many trailing characters the GPU enumerates (SearchBackend::windowChars).
constexpr int kWindowCharsDefault = windowCharsOr(5);

// A launch of a search that prunes the whole candidate walks its batches'
// lists of row groups (LISTED in search.cl) only if they leave out at least
// this percentage of its row groups; if not, it walks every group, and the
// hits the lists would have left out are dropped on the host (runBatches) -
// as the CUDA backend does (NAMEBREAK_LIST_MIN_PRUNED_PERCENT in its
// tuning.h), the list-walking kernel costing about 6% here.
constexpr int kListMinPrunedPercent = listMinPrunedPercentOr(5);
static_assert(kListMinPrunedPercent >= 0 && kListMinPrunedPercent <= 101, "NAMEBREAK_LIST_MIN_PRUNED_PERCENT must be 0-101");

// These three must match their namesakes in search.cl.
struct LaunchBatch {
    cl_uint firstRow;
    cl_uint lastRow;
    cl_int firstRowStartK;
    cl_int lastRowEndK;
    cl_uint seed1Start;
    cl_uint seed2Start;
    cl_uint groupsOffset;
    cl_uint groupCount;
};
struct Hit {
    cl_ulong trailingIdx;
    cl_uint batch;
    cl_uint unused;
};
struct BatchResults {
    cl_int matchCount;
    cl_int unused;
    Hit hits[MAX_MATCHES];
};
static_assert(sizeof(LaunchBatch) == 32 && sizeof(Hit) == 16 && offsetof(BatchResults, hits) == 8,
              "LaunchBatch, Hit and BatchResults must be laid out as in search.cl");
// How many hits come back with the count, in the one read every launch needs.
constexpr int kHitsReadWithCount = MAX_MATCHES < 16 ? MAX_MATCHES : 16;

// A dictionary search's launches (dictionary.cl): how many candidates one
// covers at most (SearchBackend::dictionaryCandidatesPerCall), how many
// cells (words, with tails) of a batch one work-item hashes, and the
// work-group's size - unless the device allows fewer. A segment of a launch
// is the last two's product. And with tails, how many a cell has (see
// DictionaryLaunchBatch). The CUDA backend's values (see its tuning.h).
constexpr uint64_t kDictionaryCandidatesPerLaunch = 1ull << 28;
// A meet-in-the-middle search's launches (meet_in_the_middle.cl): about how
// many tails one rewinds, how many hits it has room for at first, and its
// work-group size. The CUDA backend's values again
// (NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_LAUNCH,
// NAMEBREAK_MEET_IN_THE_MIDDLE_HIT_CAPACITY).
constexpr uint64_t kMeetInTheMiddleTailsPerLaunch = NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_LAUNCH;
constexpr cl_uint kMeetInTheMiddleHitCapacity = NAMEBREAK_MEET_IN_THE_MIDDLE_HIT_CAPACITY;
constexpr size_t kMeetInTheMiddleWorkGroupSize = 256;
constexpr int kDictionaryWordsPerThreadDefault = dictionaryWordsPerThreadOr(32);
constexpr int kDictionaryWorkGroupSizeDefault = dictionaryThreadsPerBlockOr(256);
constexpr int kDictionaryTailsPerCellDefault = dictionaryTailsPerCellOr(32);

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
    // `tuning` has every value set (withDefaults).
    OpenClBackend(cl_device_id device, cl_context context, cl_command_queue queue, const GpuTuning& tuning)
        : device_(device), context_(context), queue_(queue), tuning_(tuning), rowsPerThread_(*tuning.rowsPerThread),
          batchesPerLaunch_(*tuning.batchesPerLaunch), windowChars_(*tuning.windowChars), workGroupSize_((size_t) *tuning.workGroupSize),
          dictionaryWordsPerThread_(*tuning.dictionaryWordsPerThread), dictionaryWorkGroupSize_((size_t) *tuning.dictionaryWorkGroupSize),
          dictionaryTailsPerCell_(*tuning.dictionaryTailsPerCell) {
        cl_int err = CL_SUCCESS;
        results_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(BatchResults), nullptr, &err);
        CL_CHECK(err);
        batches_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, kMaxBatchesPerLaunch * sizeof(LaunchBatch), nullptr, &err);
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
        rowMasks_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, kRowFlagCount * sizeof(cl_ulong), nullptr, &err);
        CL_CHECK(err);
        inserted_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, 2 * 2 * kMaxInsertLen * sizeof(cl_uint), nullptr, &err);
        CL_CHECK(err);
    }

    ~OpenClBackend() override {
        for (auto& entry : kernels_) {
            clReleaseKernel(entry.second.kernel);
            clReleaseProgram(entry.second.program);
        }
        for (auto& entry : dictionaryKernels_) {
            clReleaseKernel(entry.second.kernel);
            clReleaseProgram(entry.second.program);
        }
        for (auto& entry : meetInTheMiddleKernels_) {
            clReleaseKernel(entry.second.kernel);
            clReleaseProgram(entry.second.program);
        }
        releaseDictionaryBuffers();
        releaseMeetInTheMiddleBuffers();
        for (cl_mem buffer : {meetInTheMiddleCounts_, meetInTheMiddleHits_}) {
            if (buffer)
                clReleaseMemObject(buffer);
        }
        for (cl_mem buffer : {dictionaryBatches_, dictionaryCounts_, dictionaryHashAHits_, dictionaryBasenameHits_}) {
            if (buffer)
                clReleaseMemObject(buffer);
        }
        for (cl_mem buffer : {results_, batches_, alphabetKey_, alphabetOrd_, suffixKey_, suffixOrd_, filterTable_, rowMasks_, inserted_})
            clReleaseMemObject(buffer);
        if (groups_)
            clReleaseMemObject(groups_);
        clReleaseCommandQueue(queue_);
        clReleaseContext(context_);
    }

    const char* name() const override { return "opencl"; }
    int windowChars() const override { return windowChars_; }
    // Row indices are 32-bit in the kernel: 63^5 < 2^32 <= 63^6.
    int maxTrailingLen() const override { return 6; }
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * rowsPerBatchOr(1u << 23); }

    void beginSearch(const SearchConstants& constants) override;
    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;
    void endSearch() override {}
    int maxBatchesPerCall() const override { return batchesPerLaunch_; }
    BatchOutcome runBatches(int trailingLen, const std::vector<BatchRequest>& batches) override;

    // A dictionary search: one launch of dictionary.cl's kernel per call.
    bool supportsDictionary() const override { return true; }
    uint64_t dictionaryCandidatesPerCall() const override { return kDictionaryCandidatesPerLaunch; }
    void beginDictionarySearch(const DictionaryConstants& constants) override;
    DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches) override;
    void endDictionarySearch() override { releaseDictionaryBuffers(); }

    // A meet-in-the-middle search: one launch of meet_in_the_middle.cl's kernel per call.
    bool supportsMeetInTheMiddle() const override { return true; }
    uint64_t meetInTheMiddleTailsPerCall() const override { return kMeetInTheMiddleTailsPerLaunch; }
    void beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) override;
    MeetInTheMiddleOutcome runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) override;
    void endMeetInTheMiddleSearch() override { releaseMeetInTheMiddleBuffers(); }

private:
    struct CompiledKernel {
        cl_program program;
        cl_kernel kernel;
        size_t workGroupSize;
    };

    const CompiledKernel& kernelFor(int trailingLen, bool listed);
    const CompiledKernel& dictionaryKernel();
    const CompiledKernel& meetInTheMiddleKernel();
    void releaseMeetInTheMiddleBuffers();
    void uploadGroups();
    // Compiles `source` with `options`, ending the process if it doesn't.
    cl_program buildProgram(const char* source, const std::string& options, const char* what);
    cl_mem createBuffer(cl_mem_flags flags, size_t bytes);
    void releaseDictionaryBuffers();

    // Prints the device and the launch geometry, before the first search
    // of any kind.
    void announceDevice();

    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    // The launch geometry (see GpuTuning), every value set.
    GpuTuning tuning_;
    int rowsPerThread_;
    int batchesPerLaunch_;
    int windowChars_;
    size_t workGroupSize_;
    int dictionaryWordsPerThread_;
    size_t dictionaryWorkGroupSize_;
    int dictionaryTailsPerCell_;
    cl_mem results_ = nullptr; // a BatchResults
    cl_mem batches_ = nullptr; // the launch's LaunchBatches
    // The host's copies: the batches, written without waiting (they must stay
    // put until the launch is done, which runBatches waits for anyway), and
    // what a launch reports.
    LaunchBatch hostBatches_[kMaxBatchesPerLaunch] = {};
    std::unique_ptr<BatchResults> hostResults_ = std::make_unique<BatchResults>();
    // Sleeps through most of each launch, instead of spinning in the blocking
    // read for all of it (see backends/common/launch_waiter.h).
    LaunchWaiter waiter_;
    cl_mem alphabetKey_ = nullptr, alphabetOrd_ = nullptr;
    cl_mem suffixKey_ = nullptr, suffixOrd_ = nullptr;
    // This search's lookup filter: kLowBitsFilterEntries entries, see
    // buildLowBitsFilterTable (backends/common/lowbits_filter.h).
    cl_mem filterTable_ = nullptr;
    // The text inserted into the trailing part: each character's key, then
    // its value (see search.cl's `inserted`), and where each insertion is -
    // compiled into the kernel (insertMacros_).
    cl_mem inserted_ = nullptr;
    std::string insertMacros_;
    // Compiled once per (alphabet size, suffix length, trailing length,
    // LISTED, insertMacros_) and kept for the backend's lifetime - a
    // coordinator client searches range after range of the same shape, and
    // each compile takes a moment.
    std::map<std::tuple<int, int, int, bool, std::string>, CompiledKernel> kernels_;
    bool announcedDevice_ = false;

    HitVerifier verifier_;
    int alphabetSize_ = 0;
    int suffixLen_ = 0;
    uint32_t targetA_ = 0;

    // The row groups each batch searches (see backends/common/row_pruning.h),
    // and their copy on the device: the first groupsUploaded_ entries of
    // rowPruning_.arena(), of generation groupsGeneration_, in a buffer of
    // groupsCapacity_ entries. Kept from one search to the next, as the
    // lists are.
    RowPruning rowPruning_;
    cl_mem rowMasks_ = nullptr; // kRowFlagCount of them
    cl_mem groups_ = nullptr;
    size_t groupsCapacity_ = 0;
    size_t groupsUploaded_ = 0;
    uint64_t groupsGeneration_ = 0;
    // Whether this search prunes the whole candidate, and so has lists of row
    // groups (LISTED in search.cl): only then - a kernel that walks every
    // group measured about 6% faster than one walking a list of them all, so
    // a launch walks them only if they leave out enough groups (see
    // runBatches) - or as SearchConstants::listWalking says.
    bool listed_ = false;
    SearchConstants::ListWalking listWalking_ = SearchConstants::ListWalking::Auto;

    // A dictionary search's: the word list, the crypt table's keys, the
    // suffix filters and the suffix's keys (backends/common/
    // dictionary_batch.h) - made by beginDictionarySearch and released by
    // endDictionarySearch - and what the kernel is compiled with.
    DictionaryHitVerifier dictionaryVerifier_;
    cl_mem dictionaryChars_ = nullptr, dictionaryEntries_ = nullptr, dictionaryPositions_ = nullptr;
    cl_mem dictionaryCryptKeys_ = nullptr, dictionaryFilters_ = nullptr, dictionarySuffixKeys_ = nullptr;
    cl_mem dictionaryTails_ = nullptr; // dictionaryTailTable
    bool dictionaryHasTails_ = false;
    cl_uint dictionaryWordCount_ = 0;
    cl_uint dictionarySeparatorCount_ = 0;
    int dictionarySuffixLen_ = 0;
    DictionaryHashes dictionaryHashes_ = DictionaryHashes::HashA;
    cl_uint dictionaryTargetA_ = 0;
    cl_uint dictionaryBasenameKey_ = 0;
    // Compiled once per (suffix length, hashes, tails or not) and kept, as
    // the row search's kernels are.
    std::map<std::tuple<int, DictionaryHashes, bool>, CompiledKernel> dictionaryKernels_;
    // Kept from one search to the next, grown as a launch needs: its batches
    // (DictionaryLaunchBatch), the hit counts, and the hits - room for
    // dictionaryHitCapacity_ of each kind.
    std::vector<DictionaryLaunchBatch> dictionaryLaunch_;
    cl_mem dictionaryBatches_ = nullptr;
    size_t dictionaryBatchesCapacity_ = 0;
    cl_mem dictionaryCounts_ = nullptr;
    cl_mem dictionaryHashAHits_ = nullptr, dictionaryBasenameHits_ = nullptr;

    // A meet-in-the-middle search's: the engine's table and the alphabet as
    // the kernel reads it - made by beginMeetInTheMiddleSearch and released
    // by endMeetInTheMiddleSearch - the kernels compiled for the rules of a
    // search (keyed by the options they were compiled with), and the counts
    // and hits a launch reports.
    MeetInTheMiddleConstants meet_in_the_middle_ = {};
    cl_mem meetInTheMiddleKeys_ = nullptr, meetInTheMiddleFilter_ = nullptr, meetInTheMiddleChars_ = nullptr;
    cl_ulong meetInTheMiddleKeyCount_ = 0;
    cl_int meetInTheMiddleFilterShift_ = 64;
    std::map<std::string, CompiledKernel> meetInTheMiddleKernels_;
    cl_mem meetInTheMiddleCounts_ = nullptr;
    cl_mem meetInTheMiddleHits_ = nullptr;
    cl_uint meetInTheMiddleHitCapacity_ = 0;
    cl_uint dictionaryHitCapacity_ = 0;
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

void OpenClBackend::announceDevice() {
    if (announcedDevice_)
        return;
    printf("OpenCL device: %s (%s)\n", deviceInfoString(device_, CL_DEVICE_NAME).c_str(), deviceInfoString(device_, CL_DEVICE_VERSION).c_str());
    printf("OpenCL launches: %s\n", describeGpuTuning("opencl", tuning_).c_str());
    announcedDevice_ = true;
}

void OpenClBackend::beginSearch(const SearchConstants& constants) {
    announceDevice();
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
    {
        const InsertLayout layout = insertLayoutFor(constants);
        std::vector<cl_uint> inserted(2 * 2 * kMaxInsertLen, 0);
        for (size_t i = 0; i < layout.key.size(); ++i) {
            inserted[2 * i] = layout.key[i];
            inserted[2 * i + 1] = layout.ord[i];
        }
        CL_CHECK(clEnqueueWriteBuffer(queue_, inserted_, CL_TRUE, 0, inserted.size() * sizeof(cl_uint), inserted.data(), 0, nullptr, nullptr));
        insertMacros_.clear();
        for (int n = 0; n < 2; ++n) {
            insertMacros_ += " -DGROUP_INSERT" + std::to_string(n) + "_AFTER=" + std::to_string(layout.groupCharsAfter[n]) + " -DGROUP_INSERT" +
                             std::to_string(n) + "_START=" + std::to_string(layout.groupStart[n]) + " -DGROUP_INSERT" + std::to_string(n) +
                             "_LEN=" + std::to_string(layout.groupLen[n]);
        }
        insertMacros_ += " -DROW_INSERT_START=" + std::to_string(layout.rowStart) + " -DROW_INSERT_LEN=" + std::to_string(layout.rowLen);
    }
    listed_ = constants.trailingRules.any();
    listWalking_ = constants.listWalking;
    if (listed_) {
        rowPruning_.begin(constants);
        CL_CHECK(clEnqueueWriteBuffer(queue_, rowMasks_, CL_TRUE, 0, kRowFlagCount * sizeof(cl_ulong), rowPruning_.rowMasks(), 0, nullptr, nullptr));
    }
    // Establishes the "matchCount is 0 at launch" invariant runBatches keeps.
    const cl_int zero = 0;
    CL_CHECK(clEnqueueWriteBuffer(queue_, results_, CL_TRUE, offsetof(BatchResults, matchCount), sizeof(zero), &zero, 0, nullptr, nullptr));
    waiter_.beginSearch();

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

const OpenClBackend::CompiledKernel& OpenClBackend::kernelFor(int trailingLen, bool listed) {
    const auto key = std::make_tuple(alphabetSize_, suffixLen_, trailingLen, listed, insertMacros_);
    auto found = kernels_.find(key);
    if (found != kernels_.end())
        return found->second;

    cl_int err = CL_SUCCESS;
    const std::string options = "-cl-std=CL1.2 -DALPHABET_SIZE=" + std::to_string(alphabetSize_) + " -DSUFFIX_LEN=" + std::to_string(suffixLen_) +
                                " -DTRAILING_LEN=" + std::to_string(trailingLen) + " -DMAX_MATCHES=" + std::to_string(MAX_MATCHES) +
                                " -DHASHA_MATCH_MASK=" + std::to_string(kHashAMatchMask) + "u" +
                                " -DFILTER_BITS=" + std::to_string(kLowBitsFilterBits) +
                                " -DROWS_PER_THREAD=" + std::to_string(rowsPerThread_) + " -DROW_FLAG_BITS=" + std::to_string(kRowFlagBits) +
                                " -DLISTED=" + (listed ? "1" : "0") + insertMacros_;
    cl_program program = buildProgram(kSearchKernelSource, options, "search");
    CompiledKernel compiled;
    compiled.program = program;
    compiled.kernel = clCreateKernel(program, "searchRows", &err);
    CL_CHECK(err);
    size_t kernelMax = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(compiled.kernel, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMax), &kernelMax, nullptr));
    compiled.workGroupSize = std::min(workGroupSize_, kernelMax);
    if (compiled.workGroupSize < (size_t) alphabetSize_) {
        fprintf(stderr, "OpenCL: the device allows only %zu work-items per group, fewer than the alphabet's %d characters\n",
                compiled.workGroupSize, alphabetSize_);
        exit(1);
    }
    return kernels_.emplace(key, compiled).first->second;
}

cl_program OpenClBackend::buildProgram(const char* source, const std::string& options, const char* what) {
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    CL_CHECK(err);
    if (clBuildProgram(program, 1, &device_, options.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        size_t size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::string log(size, '\0');
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
        fprintf(stderr, "OpenCL: compiling the %s kernel (%s) failed:\n%s\n", what, options.c_str(), log.c_str());
        exit(1);
    }
    return program;
}

// Brings the device's copy of the row groups' lists up to date with
// rowPruning_.arena(), which only grows within a generation: only what was
// added since goes up, unless the buffer is too small or the lists were
// cleared. The write is blocking, and nothing is running on the device here
// (runBatches waits for every launch before it returns).
void OpenClBackend::uploadGroups() {
    const std::vector<uint32_t>& arena = rowPruning_.arena();
    if (rowPruning_.generation() != groupsGeneration_) {
        groupsGeneration_ = rowPruning_.generation();
        groupsUploaded_ = 0;
    }
    if (arena.size() > groupsCapacity_) {
        if (groups_)
            CL_CHECK(clReleaseMemObject(groups_));
        groupsCapacity_ = std::max(arena.size(), 2 * groupsCapacity_);
        cl_int err = CL_SUCCESS;
        groups_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, groupsCapacity_ * sizeof(cl_uint), nullptr, &err);
        CL_CHECK(err);
        groupsUploaded_ = 0;
    }
    if (arena.size() > groupsUploaded_) {
        CL_CHECK(clEnqueueWriteBuffer(queue_, groups_, CL_TRUE, groupsUploaded_ * sizeof(cl_uint), (arena.size() - groupsUploaded_) * sizeof(cl_uint),
                                      arena.data() + groupsUploaded_, 0, nullptr, nullptr));
        groupsUploaded_ = arena.size();
    }
}

BatchOutcome OpenClBackend::runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) {
    return runBatches(trailingLen, {BatchRequest{start, count, params}});
}

BatchOutcome OpenClBackend::runBatches(int trailingLen, const std::vector<BatchRequest>& requests) {
    const int batchCount = (int) requests.size();
    if (batchCount < 1 || batchCount > batchesPerLaunch_) {
        fprintf(stderr, "INTERNAL ERROR: %d batches for one launch (1 to %d allowed) - exiting\n", batchCount, batchesPerLaunch_);
        exit(1);
    }
    // Each batch's rows, cut to its range in its first and last row (see
    // search.cl), its row groups to search, and the most any of them has -
    // and with the whole candidate pruned, the list of those that survive.
    uint64_t maxGroups = 0, maxListedGroups = 0, groups = 0, listedGroups = 0, candidates = 0;
    for (int b = 0; b < batchCount; ++b) {
        const RowRange rows = rowRangeFor(requests[b].start, requests[b].count, alphabetSize_);
        if (requests[b].count == 0 || rows.firstRow + rows.rowCount - 1 > UINT32_MAX || rows.rowCount > (1ull << 31)) {
            fprintf(stderr, "Batch too large for the kernel's 32-bit row index (rows %llu..%llu) - exiting\n",
                    (unsigned long long) rows.firstRow, (unsigned long long) (rows.firstRow + rows.rowCount - 1));
            exit(1);
        }
        LaunchBatch& batch = hostBatches_[b];
        batch.firstRow = (cl_uint) rows.firstRow;
        batch.lastRow = (cl_uint) (rows.firstRow + rows.rowCount - 1);
        batch.firstRowStartK = rows.firstRowStartK;
        batch.lastRowEndK = rows.lastRowEndK;
        batch.seed1Start = requests[b].params.seed1Start;
        batch.seed2Start = requests[b].params.seed2Start;
        const uint64_t batchGroups = batch.lastRow / alphabetSize_ - batch.firstRow / alphabetSize_ + 1;
        maxGroups = std::max(maxGroups, batchGroups);
        groups += batchGroups;
        if (listed_) {
            const RowPruning::Slice list =
                rowPruning_.groupsFor(trailingLen, requests[b].params.pruneEntry, batch.firstRow / alphabetSize_, batch.lastRow / alphabetSize_);
            batch.groupsOffset = list.offset;
            batch.groupCount = list.count;
            maxListedGroups = std::max<uint64_t>(maxListedGroups, list.count);
            listedGroups += list.count;
        }
        candidates += requests[b].count;
    }
    // Every row of every batch pruned: nothing to launch.
    if (listed_ && maxListedGroups == 0)
        return BatchOutcome();
    // Whether this launch walks its batches' lists, or every group they
    // touch - when the lists leave out too few of them to pay for walking
    // them (see kListMinPrunedPercent), and the hits they'd have left out
    // are dropped below instead.
    using ListWalking = SearchConstants::ListWalking;
    const bool listed = listed_ && listWalking_ != ListWalking::Never &&
                        (listWalking_ == ListWalking::Always || (groups - listedGroups) * 100 >= groups * kListMinPrunedPercent);
    if (listed) {
        maxGroups = maxListedGroups;
        uploadGroups();
    }
    const CompiledKernel& compiled = kernelFor(trailingLen, listed);
    cl_kernel kernel = compiled.kernel;
    // Not waited for: the queue runs it before the kernel, and hostBatches_
    // isn't touched again until this launch is done.
    CL_CHECK(clEnqueueWriteBuffer(queue_, batches_, CL_FALSE, 0, batchCount * sizeof(LaunchBatch), hostBatches_, 0, nullptr, nullptr));

    const cl_uint targetA = targetA_;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(targetA), &targetA));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &batches_));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &alphabetKey_));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &alphabetOrd_));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &suffixKey_));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &suffixOrd_));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &filterTable_));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &results_));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &groups_)); // null (allowed) until a search has lists
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &rowMasks_));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem), &inserted_));

    // One row of work-groups per batch (dimension 1), and in it a work-item
    // per chunk of every row group the batch searches (see CHUNKS_PER_GROUP
    // in search.cl) - as many as the largest batch needs, rounded up to whole
    // work-groups. The kernel works out each batch's chunks itself and covers
    // all of them whatever the global size, so this only spreads the work.
    const size_t local[2] = {compiled.workGroupSize, 1};
    const size_t chunksPerGroup = (size_t) (alphabetSize_ + rowsPerThread_ - 1) / rowsPerThread_;
    const size_t chunks = (size_t) maxGroups * chunksPerGroup;
    const size_t global[2] = {(chunks + local[0] - 1) / local[0] * local[0], (size_t) batchCount};
    const LaunchWaiter::Clock::time_point launched = LaunchWaiter::Clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
    CL_CHECK(clFlush(queue_)); // on its way to the device before this sleeps

    // The common launch costs exactly this one blocking read: the count and
    // the first kHitsReadWithCount hits, so a launch with a few hits needs no
    // second one either.
    const size_t readWithCount = offsetof(BatchResults, hits) + kHitsReadWithCount * sizeof(Hit);
    waiter_.wait(candidates, launched, [&] {
        CL_CHECK(clEnqueueReadBuffer(queue_, results_, CL_TRUE, 0, readWithCount, hostResults_.get(), 0, nullptr, nullptr));
    });

    BatchOutcome outcome;
    const int hitCount = hostResults_->matchCount;
    outcome.hitCount = hitCount;
    if (hitCount == 0)
        return outcome;
    if (hitCount <= MAX_MATCHES && hitCount > kHitsReadWithCount) {
        CL_CHECK(clEnqueueReadBuffer(queue_, results_, CL_TRUE, offsetof(BatchResults, hits) + kHitsReadWithCount * sizeof(Hit),
                                     (hitCount - kHitsReadWithCount) * sizeof(Hit), hostResults_->hits + kHitsReadWithCount, 0, nullptr, nullptr));
    }
    // Restores "matchCount is 0 at launch" - queued, so it runs before the next
    // launch without anything waiting for it here.
    static const cl_int kZero = 0;
    CL_CHECK(clEnqueueWriteBuffer(queue_, results_, CL_FALSE, offsetof(BatchResults, matchCount), sizeof(kZero), &kZero, 0, nullptr, nullptr));
    // Past MAX_MATCHES nothing was recorded completely - the engine searches
    // the batches again in smaller pieces.
    if (hitCount > MAX_MATCHES)
        return outcome;
    // Every hit, checked on the CPU with its own batch's prefix (HitVerifier:
    // rebuilt from its trailing index and hashed from scratch, and hashB
    // checked). If the launch searched every row group where it had lists,
    // the hits in a row the lists leave out are dropped first, as walking the
    // lists would have.
    int kept = 0;
    for (int i = 0; i < hitCount; ++i) {
        const Hit& hit = hostResults_->hits[i];
        if (hit.batch >= (cl_uint) batchCount) {
            fprintf(stderr, "INTERNAL ERROR: the kernel reported a hit in batch %u of a launch of %d - exiting\n", hit.batch, batchCount);
            exit(1);
        }
        const LaunchBatch& batch = hostBatches_[hit.batch];
        if (listed_ && !listed && !rowPruning_.survives(RowPruning::Slice{batch.groupsOffset, batch.groupCount}, hit.trailingIdx / alphabetSize_))
            continue;
        verifier_.addHits({hit.trailingIdx}, trailingLen, requests[hit.batch].params, outcome);
        ++kept;
    }
    outcome.hitCount = kept;
    return outcome;
}


cl_mem OpenClBackend::createBuffer(cl_mem_flags flags, size_t bytes) {
    cl_int err = CL_SUCCESS;
    // At least a byte: an empty buffer isn't allowed, and a kernel argument
    // needs one even when it's never read.
    cl_mem buffer = clCreateBuffer(context_, flags, std::max<size_t>(1, bytes), nullptr, &err);
    CL_CHECK(err);
    return buffer;
}

void OpenClBackend::beginDictionarySearch(const DictionaryConstants& constants) {
    announceDevice();
    releaseDictionaryBuffers();
    dictionaryVerifier_.begin(constants);
    dictionaryHashes_ = dictionaryVerifier_.hashes();
    dictionaryTargetA_ = constants.targetHashA;
    dictionaryBasenameKey_ = constants.basenameKey;
    dictionarySuffixLen_ = (int) constants.suffix.size();
    dictionaryWordCount_ = (cl_uint) constants.words.size();
    dictionarySeparatorCount_ = (cl_uint) constants.separators.size();

    DictionaryWordTable table;
    if (!makeDictionaryWordTable(constants.words, constants.separators, table)) {
        // runDictionarySearch's word lists are far smaller.
        fprintf(stderr, "INTERNAL ERROR: the word list is too long for the GPU's 32-bit offsets - exiting\n");
        exit(1);
    }
    auto upload = [&](const auto& host, cl_mem& buffer) {
        const size_t bytes = host.size() * sizeof(host[0]);
        buffer = createBuffer(CL_MEM_READ_ONLY, bytes);
        if (bytes)
            CL_CHECK(clEnqueueWriteBuffer(queue_, buffer, CL_TRUE, 0, bytes, host.data(), 0, nullptr, nullptr));
    };
    static_assert(sizeof(DictionaryWordEntry) == sizeof(cl_uint4), "an entry is a uint4 in dictionary.cl");
    upload(table.chars, dictionaryChars_);
    upload(table.entries, dictionaryEntries_);
    upload(table.positions, dictionaryPositions_);
    // The tails - only "" without any, which the kernel built without TAILS
    // doesn't read.
    dictionaryHasTails_ = constants.tails != std::vector<std::string>{""};
    upload(dictionaryTailTable(constants.tails), dictionaryTails_);

    std::vector<cl_uint> cryptKeys(constants.cryptTable + kHashAOffset, constants.cryptTable + kHashAOffset + 256);
    cryptKeys.insert(cryptKeys.end(), constants.cryptTable + kFileKeyOffset, constants.cryptTable + kFileKeyOffset + 256);
    upload(cryptKeys, dictionaryCryptKeys_);
    // The suffix filters, hashA's and the basename hash's - each checked in
    // full before it's used, as the row search's filter table is (see
    // beginSearch): a wrong one could drop a match without any other sign.
    std::vector<cl_uint> filters;
    for (bool basename : {false, true}) {
        const int offset = basename ? kFileKeyOffset : kHashAOffset;
        const uint32_t target = basename ? constants.basenameKey : constants.targetHashA, mask = dictionaryFilterMask(basename);
        const std::vector<uint32_t> filter = buildDictionarySuffixFilter(constants.suffix, constants.cryptTable, offset, target, mask);
        std::string error;
        if (!checkDictionarySuffixFilter(filter, constants.suffix, constants.cryptTable, offset, target, mask, std::random_device{}(), error))
            refuseFilterTable("of the dictionary search's suffix failed its check", error);
        filters.insert(filters.end(), filter.begin(), filter.end());
    }
    upload(filters, dictionaryFilters_);
    upload(dictionarySuffixKeys(constants.suffix, constants.cryptTable), dictionarySuffixKeys_);

    // The counts are 0 at every launch: zeroed here, and after every launch
    // that had hits (see runDictionaryBatches).
    if (!dictionaryCounts_)
        dictionaryCounts_ = createBuffer(CL_MEM_READ_WRITE, 2 * sizeof(cl_int));
    const cl_int zeros[2] = {0, 0};
    CL_CHECK(clEnqueueWriteBuffer(queue_, dictionaryCounts_, CL_TRUE, 0, sizeof(zeros), zeros, 0, nullptr, nullptr));
    waiter_.beginSearch();
}

void OpenClBackend::releaseDictionaryBuffers() {
    for (cl_mem* buffer : {&dictionaryChars_, &dictionaryEntries_, &dictionaryPositions_, &dictionaryTails_, &dictionaryCryptKeys_,
                           &dictionaryFilters_, &dictionarySuffixKeys_}) {
        if (*buffer)
            clReleaseMemObject(*buffer);
        *buffer = nullptr;
    }
}

void OpenClBackend::beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) {
    announceDevice();
    releaseMeetInTheMiddleBuffers();
    meet_in_the_middle_ = constants;
    const MeetInTheMiddleTable& table = *constants.table;

    // The alphabet, as the kernel reads it: each character's hashA crypt
    // table key, its own value, and what the pruning rules make of it.
    std::vector<cl_uint4> chars(std::max<size_t>(1, constants.alphabet.size()), cl_uint4{{0, 0, 0, 0}});
    for (size_t k = 0; k < constants.alphabet.size(); ++k) {
        const unsigned char ch = (unsigned char) constants.alphabet[k];
        chars[k].s[0] = constants.cryptTable[kHashAOffset + ch];
        chars[k].s[1] = ch;
        chars[k].s[2] = (isRunSymbol_CPU((char) ch) ? 1u : 0u) | (ch == '\\' ? 2u : 0u);
    }
    meetInTheMiddleChars_ = createBuffer(CL_MEM_READ_ONLY, chars.size() * sizeof(cl_uint4));
    CL_CHECK(clEnqueueWriteBuffer(queue_, meetInTheMiddleChars_, CL_TRUE, 0, chars.size() * sizeof(cl_uint4), chars.data(), 0, nullptr, nullptr));

    // The table: the one thing a meet-in-the-middle search has that doesn't
    // fit in a kernel argument, and the one thing that can be too big for the
    // device
    // - the engine sizes it to the memory it was given, which is the
    // machine's, not this device's.
    meetInTheMiddleKeyCount_ = table.size();
    meetInTheMiddleFilterShift_ = table.filterShift();
    const size_t keyBytes = std::max<size_t>(1, table.size() * sizeof(cl_ulong));
    const size_t filterBytes = std::max<size_t>(1, (size_t) table.filterWords() * sizeof(cl_ulong));
    cl_int keyErr = CL_SUCCESS, filterErr = CL_SUCCESS;
    meetInTheMiddleKeys_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, keyBytes, nullptr, &keyErr);
    meetInTheMiddleFilter_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, filterBytes, nullptr, &filterErr);
    if (keyErr != CL_SUCCESS || filterErr != CL_SUCCESS) {
        cl_ulong deviceBytes = 0;
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceBytes), &deviceBytes, nullptr);
        fprintf(stderr,
                "The meet-in-the-middle search's table needs %zu MB on the OpenCL device, and it has %llu MB in all.\n"
                "Give the table less memory (config.conf's table_memory, a character shorter is %u times smaller),\n"
                "or search on the cpu backend (--backend cpu), which uses the machine's own memory.\n",
                (keyBytes + filterBytes) / (1024 * 1024), (unsigned long long) (deviceBytes / (1024 * 1024)),
                (unsigned) constants.alphabet.size());
        exit(1);
    }
    if (table.size() > 0)
        CL_CHECK(clEnqueueWriteBuffer(queue_, meetInTheMiddleKeys_, CL_TRUE, 0, table.size() * sizeof(cl_ulong), table.keys(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(queue_, meetInTheMiddleFilter_, CL_TRUE, 0, (size_t) table.filterWords() * sizeof(cl_ulong), table.filter(), 0,
                                  nullptr, nullptr));

    // The counts are 0 at every launch: zeroed here, and after every launch
    // that had anything to report (see runMeetInTheMiddleBatch).
    if (!meetInTheMiddleCounts_)
        meetInTheMiddleCounts_ = createBuffer(CL_MEM_READ_WRITE, 2 * sizeof(cl_uint));
    const cl_uint zeros[2] = {0, 0};
    CL_CHECK(clEnqueueWriteBuffer(queue_, meetInTheMiddleCounts_, CL_TRUE, 0, sizeof(zeros), zeros, 0, nullptr, nullptr));
    if (!meetInTheMiddleHits_) {
        meetInTheMiddleHitCapacity_ = kMeetInTheMiddleHitCapacity;
        meetInTheMiddleHits_ = createBuffer(CL_MEM_READ_WRITE, meetInTheMiddleHitCapacity_ * sizeof(cl_ulong));
    }
    waiter_.beginSearch();
}

void OpenClBackend::releaseMeetInTheMiddleBuffers() {
    for (cl_mem* buffer : {&meetInTheMiddleKeys_, &meetInTheMiddleFilter_, &meetInTheMiddleChars_}) {
        if (*buffer)
            clReleaseMemObject(*buffer);
        *buffer = nullptr;
    }
}

const OpenClBackend::CompiledKernel& OpenClBackend::meetInTheMiddleKernel() {
    // The alphabet and the rules are compiled in: they don't change within a
    // search, and the alphabet's size makes the kernel's divisions by it
    // multiplications.
    const PruneRules& rules = meet_in_the_middle_.rules;
    const std::string options = "-cl-std=CL1.2 -DALPHABET_SIZE=" + std::to_string(meet_in_the_middle_.alphabet.size()) +
                                " -DMAX_CANDIDATE_LEN=" + std::to_string(MAX_CANDIDATE_LEN) +
                                " -DMEET_IN_THE_MIDDLE_SEED_MASK=" + std::to_string(kMeetInTheMiddleSeedMask) + "u" +
                                " -DRULE_ANY=" + std::to_string(rules.any() ? 1 : 0) +
                                " -DRULE_SYMBOL_RUNS=" + std::to_string(rules.symbolRuns ? 1 : 0) +
                                " -DRULE_ADJACENT_BACKSLASHES=" + std::to_string(rules.adjacentBackslashes ? 1 : 0) +
                                " -DRULE_MAX_BACKSLASH_COUNT=" + std::to_string(rules.maxBackslashCount) +
                                " -DENTRY_SYMBOL_RUN=" + std::to_string(meet_in_the_middle_.tailEntry.symbolRun) +
                                " -DENTRY_BACKSLASHES=" + std::to_string(meet_in_the_middle_.tailEntry.backslashes) +
                                " -DENTRY_NEXT_IS_BACKSLASH=" + std::to_string(meet_in_the_middle_.tailEntry.nextIsBackslash ? 1 : 0);
    auto found = meetInTheMiddleKernels_.find(options);
    if (found != meetInTheMiddleKernels_.end())
        return found->second;
    CompiledKernel compiled;
    compiled.program = buildProgram(kMeetInTheMiddleKernelSource, options, "meet-in-the-middle");
    cl_int err = CL_SUCCESS;
    compiled.kernel = clCreateKernel(compiled.program, "searchMeetInTheMiddle", &err);
    CL_CHECK(err);
    size_t kernelMax = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(compiled.kernel, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMax), &kernelMax, nullptr));
    compiled.workGroupSize = std::min(kMeetInTheMiddleWorkGroupSize, kernelMax);
    return meetInTheMiddleKernels_.emplace(options, compiled).first->second;
}

MeetInTheMiddleOutcome OpenClBackend::runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) {
    MeetInTheMiddleOutcome outcome;
    if (batch.count == 0)
        return outcome;
    const CompiledKernel& compiled = meetInTheMiddleKernel();
    const uint64_t alphabetSize = meet_in_the_middle_.alphabet.size();
    const MeetInTheMiddleRows rows = meetInTheMiddleRowsOf(batch, alphabetSize);
    const cl_ulong firstRow = rows.firstRow, lastRow = rows.lastRow;
    const cl_uint firstRowStartK = rows.firstRowStartK, lastRowEndK = rows.lastRowEndK;
    const cl_uint meetingPointSeed1 = meet_in_the_middle_.meetingPoint.first, meetingPointSeed2 = meet_in_the_middle_.meetingPoint.second;
    const cl_int tailLength = batch.tailLength;
    const size_t local = compiled.workGroupSize;
    const size_t global = (size_t) ((rows.rowCount() + local - 1) / local) * local;

    // Launched again, with room for every hit, as long as it had more than it
    // had room for - which a real search, comparing the meeting point's whole
    // 64-bit state, practically never does.
    cl_uint counts[2] = {0, 0};
    for (;;) {
        cl_kernel kernel = compiled.kernel;
        int arg = 0;
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &meetInTheMiddleKeys_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(meetInTheMiddleKeyCount_), &meetInTheMiddleKeyCount_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &meetInTheMiddleFilter_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(meetInTheMiddleFilterShift_), &meetInTheMiddleFilterShift_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &meetInTheMiddleChars_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(meetingPointSeed1), &meetingPointSeed1));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(meetingPointSeed2), &meetingPointSeed2));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(tailLength), &tailLength));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(firstRow), &firstRow));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(lastRow), &lastRow));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(firstRowStartK), &firstRowStartK));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(lastRowEndK), &lastRowEndK));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &meetInTheMiddleCounts_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &meetInTheMiddleHits_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(meetInTheMiddleHitCapacity_), &meetInTheMiddleHitCapacity_));
        const LaunchWaiter::Clock::time_point launched = LaunchWaiter::Clock::now();
        CL_CHECK(clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr));
        CL_CHECK(clFlush(queue_)); // on its way to the device before this sleeps
        waiter_.wait(batch.count, launched, [&] {
            CL_CHECK(clEnqueueReadBuffer(queue_, meetInTheMiddleCounts_, CL_TRUE, 0, sizeof(counts), counts, 0, nullptr, nullptr));
        });
        if (counts[0] > batch.count || counts[1] > batch.count) {
            fprintf(stderr, "INTERNAL ERROR: a meet-in-the-middle launch of %llu tails looked up %u of them and found %u - exiting\n",
                    (unsigned long long) batch.count, counts[0], counts[1]);
            exit(1);
        }
        if (counts[1] <= meetInTheMiddleHitCapacity_)
            break;
        CL_CHECK(clReleaseMemObject(meetInTheMiddleHits_));
        meetInTheMiddleHitCapacity_ = std::max(counts[1], 2 * meetInTheMiddleHitCapacity_);
        meetInTheMiddleHits_ = createBuffer(CL_MEM_READ_WRITE, meetInTheMiddleHitCapacity_ * sizeof(cl_ulong));
        const cl_uint zeros[2] = {0, 0};
        CL_CHECK(clEnqueueWriteBuffer(queue_, meetInTheMiddleCounts_, CL_TRUE, 0, sizeof(zeros), zeros, 0, nullptr, nullptr));
    }

    outcome.probed = counts[0];
    if (counts[1] > 0) {
        std::vector<cl_ulong> hits(counts[1]);
        CL_CHECK(clEnqueueReadBuffer(queue_, meetInTheMiddleHits_, CL_TRUE, 0, counts[1] * sizeof(cl_ulong), hits.data(), 0, nullptr, nullptr));
        for (cl_ulong index : hits) {
            if (index < batch.start || index >= batch.start + batch.count) {
                fprintf(stderr, "INTERNAL ERROR: the meet-in-the-middle kernel reported the tail at %llu, which its batch doesn't have - exiting\n",
                        (unsigned long long) index);
                exit(1);
            }
            outcome.hits.push_back({batch.tailLength, index});
        }
    }
    if (counts[0] != 0 || counts[1] != 0) {
        static const cl_uint kZeros[2] = {0, 0};
        CL_CHECK(clEnqueueWriteBuffer(queue_, meetInTheMiddleCounts_, CL_FALSE, 0, sizeof(kZeros), kZeros, 0, nullptr, nullptr));
    }
    return outcome;
}

const OpenClBackend::CompiledKernel& OpenClBackend::dictionaryKernel() {
    const auto key = std::make_tuple(dictionarySuffixLen_, dictionaryHashes_, dictionaryHasTails_);
    auto found = dictionaryKernels_.find(key);
    if (found != dictionaryKernels_.end())
        return found->second;
    const std::string options = "-cl-std=CL1.2 -DSUFFIX_LEN=" + std::to_string(dictionarySuffixLen_) +
                                " -DHASHES=" + std::to_string((int) dictionaryHashes_) +
                                " -DWORDS_PER_THREAD=" + std::to_string(dictionaryWordsPerThread_) +
                                " -DTAILS=" + (dictionaryHasTails_ ? "1" : "0") +
                                " -DTAILS_PER_CELL=" + std::to_string(dictionaryTailsPerCell_) +
                                " -DPAIR_SLOTS=" + std::to_string(kDictionaryPairSlots) +
                                " -DFILTER_BITS=" + std::to_string(kDictionaryFilterBits) +
                                " -DFILTER_WORDS=" + std::to_string(kDictionaryFilterWords) +
                                " -DHASHA_MATCH_MASK=" + std::to_string(kHashAMatchMask) + "u" +
                                " -DBASENAME_MATCH_MASK=" + std::to_string(kBasenameMatchMask) + "u";
    CompiledKernel compiled;
    compiled.program = buildProgram(kDictionaryKernelSource, options, "dictionary");
    cl_int err = CL_SUCCESS;
    compiled.kernel = clCreateKernel(compiled.program, "searchDictionary", &err);
    CL_CHECK(err);
    size_t kernelMax = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(compiled.kernel, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMax), &kernelMax, nullptr));
    compiled.workGroupSize = std::min(dictionaryWorkGroupSize_, kernelMax);
    return dictionaryKernels_.emplace(key, compiled).first->second;
}

DictionaryOutcome OpenClBackend::runDictionaryBatches(const std::vector<DictionaryBatch>& batches) {
    DictionaryOutcome outcome;
    if (batches.empty())
        return outcome;
    const CompiledKernel& compiled = dictionaryKernel();
    const uint32_t cellsPerSegment = (uint32_t) (dictionaryWordsPerThread_ * compiled.workGroupSize);
    const uint64_t segments = planDictionaryLaunch(batches, cellsPerSegment, dictionaryHasTails_ ? dictionaryTailsPerCell_ : 1, dictionaryLaunch_);
    uint64_t candidates = 0;
    for (const DictionaryBatch& batch : batches)
        candidates += dictionaryBatchCandidates(batch);
    // The kernel counts a batch's cells in 32 bits.
    if (segments > UINT32_MAX || batches.size() > UINT32_MAX || candidates > UINT32_MAX) {
        // runDictionarySearch's calls are far smaller (dictionaryCandidatesPerCall).
        fprintf(stderr, "INTERNAL ERROR: a dictionary launch of %llu segments, %llu candidates - exiting\n", (unsigned long long) segments,
                (unsigned long long) candidates);
        exit(1);
    }
    if (dictionaryLaunch_.size() > dictionaryBatchesCapacity_) {
        if (dictionaryBatches_)
            CL_CHECK(clReleaseMemObject(dictionaryBatches_));
        dictionaryBatchesCapacity_ = std::max(dictionaryLaunch_.size(), 2 * dictionaryBatchesCapacity_);
        dictionaryBatches_ = createBuffer(CL_MEM_READ_ONLY, dictionaryBatchesCapacity_ * sizeof(DictionaryLaunchBatch));
    }
    // Not waited for: the queue runs it before the kernel, and
    // dictionaryLaunch_ isn't touched again until the launch is done.
    CL_CHECK(clEnqueueWriteBuffer(queue_, dictionaryBatches_, CL_FALSE, 0, dictionaryLaunch_.size() * sizeof(DictionaryLaunchBatch),
                                  dictionaryLaunch_.data(), 0, nullptr, nullptr));
    if (!dictionaryHashAHits_) {
        dictionaryHitCapacity_ = NAMEBREAK_DICTIONARY_HIT_CAPACITY;
        dictionaryHashAHits_ = createBuffer(CL_MEM_READ_WRITE, dictionaryHitCapacity_ * sizeof(DictionaryHit));
        dictionaryBasenameHits_ = createBuffer(CL_MEM_READ_WRITE, dictionaryHitCapacity_ * sizeof(DictionaryHit));
    }

    // Launched again, with room for every hit, as long as it had more than
    // it had room for - which a real search, with a 32-bit hashA and key,
    // practically never does.
    cl_int counts[2] = {0, 0};
    for (;;) {
        cl_kernel kernel = compiled.kernel;
        const cl_uint batchCount = (cl_uint) dictionaryLaunch_.size();
        int arg = 0;
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryBatches_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(batchCount), &batchCount));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryChars_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryEntries_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryPositions_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(dictionaryWordCount_), &dictionaryWordCount_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(dictionarySeparatorCount_), &dictionarySeparatorCount_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryTails_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryCryptKeys_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryFilters_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionarySuffixKeys_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(dictionaryTargetA_), &dictionaryTargetA_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(dictionaryBasenameKey_), &dictionaryBasenameKey_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryCounts_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryHashAHits_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dictionaryBasenameHits_));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(dictionaryHitCapacity_), &dictionaryHitCapacity_));
        // A work-group per segment.
        const size_t local = compiled.workGroupSize;
        const size_t global = (size_t) segments * local;
        const LaunchWaiter::Clock::time_point launched = LaunchWaiter::Clock::now();
        CL_CHECK(clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr));
        CL_CHECK(clFlush(queue_)); // on its way to the device before this sleeps
        waiter_.wait(candidates, launched, [&] {
            CL_CHECK(clEnqueueReadBuffer(queue_, dictionaryCounts_, CL_TRUE, 0, sizeof(counts), counts, 0, nullptr, nullptr));
        });
        if (counts[0] < 0 || counts[1] < 0 || (uint64_t) counts[0] > candidates || (uint64_t) counts[1] > candidates) {
            fprintf(stderr, "INTERNAL ERROR: a dictionary launch of %llu candidates reported %d and %d hits - exiting\n",
                    (unsigned long long) candidates, counts[0], counts[1]);
            exit(1);
        }
        const cl_uint needed = (cl_uint) std::max(counts[0], counts[1]);
        if (needed <= dictionaryHitCapacity_)
            break;
        CL_CHECK(clReleaseMemObject(dictionaryHashAHits_));
        CL_CHECK(clReleaseMemObject(dictionaryBasenameHits_));
        dictionaryHitCapacity_ = std::max(needed, 2 * dictionaryHitCapacity_);
        dictionaryHashAHits_ = createBuffer(CL_MEM_READ_WRITE, dictionaryHitCapacity_ * sizeof(DictionaryHit));
        dictionaryBasenameHits_ = createBuffer(CL_MEM_READ_WRITE, dictionaryHitCapacity_ * sizeof(DictionaryHit));
        const cl_int zeros[2] = {0, 0};
        CL_CHECK(clEnqueueWriteBuffer(queue_, dictionaryCounts_, CL_TRUE, 0, sizeof(zeros), zeros, 0, nullptr, nullptr));
    }
    if (counts[0] == 0 && counts[1] == 0) {
        dictionaryVerifier_.addHits(batches, {}, {}, outcome);
        return outcome;
    }

    // The exception: hits to bring back, and the counts to zero again.
    std::vector<DictionaryHit> hashAHits(counts[0]), basenameHits(counts[1]);
    if (counts[0] > 0)
        CL_CHECK(clEnqueueReadBuffer(queue_, dictionaryHashAHits_, CL_TRUE, 0, counts[0] * sizeof(DictionaryHit), hashAHits.data(), 0, nullptr,
                                     nullptr));
    if (counts[1] > 0)
        CL_CHECK(clEnqueueReadBuffer(queue_, dictionaryBasenameHits_, CL_TRUE, 0, counts[1] * sizeof(DictionaryHit), basenameHits.data(), 0,
                                     nullptr, nullptr));
    static const cl_int kZeros[2] = {0, 0};
    CL_CHECK(clEnqueueWriteBuffer(queue_, dictionaryCounts_, CL_FALSE, 0, sizeof(kZeros), kZeros, 0, nullptr, nullptr));
    for (const std::vector<DictionaryHit>* hits : {&hashAHits, &basenameHits}) {
        for (const DictionaryHit& hit : *hits) {
            if (!dictionaryHitInBatches(batches, hit)) {
                fprintf(stderr,
                        "INTERNAL ERROR: the dictionary kernel reported word %u, tail %u, pair %u of batch %u, which it doesn't have - exiting\n",
                        hit.word, hit.tail, hit.pair, hit.batch);
                exit(1);
            }
        }
    }
    dictionaryVerifier_.addHits(batches, hashAHits, basenameHits, outcome);
    return outcome;
}

} // namespace

GpuTuning openClTuningDefaults() {
    GpuTuning defaults;
    defaults.rowsPerThread = kRowsPerThreadDefault;
    defaults.batchesPerLaunch = kBatchesPerLaunchDefault;
    defaults.windowChars = kWindowCharsDefault;
    defaults.workGroupSize = kWorkGroupSizeDefault;
    defaults.dictionaryWordsPerThread = kDictionaryWordsPerThreadDefault;
    defaults.dictionaryWorkGroupSize = kDictionaryWorkGroupSizeDefault;
    defaults.dictionaryTailsPerCell = kDictionaryTailsPerCellDefault;
    return defaults;
}

std::unique_ptr<SearchBackend> makeOpenClBackend(const GpuTuning& tuning, std::string& error) {
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
        return std::make_unique<OpenClBackend>(device, context, queue, withDefaults(tuning, openClTuningDefaults()));
    }
    error = "no OpenCL GPU found";
    return nullptr;
}
