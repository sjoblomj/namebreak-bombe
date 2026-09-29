// Objective-C++, compiled with ARC (see CMakeLists.txt): Metal's API is
// Objective-C, while the rest of this backend is the same C++ as the others.

#include "backends/metal/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "backends/common/lowbits_filter.h"
#include "backends/common/row_batch.h"
#include "backends/metal/search_kernel.h" // generated from search.metal - see CMakeLists.txt
#include "engine/hash_match.h"
#include "engine/limits.h"

namespace {

// Threads per threadgroup, unless the pipeline allows fewer. Must be >= the
// alphabet size: the kernel fills its threadgroup tables one entry per thread.
constexpr NSUInteger kPreferredThreadgroupSize = 256;

// About how many rows of a row group one thread searches (ROWS_PER_THREAD in
// search.metal; see kChunksPerGroup in the CUDA backend). Not tuned yet: this
// is CUDA's value, two chunks per group at the usual alphabet sizes, rather
// than OpenCL's whole group per thread, so that a batch still has enough
// threads for the largest Apple GPUs. See PERFORMANCE.md.
constexpr int kRowsPerThread = rowsPerThreadOr(25);

// Must match RowArgs in search.metal.
struct RowArgs {
    uint32_t firstRow;
    uint32_t lastRow;
    int32_t firstRowStartK;
    int32_t lastRowEndK;
    uint32_t targetA;
    uint32_t seed1Start;
    uint32_t seed2Start;
};

class MetalBackend : public SearchBackend {
public:
    explicit MetalBackend(id<MTLDevice> device) : device_(device) {
        queue_ = [device_ newCommandQueue];
        // Shared storage: CPU and GPU see the same memory (unified on Apple
        // Silicon), so there's nothing to copy back and forth.
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        matchCount_ = [device_ newBufferWithLength:sizeof(int32_t) options:shared];
        matchIdx_ = [device_ newBufferWithLength:MAX_MATCHES * sizeof(uint64_t) options:shared];
        alphabetKey_ = [device_ newBufferWithLength:MAX_ALPHABET_SIZE * sizeof(uint32_t) options:shared];
        alphabetOrd_ = [device_ newBufferWithLength:MAX_ALPHABET_SIZE * sizeof(uint32_t) options:shared];
        suffixKey_ = [device_ newBufferWithLength:kMaxSuffixSize * sizeof(uint32_t) options:shared];
        suffixOrd_ = [device_ newBufferWithLength:kMaxSuffixSize * sizeof(uint32_t) options:shared];
        filterTable_ = [device_ newBufferWithLength:kLowBitsFilterEntries * sizeof(uint64_t) options:shared];
        if (!queue_ || !matchCount_ || !matchIdx_ || !alphabetKey_ || !alphabetOrd_ || !suffixKey_ || !suffixOrd_ || !filterTable_) {
            fprintf(stderr, "Metal: couldn't allocate the search's buffers\n");
            exit(1);
        }
    }

    const char* name() const override { return "metal"; }
    // The kernel is compiled for the alphabet size at hand, so any size works.
    std::vector<int> supportedAlphabetSizes() const override { return {}; }
    int windowChars() const override { return windowCharsOr(5); }
    // Row indices are 32-bit in the kernel: 50^5 < 2^32 <= 50^6.
    int maxTrailingLen() const override { return 6; }
    // A quarter of CUDA's batch: Apple GPUs are slower, and a batch should
    // stay well under a second for pause and abort to feel immediate.
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * rowsPerBatchOr(1u << 21); }

    void beginSearch(const SearchConstants& constants) override;
    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;
    void endSearch() override {}

private:
    id<MTLComputePipelineState> pipelineFor(int trailingLen);

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLBuffer> matchCount_, matchIdx_;
    id<MTLBuffer> alphabetKey_, alphabetOrd_, suffixKey_, suffixOrd_;
    // This search's lookup filter: kLowBitsFilterEntries entries, see
    // buildLowBitsFilterTable (backends/common/lowbits_filter.h).
    id<MTLBuffer> filterTable_;
    // Compiled once per (alphabet size, suffix length, trailing length) and
    // kept for the backend's lifetime - a coordinator client searches range
    // after range of the same shape, and each compile takes a moment.
    std::map<std::tuple<int, int, int>, id<MTLComputePipelineState>> pipelines_;
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
// match without any other sign. This ends the process, as the other Metal
// errors do - a coordinator range is then reassigned when its lease runs out.
[[noreturn]] void refuseFilterTable(const char* what, const std::string& detail) {
    fprintf(stderr, "INTERNAL ERROR: the lookup filter table %s (%s) - refusing to search with it\n", what, detail.c_str());
    exit(1);
}

void MetalBackend::beginSearch(const SearchConstants& constants) {
    if (!announcedDevice_) {
        printf("Metal device: %s\n", device_.name.UTF8String);
        announcedDevice_ = true;
    }
    verifier_.begin(constants);
    alphabetSize_ = (int) constants.alphabet.size();
    suffixLen_ = (int) constants.suffix.size();
    targetA_ = constants.targetHashA;

    auto* key = static_cast<uint32_t*>(alphabetKey_.contents);
    auto* ord = static_cast<uint32_t*>(alphabetOrd_.contents);
    std::fill(key, key + MAX_ALPHABET_SIZE, 0u);
    std::fill(ord, ord + MAX_ALPHABET_SIZE, 0u);
    for (int k = 0; k < alphabetSize_; ++k) {
        ord[k] = (unsigned char) constants.alphabet[k];
        key[k] = constants.cryptTable[0x100 + ord[k]];
    }
    auto* suffixKey = static_cast<uint32_t*>(suffixKey_.contents);
    auto* suffixOrd = static_cast<uint32_t*>(suffixOrd_.contents);
    std::fill(suffixKey, suffixKey + kMaxSuffixSize, 0u);
    std::fill(suffixOrd, suffixOrd + kMaxSuffixSize, 0u);
    for (int i = 0; i < suffixLen_; ++i) {
        suffixOrd[i] = (unsigned char) constants.suffix[i];
        suffixKey[i] = constants.cryptTable[0x100 + suffixOrd[i]];
    }
    // Establishes the "matchCount is 0 at launch" invariant runBatch keeps.
    *static_cast<int32_t*>(matchCount_.contents) = 0;

    // This search's lookup filter - it depends on the alphabet, the suffix and
    // the target, so it's built for every search. Checked against its
    // definition before it's used, and compared with what the GPU will read
    // after the copy.
    const std::vector<uint64_t> table = buildLowBitsFilterTable(constants);
    std::string error;
    if (!checkLowBitsFilterTable(table, constants, kFilterEntriesCheckedPerSearch, 2, std::random_device{}(), error))
        refuseFilterTable("failed its check", error);
    const size_t tableBytes = table.size() * sizeof(uint64_t);
    if (filterTable_.length < tableBytes)
        refuseFilterTable("doesn't fit its buffer", std::to_string(tableBytes) + " bytes");
    memcpy(filterTable_.contents, table.data(), tableBytes);
    if (memcmp(filterTable_.contents, table.data(), tableBytes) != 0)
        refuseFilterTable("in the GPU's buffer differs from the one built", std::to_string(tableBytes) + " bytes");
}

id<MTLComputePipelineState> MetalBackend::pipelineFor(int trailingLen) {
    const auto key = std::make_tuple(alphabetSize_, suffixLen_, trailingLen);
    auto found = pipelines_.find(key);
    if (found != pipelines_.end())
        return found->second;

    @autoreleasepool {
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.preprocessorMacros = @{
            @"ALPHABET_SIZE": @(alphabetSize_),
            @"SUFFIX_LEN": @(suffixLen_),
            @"TRAILING_LEN": @(trailingLen),
            @"MAX_MATCHES": @(MAX_MATCHES),
            @"HASHA_MATCH_MASK": @(kHashAMatchMask),
            @"FILTER_BITS": @(kLowBitsFilterBits),
            @"ROWS_PER_THREAD": @(kRowsPerThread),
        };
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:@(kSearchKernelSource) options:options error:&error];
        if (!library) {
            fprintf(stderr, "Metal: compiling the search kernel (alphabet %d, suffix %d, trailing %d) failed:\n%s\n", alphabetSize_,
                    suffixLen_, trailingLen, error.localizedDescription.UTF8String);
            exit(1);
        }
        id<MTLFunction> function = [library newFunctionWithName:@"searchRows"];
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "Metal: creating the search pipeline failed: %s\n", error.localizedDescription.UTF8String);
            exit(1);
        }
        if (std::min(kPreferredThreadgroupSize, pipeline.maxTotalThreadsPerThreadgroup) < (NSUInteger) alphabetSize_) {
            fprintf(stderr, "Metal: the GPU allows only %lu threads per threadgroup, fewer than the alphabet's %d characters\n",
                    (unsigned long) pipeline.maxTotalThreadsPerThreadgroup, alphabetSize_);
            exit(1);
        }
        pipelines_[key] = pipeline;
        return pipeline;
    }
}

BatchOutcome MetalBackend::runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) {
    const RowRange rows = rowRangeFor(start, count, alphabetSize_);
    if (rows.firstRow + rows.rowCount - 1 > UINT32_MAX || rows.rowCount > (1ull << 31)) {
        fprintf(stderr, "Batch too large for the kernel's 32-bit row index (rows %llu..%llu) - exiting\n",
                (unsigned long long) rows.firstRow, (unsigned long long) (rows.firstRow + rows.rowCount - 1));
        exit(1);
    }
    id<MTLComputePipelineState> pipeline = pipelineFor(trailingLen);

    RowArgs args;
    args.firstRow = (uint32_t) rows.firstRow;
    args.lastRow = (uint32_t) (rows.firstRow + rows.rowCount - 1);
    args.firstRowStartK = rows.firstRowStartK;
    args.lastRowEndK = rows.lastRowEndK;
    args.targetA = targetA_;
    args.seed1Start = params.seed1Start;
    args.seed2Start = params.seed2Start;

    @autoreleasepool {
        id<MTLCommandBuffer> commands = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&args length:sizeof(args) atIndex:0];
        [encoder setBuffer:alphabetKey_ offset:0 atIndex:1];
        [encoder setBuffer:alphabetOrd_ offset:0 atIndex:2];
        [encoder setBuffer:suffixKey_ offset:0 atIndex:3];
        [encoder setBuffer:suffixOrd_ offset:0 atIndex:4];
        [encoder setBuffer:matchCount_ offset:0 atIndex:5];
        [encoder setBuffer:matchIdx_ offset:0 atIndex:6];
        [encoder setBuffer:filterTable_ offset:0 atIndex:7];
        // A thread per chunk of every row group the batch touches (see
        // CHUNKS_PER_GROUP in search.metal), rounded up to whole threadgroups.
        // The kernel works out the chunks itself and covers all of them
        // whatever the grid's size, so this only spreads the work.
        const NSUInteger threadgroupSize = std::min(kPreferredThreadgroupSize, pipeline.maxTotalThreadsPerThreadgroup);
        const NSUInteger chunksPerGroup = (NSUInteger) (alphabetSize_ + kRowsPerThread - 1) / kRowsPerThread;
        const NSUInteger chunks = (NSUInteger) (args.lastRow / alphabetSize_ - args.firstRow / alphabetSize_ + 1) * chunksPerGroup;
        const NSUInteger threadgroups = (chunks + threadgroupSize - 1) / threadgroupSize;
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Metal: the search kernel failed: %s\n", commands.error.localizedDescription.UTF8String);
            exit(1);
        }
    }

    auto* matchCount = static_cast<int32_t*>(matchCount_.contents);
    BatchOutcome outcome;
    outcome.hitCount = *matchCount;
    if (outcome.hitCount == 0)
        return outcome;
    if (outcome.hitCount <= MAX_MATCHES) {
        const auto* recorded = static_cast<const uint64_t*>(matchIdx_.contents);
        std::vector<uint64_t> trailingIndices(recorded, recorded + outcome.hitCount);
        verifier_.addHits(trailingIndices, trailingLen, params, outcome);
    }
    // Past MAX_MATCHES nothing was recorded completely - the engine searches
    // the range again in halves. Either way, restore "matchCount is 0".
    *matchCount = 0;
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeMetalBackend(std::string& error) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        error = "no Metal device";
        return nullptr;
    }
    return std::make_unique<MetalBackend>(device);
}
