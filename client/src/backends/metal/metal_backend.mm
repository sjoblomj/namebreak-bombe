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

#include "backends/common/dictionary_batch.h"
#include "backends/common/lowbits_filter.h"
#include "backends/common/meet_in_the_middle_launch.h"
#include "backends/common/row_batch.h"
#include "backends/common/row_pruning.h"
#include "backends/metal/dictionary_kernel.h"         // generated from dictionary.metal - see CMakeLists.txt
#include "backends/metal/meet_in_the_middle_kernel.h" // generated from meet_in_the_middle.metal
#include "backends/metal/search_kernel.h"             // generated from search.metal
#include "engine/candidate.h"
#include "engine/dictionary_search.h"
#include "engine/hash_match.h"
#include "engine/limits.h"

namespace {

// Threads per threadgroup, unless the pipeline allows fewer. Must be >= the
// alphabet size: the kernel fills its threadgroup tables one entry per thread.
// This and the other k*Default values below are what the backend uses where
// config.conf's [metal] says nothing (GpuTuning, see backends/gpu_tuning.h).
constexpr int kThreadgroupSizeDefault = 256;

// About how many rows of a row group one thread searches (ROWS_PER_THREAD in
// search.metal; see kChunksPerGroup in the CUDA backend). Not tuned yet: this
// is CUDA's value, two chunks per group at the usual alphabet sizes, rather
// than OpenCL's whole group per thread, so that a batch still has enough
// threads for the largest Apple GPUs. See PERFORMANCE.md.
constexpr int kRowsPerThreadDefault = rowsPerThreadOr(25);
// How many trailing characters the GPU enumerates (SearchBackend::windowChars).
constexpr int kWindowCharsDefault = windowCharsOr(5);

// A dictionary search's launches (dictionary.metal): how many candidates one
// covers at most (SearchBackend::dictionaryCandidatesPerCall) - a quarter
// of CUDA's, as the row search's batches are - how many cells (words, with
// tails) of a batch one thread hashes, and the threadgroup's size, unless
// the pipeline allows fewer. A segment of a launch is the last two's
// product. And with tails, how many a cell has (see DictionaryLaunchBatch).
// Not tuned yet: the CUDA backend's values (see its tuning.h).
constexpr uint64_t kDictionaryCandidatesPerLaunch = 1ull << 26;
constexpr int kDictionaryWordsPerThreadDefault = dictionaryWordsPerThreadOr(32);
constexpr int kDictionaryThreadgroupSizeDefault = dictionaryThreadsPerBlockOr(256);
constexpr int kDictionaryTailsPerCellDefault = dictionaryTailsPerCellOr(32);

// A meet-in-the-middle search's launches (meet_in_the_middle.metal): about
// how many tails one rewinds, how many hits it has room for at first (see
// backends/common/meet_in_the_middle_launch.h) and the threadgroup's size,
// unless the pipeline allows fewer. Not tuned yet: the OpenCL backend's
// values.
constexpr uint64_t kMeetInTheMiddleTailsPerLaunch = NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_LAUNCH;
constexpr uint32_t kMeetInTheMiddleHitCapacity = NAMEBREAK_MEET_IN_THE_MIDDLE_HIT_CAPACITY;
constexpr NSUInteger kMeetInTheMiddleThreadgroupSize = 256;

// Must match MeetInTheMiddleArgs in meet_in_the_middle.metal.
struct MeetInTheMiddleArgs {
    uint64_t keyCount;
    uint64_t firstRow;
    uint64_t lastRow;
    int32_t filterShift;
    uint32_t meetingPointSeed1;
    uint32_t meetingPointSeed2;
    int32_t tailLength;
    uint32_t firstRowStartK;
    uint32_t lastRowEndK;
    uint32_t capacity;
    uint32_t unused;
};
static_assert(sizeof(MeetInTheMiddleArgs) == 56, "MeetInTheMiddleArgs must match meet_in_the_middle.metal's");

// Must match DictionaryArgs in dictionary.metal.
struct DictionaryArgs {
    uint32_t batchCount;
    uint32_t wordCount;
    uint32_t targetA;
    uint32_t basenameKey;
    uint32_t capacity;
    uint32_t separatorCount;
};

// Must match RowArgs in search.metal.
struct RowArgs {
    uint32_t firstRow;
    uint32_t lastRow;
    int32_t firstRowStartK;
    int32_t lastRowEndK;
    uint32_t targetA;
    uint32_t seed1Start;
    uint32_t seed2Start;
    uint32_t groupsOffset;
    uint32_t groupCount;
};

class MetalBackend : public SearchBackend {
public:
    // `tuning` has every value it takes set (withDefaults).
    MetalBackend(id<MTLDevice> device, const GpuTuning& tuning)
        : device_(device), tuning_(tuning), rowsPerThread_(*tuning.rowsPerThread), windowChars_(*tuning.windowChars),
          threadgroupSize_((NSUInteger) *tuning.workGroupSize), dictionaryWordsPerThread_(*tuning.dictionaryWordsPerThread),
          dictionaryThreadgroupSize_((NSUInteger) *tuning.dictionaryWorkGroupSize), dictionaryTailsPerCell_(*tuning.dictionaryTailsPerCell) {
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
        rowMasks_ = [device_ newBufferWithLength:kRowFlagCount * sizeof(uint64_t) options:shared];
        inserted_ = [device_ newBufferWithLength:2 * 2 * kMaxInsertLen * sizeof(uint32_t) options:shared];
        // Bound to the kernel even when it doesn't read it (LISTED 0), so it
        // always exists; grown when a search's lists need more.
        groups_ = [device_ newBufferWithLength:sizeof(uint32_t) options:shared];
        if (!queue_ || !matchCount_ || !matchIdx_ || !alphabetKey_ || !alphabetOrd_ || !suffixKey_ || !suffixOrd_ || !filterTable_ ||
            !rowMasks_ || !groups_ || !inserted_) {
            fprintf(stderr, "Metal: couldn't allocate the search's buffers\n");
            exit(1);
        }
    }

    const char* name() const override { return "metal"; }
    std::string deviceName() const override { return device_.name.UTF8String; }
    int windowChars() const override { return windowChars_; }
    // Row indices are 32-bit in the kernel: 63^5 < 2^32 <= 63^6.
    int maxTrailingLen() const override { return 6; }
    // A quarter of CUDA's batch: Apple GPUs are slower, and a batch should
    // stay well under a second for pause and abort to feel immediate.
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * rowsPerBatchOr(1u << 21); }

    void beginSearch(const SearchConstants& constants) override;
    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;
    void endSearch() override {}

    // A dictionary search: one launch of dictionary.metal's kernel per call.
    bool supportsDictionary() const override { return true; }
    uint64_t dictionaryCandidatesPerCall() const override { return kDictionaryCandidatesPerLaunch; }
    void beginDictionarySearch(const DictionaryConstants& constants) override;
    DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches) override;
    void endDictionarySearch() override;

    // A meet-in-the-middle search: one launch of meet_in_the_middle.metal's
    // kernel per call.
    bool supportsMeetInTheMiddle() const override { return true; }
    uint64_t meetInTheMiddleTailsPerCall() const override { return kMeetInTheMiddleTailsPerLaunch; }
    void beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) override;
    MeetInTheMiddleOutcome runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) override;
    void endMeetInTheMiddleSearch() override;

private:
    id<MTLComputePipelineState> pipelineFor(int trailingLen, bool listed);
    id<MTLComputePipelineState> dictionaryPipeline();
    id<MTLComputePipelineState> meetInTheMiddlePipeline();
    void uploadGroups();
    // A shared-storage buffer holding `bytes` bytes (at least one) - with
    // `data` copied in, if given. Ends the process if it can't be made.
    id<MTLBuffer> sharedBuffer(size_t bytes, const void* data = nullptr);

    // Prints the device and the launch geometry, before the first search
    // of any kind.
    void announceDevice();

    id<MTLDevice> device_;
    // The launch geometry (see GpuTuning), every value it takes set.
    GpuTuning tuning_;
    int rowsPerThread_;
    int windowChars_;
    NSUInteger threadgroupSize_;
    int dictionaryWordsPerThread_;
    NSUInteger dictionaryThreadgroupSize_;
    int dictionaryTailsPerCell_;
    id<MTLCommandQueue> queue_;
    id<MTLBuffer> matchCount_, matchIdx_;
    id<MTLBuffer> alphabetKey_, alphabetOrd_, suffixKey_, suffixOrd_;
    // This search's lookup filter: kLowBitsFilterEntries entries, see
    // buildLowBitsFilterTable (backends/common/lowbits_filter.h).
    id<MTLBuffer> filterTable_;
    // The text inserted into the trailing part: each character's key, then
    // its value (see search.metal's `inserted`), and where each insertion is
    // - compiled into the kernel.
    id<MTLBuffer> inserted_;
    InsertLayout insertLayout_;
    // Compiled once per (alphabet size, suffix length, trailing length,
    // LISTED, where the inserted text goes) and kept for the backend's
    // lifetime - a coordinator client searches range after range of the same
    // shape, and each compile takes a moment.
    std::map<std::tuple<int, int, int, bool, std::vector<int>>, id<MTLComputePipelineState>> pipelines_;
    bool announcedDevice_ = false;

    HitVerifier verifier_;
    int alphabetSize_ = 0;
    int suffixLen_ = 0;
    uint32_t targetA_ = 0;

    // The row groups a batch searches when the whole candidate is pruned
    // (see backends/common/row_pruning.h), and their copy in groups_: the
    // first groupsUploaded_ entries of rowPruning_.arena(), of generation
    // groupsGeneration_. Kept from one search to the next, as the lists are.
    // Only then does the kernel walk lists (LISTED in search.metal) - the
    // CUDA and OpenCL kernels walking every group measured 5-6% faster than
    // walking a list of them all.
    RowPruning rowPruning_;
    bool listed_ = false;
    id<MTLBuffer> rowMasks_, groups_;
    size_t groupsUploaded_ = 0;
    uint64_t groupsGeneration_ = 0;

    // A dictionary search's: the word list, the crypt table's keys, the
    // suffix filters and the suffix's keys (backends/common/
    // dictionary_batch.h) - made by beginDictionarySearch and let go by
    // endDictionarySearch - and what the kernel is compiled with.
    DictionaryHitVerifier dictionaryVerifier_;
    id<MTLBuffer> dictionaryChars_, dictionaryEntries_, dictionaryPositions_;
    id<MTLBuffer> dictionaryTails_; // dictionaryTailTable
    bool dictionaryHasTails_ = false;
    id<MTLBuffer> dictionaryCryptKeys_, dictionaryFilters_, dictionarySuffixKeys_;
    uint32_t dictionaryWordCount_ = 0;
    uint32_t dictionarySeparatorCount_ = 0;
    int dictionarySuffixLen_ = 0;
    DictionaryHashes dictionaryHashes_ = DictionaryHashes::HashA;
    uint32_t dictionaryTargetA_ = 0;
    uint32_t dictionaryBasenameKey_ = 0;
    // Compiled once per (suffix length, hashes, tails or not) and kept, as
    // the row search's pipelines are.
    std::map<std::tuple<int, DictionaryHashes, bool>, id<MTLComputePipelineState>> dictionaryPipelines_;
    // Kept from one search to the next, grown as a launch needs: its batches
    // (DictionaryLaunchBatch), the hit counts - 0 at every launch - and the
    // hits, room for dictionaryHitCapacity_ of each kind.
    std::vector<DictionaryLaunchBatch> dictionaryLaunch_;
    id<MTLBuffer> dictionaryBatches_, dictionaryCounts_, dictionaryHashAHits_, dictionaryBasenameHits_;
    uint32_t dictionaryHitCapacity_ = 0;

    // A meet-in-the-middle search's: the engine's table and the alphabet as
    // the kernel reads it - made by beginMeetInTheMiddleSearch and let go by
    // endMeetInTheMiddleSearch - and what the kernel is compiled with.
    MeetInTheMiddleConstants meet_in_the_middle_ = {};
    id<MTLBuffer> meetInTheMiddleKeys_, meetInTheMiddleFilter_, meetInTheMiddleChars_;
    // Compiled once per alphabet size, rules and the rules' state to the
    // directory's right, and kept, as the other searches' pipelines are.
    std::map<std::vector<int>, id<MTLComputePipelineState>> meetInTheMiddlePipelines_;
    // Kept from one search to the next: the counts - tails looked up and
    // hits, 0 at every launch - and the hits, room for
    // meetInTheMiddleHitCapacity_ of them.
    id<MTLBuffer> meetInTheMiddleCounts_, meetInTheMiddleHits_;
    uint32_t meetInTheMiddleHitCapacity_ = 0;
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

void MetalBackend::announceDevice() {
    if (announcedDevice_)
        return;
    printf("Metal device: %s\n", device_.name.UTF8String);
    printf("Metal launches: %s\n", describeGpuTuning("metal", tuning_).c_str());
    announcedDevice_ = true;
}

void MetalBackend::beginSearch(const SearchConstants& constants) {
    announceDevice();
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
    insertLayout_ = insertLayoutFor(constants);
    auto* inserted = static_cast<uint32_t*>(inserted_.contents);
    std::fill(inserted, inserted + 2 * 2 * kMaxInsertLen, 0u);
    for (size_t i = 0; i < insertLayout_.key.size(); ++i) {
        inserted[2 * i] = insertLayout_.key[i];
        inserted[2 * i + 1] = insertLayout_.ord[i];
    }
    // Establishes the "matchCount is 0 at launch" invariant runBatch keeps.
    *static_cast<int32_t*>(matchCount_.contents) = 0;
    listed_ = constants.trailingRules.any();
    if (listed_) {
        rowPruning_.begin(constants);
        memcpy(rowMasks_.contents, rowPruning_.rowMasks(), kRowFlagCount * sizeof(uint64_t));
    }

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

// Brings groups_ up to date with rowPruning_.arena(), which only grows within
// a generation: only what was added since is copied, unless the buffer is too
// small or the lists were cleared. Nothing is running on the GPU here
// (runBatch waits for its command buffer before it returns).
void MetalBackend::uploadGroups() {
    const std::vector<uint32_t>& arena = rowPruning_.arena();
    if (rowPruning_.generation() != groupsGeneration_) {
        groupsGeneration_ = rowPruning_.generation();
        groupsUploaded_ = 0;
    }
    const size_t capacity = groups_.length / sizeof(uint32_t);
    if (arena.size() > capacity) {
        groups_ = [device_ newBufferWithLength:std::max(arena.size(), 2 * capacity) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!groups_) {
            fprintf(stderr, "Metal: couldn't allocate the row groups' lists (%zu entries)\n", arena.size());
            exit(1);
        }
        groupsUploaded_ = 0;
    }
    if (arena.size() > groupsUploaded_) {
        memcpy(static_cast<uint32_t*>(groups_.contents) + groupsUploaded_, arena.data() + groupsUploaded_,
               (arena.size() - groupsUploaded_) * sizeof(uint32_t));
        groupsUploaded_ = arena.size();
    }
}

id<MTLComputePipelineState> MetalBackend::pipelineFor(int trailingLen, bool listed) {
    const InsertLayout& l = insertLayout_;
    const std::vector<int> insertKey = {l.groupCharsAfter[0], l.groupStart[0], l.groupLen[0], l.groupCharsAfter[1],
                                        l.groupStart[1],      l.groupLen[1],   l.rowStart,    l.rowLen};
    const auto key = std::make_tuple(alphabetSize_, suffixLen_, trailingLen, listed, insertKey);
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
            @"ROWS_PER_THREAD": @(rowsPerThread_),
            @"ROW_FLAG_BITS": @(kRowFlagBits),
            @"LISTED": @(listed ? 1 : 0),
            @"GROUP_INSERT0_AFTER": @(l.groupCharsAfter[0]),
            @"GROUP_INSERT0_START": @(l.groupStart[0]),
            @"GROUP_INSERT0_LEN": @(l.groupLen[0]),
            @"GROUP_INSERT1_AFTER": @(l.groupCharsAfter[1]),
            @"GROUP_INSERT1_START": @(l.groupStart[1]),
            @"GROUP_INSERT1_LEN": @(l.groupLen[1]),
            @"ROW_INSERT_START": @(l.rowStart),
            @"ROW_INSERT_LEN": @(l.rowLen),
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
        if (std::min(threadgroupSize_, pipeline.maxTotalThreadsPerThreadgroup) < (NSUInteger) alphabetSize_) {
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
    RowArgs args;
    args.firstRow = (uint32_t) rows.firstRow;
    args.lastRow = (uint32_t) (rows.firstRow + rows.rowCount - 1);
    args.firstRowStartK = rows.firstRowStartK;
    args.lastRowEndK = rows.lastRowEndK;
    args.targetA = targetA_;
    args.seed1Start = params.seed1Start;
    args.seed2Start = params.seed2Start;
    // The row groups to search: all those the rows touch, or with the whole
    // candidate pruned, those of them in the batch's list.
    NSUInteger groupCount = args.lastRow / alphabetSize_ - args.firstRow / alphabetSize_ + 1;
    args.groupsOffset = 0;
    args.groupCount = 0;
    if (listed_) {
        const RowPruning::Slice groups =
            rowPruning_.groupsFor(trailingLen, params.pruneEntry, args.firstRow / alphabetSize_, args.lastRow / alphabetSize_);
        args.groupsOffset = groups.offset;
        args.groupCount = groups.count;
        groupCount = groups.count;
        if (groupCount == 0)
            return BatchOutcome(); // every row pruned: nothing to launch
        uploadGroups();
    }
    id<MTLComputePipelineState> pipeline = pipelineFor(trailingLen, listed_);

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
        [encoder setBuffer:groups_ offset:0 atIndex:8];
        [encoder setBuffer:rowMasks_ offset:0 atIndex:9];
        [encoder setBuffer:inserted_ offset:0 atIndex:10];
        // A thread per chunk of every row group the batch searches (see
        // CHUNKS_PER_GROUP in search.metal), rounded up to whole threadgroups.
        // The kernel works out the chunks itself and covers all of them
        // whatever the grid's size, so this only spreads the work.
        const NSUInteger threadgroupSize = std::min(threadgroupSize_, pipeline.maxTotalThreadsPerThreadgroup);
        const NSUInteger chunksPerGroup = (NSUInteger) (alphabetSize_ + rowsPerThread_ - 1) / rowsPerThread_;
        const NSUInteger chunks = groupCount * chunksPerGroup;
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


id<MTLBuffer> MetalBackend::sharedBuffer(size_t bytes, const void* data) {
    id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<size_t>(1, bytes) options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "Metal: couldn't allocate a buffer of %zu bytes\n", bytes);
        exit(1);
    }
    if (data && bytes)
        memcpy(buffer.contents, data, bytes);
    return buffer;
}

void MetalBackend::beginDictionarySearch(const DictionaryConstants& constants) {
    announceDevice();
    dictionaryVerifier_.begin(constants);
    dictionaryHashes_ = dictionaryVerifier_.hashes();
    dictionaryTargetA_ = constants.targetHashA;
    dictionaryBasenameKey_ = constants.basenameKey;
    dictionarySuffixLen_ = (int) constants.suffix.size();
    dictionaryWordCount_ = (uint32_t) constants.words.size();
    dictionarySeparatorCount_ = (uint32_t) constants.separators.size();

    DictionaryWordTable table;
    if (!makeDictionaryWordTable(constants.words, constants.separators, table)) {
        // runDictionarySearch's word lists are far smaller.
        fprintf(stderr, "INTERNAL ERROR: the word list is too long for the GPU's 32-bit offsets - exiting\n");
        exit(1);
    }
    dictionaryChars_ = sharedBuffer(table.chars.size() * sizeof(uint32_t), table.chars.data());
    dictionaryEntries_ = sharedBuffer(table.entries.size() * sizeof(DictionaryWordEntry), table.entries.data());
    dictionaryPositions_ = sharedBuffer(table.positions.size() * sizeof(uint32_t), table.positions.data());
    // The tails - only "" without any, which the kernel built without TAILS
    // doesn't read.
    dictionaryHasTails_ = constants.tails != std::vector<std::string>{""};
    const std::vector<uint32_t> tailTable = dictionaryTailTable(constants.tails);
    dictionaryTails_ = sharedBuffer(tailTable.size() * sizeof(uint32_t), tailTable.data());

    std::vector<uint32_t> cryptKeys(constants.cryptTable + kHashAOffset, constants.cryptTable + kHashAOffset + 256);
    cryptKeys.insert(cryptKeys.end(), constants.cryptTable + kFileKeyOffset, constants.cryptTable + kFileKeyOffset + 256);
    dictionaryCryptKeys_ = sharedBuffer(cryptKeys.size() * sizeof(uint32_t), cryptKeys.data());
    // The suffix filters, hashA's and the basename hash's - each checked in
    // full before it's used, as the row search's filter table is (see
    // beginSearch): a wrong one could drop a match without any other sign.
    std::vector<uint32_t> filters;
    for (bool basename : {false, true}) {
        const int offset = basename ? kFileKeyOffset : kHashAOffset;
        const uint32_t target = basename ? constants.basenameKey : constants.targetHashA, mask = dictionaryFilterMask(basename);
        const std::vector<uint32_t> filter = buildDictionarySuffixFilter(constants.suffix, constants.cryptTable, offset, target, mask);
        std::string error;
        if (!checkDictionarySuffixFilter(filter, constants.suffix, constants.cryptTable, offset, target, mask, std::random_device{}(), error))
            refuseFilterTable("of the dictionary search's suffix failed its check", error);
        filters.insert(filters.end(), filter.begin(), filter.end());
    }
    dictionaryFilters_ = sharedBuffer(filters.size() * sizeof(uint32_t), filters.data());
    const std::vector<uint32_t> suffixKeys = dictionarySuffixKeys(constants.suffix, constants.cryptTable);
    dictionarySuffixKeys_ = sharedBuffer(suffixKeys.size() * sizeof(uint32_t), suffixKeys.data());

    // The counts are 0 at every launch: zeroed here, and after every launch
    // that had hits (see runDictionaryBatches).
    if (!dictionaryCounts_)
        dictionaryCounts_ = sharedBuffer(2 * sizeof(int32_t));
    memset(dictionaryCounts_.contents, 0, 2 * sizeof(int32_t));
}

void MetalBackend::endDictionarySearch() {
    dictionaryChars_ = nil;
    dictionaryEntries_ = nil;
    dictionaryPositions_ = nil;
    dictionaryTails_ = nil;
    dictionaryCryptKeys_ = nil;
    dictionaryFilters_ = nil;
    dictionarySuffixKeys_ = nil;
}

id<MTLComputePipelineState> MetalBackend::dictionaryPipeline() {
    const auto key = std::make_tuple(dictionarySuffixLen_, dictionaryHashes_, dictionaryHasTails_);
    auto found = dictionaryPipelines_.find(key);
    if (found != dictionaryPipelines_.end())
        return found->second;

    @autoreleasepool {
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.preprocessorMacros = @{
            @"SUFFIX_LEN": @(dictionarySuffixLen_),
            @"HASHES": @((int) dictionaryHashes_),
            @"WORDS_PER_THREAD": @(dictionaryWordsPerThread_),
            @"TAILS": @(dictionaryHasTails_ ? 1 : 0),
            @"TAILS_PER_CELL": @(dictionaryTailsPerCell_),
            @"PAIR_SLOTS": @(kDictionaryPairSlots),
            @"FILTER_BITS": @(kDictionaryFilterBits),
            @"FILTER_WORDS": @(kDictionaryFilterWords),
            @"HASHA_MATCH_MASK": @(kHashAMatchMask),
            @"BASENAME_MATCH_MASK": @(kBasenameMatchMask),
        };
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:@(kDictionaryKernelSource) options:options error:&error];
        if (!library) {
            fprintf(stderr, "Metal: compiling the dictionary kernel (suffix %d) failed:\n%s\n", dictionarySuffixLen_,
                    error.localizedDescription.UTF8String);
            exit(1);
        }
        id<MTLFunction> function = [library newFunctionWithName:@"searchDictionary"];
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "Metal: creating the dictionary pipeline failed: %s\n", error.localizedDescription.UTF8String);
            exit(1);
        }
        dictionaryPipelines_[key] = pipeline;
        return pipeline;
    }
}

DictionaryOutcome MetalBackend::runDictionaryBatches(const std::vector<DictionaryBatch>& batches) {
    DictionaryOutcome outcome;
    if (batches.empty())
        return outcome;
    id<MTLComputePipelineState> pipeline = dictionaryPipeline();
    const NSUInteger threadgroupSize = std::min(dictionaryThreadgroupSize_, pipeline.maxTotalThreadsPerThreadgroup);
    const uint32_t cellsPerSegment = (uint32_t) (dictionaryWordsPerThread_ * threadgroupSize);
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
    const size_t batchBytes = dictionaryLaunch_.size() * sizeof(DictionaryLaunchBatch);
    if (!dictionaryBatches_ || dictionaryBatches_.length < batchBytes)
        dictionaryBatches_ = sharedBuffer(std::max<size_t>(batchBytes, dictionaryBatches_ ? 2 * dictionaryBatches_.length : 0));
    memcpy(dictionaryBatches_.contents, dictionaryLaunch_.data(), batchBytes);
    if (!dictionaryHashAHits_) {
        dictionaryHitCapacity_ = NAMEBREAK_DICTIONARY_HIT_CAPACITY;
        dictionaryHashAHits_ = sharedBuffer(dictionaryHitCapacity_ * sizeof(DictionaryHit));
        dictionaryBasenameHits_ = sharedBuffer(dictionaryHitCapacity_ * sizeof(DictionaryHit));
    }

    // Launched again, with room for every hit, as long as it had more than
    // it had room for - which a real search, with a 32-bit hashA and key,
    // practically never does.
    auto* counts = static_cast<int32_t*>(dictionaryCounts_.contents);
    for (;;) {
        DictionaryArgs args;
        args.batchCount = (uint32_t) dictionaryLaunch_.size();
        args.wordCount = dictionaryWordCount_;
        args.targetA = dictionaryTargetA_;
        args.basenameKey = dictionaryBasenameKey_;
        args.capacity = dictionaryHitCapacity_;
        args.separatorCount = dictionarySeparatorCount_;
        @autoreleasepool {
            id<MTLCommandBuffer> commands = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&args length:sizeof(args) atIndex:0];
            [encoder setBuffer:dictionaryBatches_ offset:0 atIndex:1];
            [encoder setBuffer:dictionaryChars_ offset:0 atIndex:2];
            [encoder setBuffer:dictionaryEntries_ offset:0 atIndex:3];
            [encoder setBuffer:dictionaryPositions_ offset:0 atIndex:4];
            [encoder setBuffer:dictionaryCryptKeys_ offset:0 atIndex:5];
            [encoder setBuffer:dictionaryFilters_ offset:0 atIndex:6];
            [encoder setBuffer:dictionarySuffixKeys_ offset:0 atIndex:7];
            [encoder setBuffer:dictionaryCounts_ offset:0 atIndex:8];
            [encoder setBuffer:dictionaryHashAHits_ offset:0 atIndex:9];
            [encoder setBuffer:dictionaryBasenameHits_ offset:0 atIndex:10];
            [encoder setBuffer:dictionaryTails_ offset:0 atIndex:11];
            // A threadgroup per segment.
            [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger) segments, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "Metal: the dictionary kernel failed: %s\n", commands.error.localizedDescription.UTF8String);
                exit(1);
            }
        }
        if (counts[0] < 0 || counts[1] < 0 || (uint64_t) counts[0] > candidates || (uint64_t) counts[1] > candidates) {
            fprintf(stderr, "INTERNAL ERROR: a dictionary launch of %llu candidates reported %d and %d hits - exiting\n",
                    (unsigned long long) candidates, counts[0], counts[1]);
            exit(1);
        }
        const uint32_t needed = (uint32_t) std::max(counts[0], counts[1]);
        if (needed <= dictionaryHitCapacity_)
            break;
        dictionaryHitCapacity_ = std::max(needed, 2 * dictionaryHitCapacity_);
        dictionaryHashAHits_ = sharedBuffer(dictionaryHitCapacity_ * sizeof(DictionaryHit));
        dictionaryBasenameHits_ = sharedBuffer(dictionaryHitCapacity_ * sizeof(DictionaryHit));
        counts[0] = counts[1] = 0;
    }

    const auto* hashAHits = static_cast<const DictionaryHit*>(dictionaryHashAHits_.contents);
    const auto* basenameHits = static_cast<const DictionaryHit*>(dictionaryBasenameHits_.contents);
    const std::vector<DictionaryHit> hashA(hashAHits, hashAHits + counts[0]), basename(basenameHits, basenameHits + counts[1]);
    counts[0] = counts[1] = 0;
    for (const std::vector<DictionaryHit>* hits : {&hashA, &basename}) {
        for (const DictionaryHit& hit : *hits) {
            if (!dictionaryHitInBatches(batches, hit)) {
                fprintf(stderr,
                        "INTERNAL ERROR: the dictionary kernel reported word %u, tail %u, pair %u of batch %u, which it doesn't have - exiting\n",
                        hit.word, hit.tail, hit.pair, hit.batch);
                exit(1);
            }
        }
    }
    dictionaryVerifier_.addHits(batches, hashA, basename, outcome);
    return outcome;
}

void MetalBackend::beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) {
    announceDevice();
    endMeetInTheMiddleSearch();
    meet_in_the_middle_ = constants;
    const MeetInTheMiddleTable& table = *constants.table;

    // The alphabet, as the kernel reads it: each character's hashA crypt
    // table key, its own value, and what the pruning rules make of it.
    std::vector<uint32_t> chars(4 * std::max<size_t>(1, constants.alphabet.size()), 0u);
    for (size_t k = 0; k < constants.alphabet.size(); ++k) {
        const unsigned char ch = (unsigned char) constants.alphabet[k];
        chars[4 * k] = constants.cryptTable[kHashAOffset + ch];
        chars[4 * k + 1] = ch;
        chars[4 * k + 2] = (isRunSymbol_CPU((char) ch) ? 1u : 0u) | (ch == '\\' ? 2u : 0u);
    }
    meetInTheMiddleChars_ = sharedBuffer(chars.size() * sizeof(uint32_t), chars.data());

    // The table: the one thing a meet-in-the-middle search has that can be
    // too big for the GPU - the engine sizes it to the memory it was given,
    // not to what Metal lets a buffer have, or a GPU work with.
    const size_t keyBytes = table.size() * sizeof(uint64_t);
    const size_t filterBytes = (size_t) table.filterWords() * sizeof(uint64_t);
    const uint64_t workingSet = device_.recommendedMaxWorkingSetSize;
    if (std::max(keyBytes, filterBytes) > device_.maxBufferLength || keyBytes + filterBytes > workingSet) {
        fprintf(stderr,
                "The meet-in-the-middle search's table needs %zu MB on the Metal device, which works with %llu MB at most\n"
                "and %llu MB in one buffer.\n"
                "Give the table less memory (config.conf's table_memory, a character shorter is %u times smaller),\n"
                "or search on the cpu backend (--backend cpu), which uses the machine's own memory.\n",
                (keyBytes + filterBytes) / (1024 * 1024), (unsigned long long) (workingSet / (1024 * 1024)),
                (unsigned long long) (device_.maxBufferLength / (1024 * 1024)), (unsigned) constants.alphabet.size());
        exit(1);
    }
    meetInTheMiddleKeys_ = sharedBuffer(keyBytes, table.keys());
    meetInTheMiddleFilter_ = sharedBuffer(filterBytes, table.filter());

    // The counts are 0 at every launch: zeroed here, and after every launch
    // that had anything to report (see runMeetInTheMiddleBatch).
    if (!meetInTheMiddleCounts_)
        meetInTheMiddleCounts_ = sharedBuffer(2 * sizeof(uint32_t));
    memset(meetInTheMiddleCounts_.contents, 0, 2 * sizeof(uint32_t));
    if (!meetInTheMiddleHits_) {
        meetInTheMiddleHitCapacity_ = kMeetInTheMiddleHitCapacity;
        meetInTheMiddleHits_ = sharedBuffer(meetInTheMiddleHitCapacity_ * sizeof(uint64_t));
    }
}

void MetalBackend::endMeetInTheMiddleSearch() {
    meetInTheMiddleKeys_ = nil;
    meetInTheMiddleFilter_ = nil;
    meetInTheMiddleChars_ = nil;
}

id<MTLComputePipelineState> MetalBackend::meetInTheMiddlePipeline() {
    // The alphabet and the rules are compiled in: they don't change within a
    // search, and the alphabet's size makes the kernel's divisions by it
    // multiplications.
    const PruneRules& rules = meet_in_the_middle_.rules;
    const TailPruneState& entry = meet_in_the_middle_.tailEntry;
    const std::vector<int> key = {(int) meet_in_the_middle_.alphabet.size(),
                                  rules.any() ? 1 : 0,
                                  rules.symbolRuns ? 1 : 0,
                                  rules.adjacentBackslashes ? 1 : 0,
                                  rules.maxBackslashCount,
                                  entry.symbolRun,
                                  entry.backslashes,
                                  entry.nextIsBackslash ? 1 : 0};
    auto found = meetInTheMiddlePipelines_.find(key);
    if (found != meetInTheMiddlePipelines_.end())
        return found->second;

    @autoreleasepool {
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.preprocessorMacros = @{
            @"ALPHABET_SIZE": @(key[0]),
            @"MAX_CANDIDATE_LEN": @(MAX_CANDIDATE_LEN),
            @"MEET_IN_THE_MIDDLE_SEED_MASK": @(kMeetInTheMiddleSeedMask),
            @"RULE_ANY": @(key[1]),
            @"RULE_SYMBOL_RUNS": @(key[2]),
            @"RULE_ADJACENT_BACKSLASHES": @(key[3]),
            @"RULE_MAX_BACKSLASH_COUNT": @(key[4]),
            @"ENTRY_SYMBOL_RUN": @(key[5]),
            @"ENTRY_BACKSLASHES": @(key[6]),
            @"ENTRY_NEXT_IS_BACKSLASH": @(key[7]),
        };
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:@(kMeetInTheMiddleKernelSource) options:options error:&error];
        if (!library) {
            fprintf(stderr, "Metal: compiling the meet-in-the-middle kernel (alphabet %d) failed:\n%s\n", key[0],
                    error.localizedDescription.UTF8String);
            exit(1);
        }
        id<MTLFunction> function = [library newFunctionWithName:@"searchMeetInTheMiddle"];
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "Metal: creating the meet-in-the-middle pipeline failed: %s\n", error.localizedDescription.UTF8String);
            exit(1);
        }
        meetInTheMiddlePipelines_[key] = pipeline;
        return pipeline;
    }
}

MeetInTheMiddleOutcome MetalBackend::runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) {
    MeetInTheMiddleOutcome outcome;
    if (batch.count == 0)
        return outcome;
    id<MTLComputePipelineState> pipeline = meetInTheMiddlePipeline();
    const MeetInTheMiddleRows rows = meetInTheMiddleRowsOf(batch, meet_in_the_middle_.alphabet.size());
    // The grid's threads are counted in 32 bits.
    if (rows.rowCount() > UINT32_MAX) {
        // runMeetInTheMiddleSearch's calls are far smaller (meetInTheMiddleTailsPerCall).
        fprintf(stderr, "INTERNAL ERROR: a meet-in-the-middle launch of %llu rows - exiting\n", (unsigned long long) rows.rowCount());
        exit(1);
    }
    MeetInTheMiddleArgs args = {};
    args.keyCount = meet_in_the_middle_.table->size();
    args.firstRow = rows.firstRow;
    args.lastRow = rows.lastRow;
    args.filterShift = meet_in_the_middle_.table->filterShift();
    args.meetingPointSeed1 = meet_in_the_middle_.meetingPoint.first;
    args.meetingPointSeed2 = meet_in_the_middle_.meetingPoint.second;
    args.tailLength = batch.tailLength;
    args.firstRowStartK = rows.firstRowStartK;
    args.lastRowEndK = rows.lastRowEndK;
    const NSUInteger threadgroupSize = std::min(kMeetInTheMiddleThreadgroupSize, pipeline.maxTotalThreadsPerThreadgroup);
    // A thread per row, rounded up to whole threadgroups.
    const NSUInteger threadgroups = (NSUInteger) ((rows.rowCount() + threadgroupSize - 1) / threadgroupSize);

    // Launched again, with room for every hit, as long as it had more than it
    // had room for - which a real search, comparing the meeting point's whole
    // 64-bit state, practically never does.
    auto* counts = static_cast<uint32_t*>(meetInTheMiddleCounts_.contents);
    for (;;) {
        args.capacity = meetInTheMiddleHitCapacity_;
        @autoreleasepool {
            id<MTLCommandBuffer> commands = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&args length:sizeof(args) atIndex:0];
            [encoder setBuffer:meetInTheMiddleKeys_ offset:0 atIndex:1];
            [encoder setBuffer:meetInTheMiddleFilter_ offset:0 atIndex:2];
            [encoder setBuffer:meetInTheMiddleChars_ offset:0 atIndex:3];
            [encoder setBuffer:meetInTheMiddleCounts_ offset:0 atIndex:4];
            [encoder setBuffer:meetInTheMiddleHits_ offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "Metal: the meet-in-the-middle kernel failed: %s\n", commands.error.localizedDescription.UTF8String);
                exit(1);
            }
        }
        if (counts[0] > batch.count || counts[1] > batch.count) {
            fprintf(stderr, "INTERNAL ERROR: a meet-in-the-middle launch of %llu tails looked up %u of them and found %u - exiting\n",
                    (unsigned long long) batch.count, counts[0], counts[1]);
            exit(1);
        }
        if (counts[1] <= meetInTheMiddleHitCapacity_)
            break;
        meetInTheMiddleHitCapacity_ = std::max(counts[1], 2 * meetInTheMiddleHitCapacity_);
        meetInTheMiddleHits_ = sharedBuffer(meetInTheMiddleHitCapacity_ * sizeof(uint64_t));
        counts[0] = counts[1] = 0;
    }

    outcome.probed = counts[0];
    const auto* hits = static_cast<const uint64_t*>(meetInTheMiddleHits_.contents);
    for (uint32_t i = 0; i < counts[1]; ++i) {
        const uint64_t index = hits[i];
        if (index < batch.start || index >= batch.start + batch.count) {
            fprintf(stderr, "INTERNAL ERROR: the meet-in-the-middle kernel reported the tail at %llu, which its batch doesn't have - exiting\n",
                    (unsigned long long) index);
            exit(1);
        }
        outcome.hits.push_back({batch.tailLength, index});
    }
    counts[0] = counts[1] = 0;
    return outcome;
}

} // namespace

GpuTuning metalTuningDefaults() {
    GpuTuning defaults;
    defaults.rowsPerThread = kRowsPerThreadDefault;
    defaults.windowChars = kWindowCharsDefault;
    defaults.workGroupSize = kThreadgroupSizeDefault;
    defaults.dictionaryWordsPerThread = kDictionaryWordsPerThreadDefault;
    defaults.dictionaryWorkGroupSize = kDictionaryThreadgroupSizeDefault;
    defaults.dictionaryTailsPerCell = kDictionaryTailsPerCellDefault;
    return defaults;
}

std::unique_ptr<SearchBackend> makeMetalBackend(const GpuTuning& tuning, std::string& error) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        error = "no Metal device";
        return nullptr;
    }
    return std::make_unique<MetalBackend>(device, withDefaults(tuning, metalTuningDefaults()));
}
