// The CUDA backend: the search kernels, and the SearchBackend that launches
// them (see engine/backend.h for what the engine asks of a backend). The
// same file, compiled with HIP for AMD GPUs, is the HIP backend - see
// gpu_runtime.h.

#include "backends/cuda/cuda_backend.h"

#include "backends/cuda/gpu_runtime.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include "backends/common/launch_waiter.h"
#include "backends/common/lowbits_filter.h"
#include "backends/common/row_batch.h"
#include "backends/common/row_pruning.h"
#include "backends/cuda/hash_kernels.cuh"
#include "backends/cuda/tuning.h"
#include "engine/backend.h"
#include "engine/hash_match.h"
#include "engine/limits.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, NAMEBREAK_GPU_RUNTIME_NAME " error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// Terminology (see also engine/backend.h):
// * Trailing part = the last `trailingLen` characters of the candidate; the
//   part the GPU enumerates (the rest - the "leading" part - is folded into
//   the prefix on the CPU, see runSearch).
// * Row = one combination of the trailing part's first `trailingLen - 1`
//   characters. A row holds `alphabetSize` candidates, one per value of the
//   last character, so row r's candidate with last character k has trailing
//   index r * alphabetSize + k. One GPU thread handles one whole row.

// One hashA hit, as filteredRowsKernel records it: the candidate's trailing
// index, and which of the launch's batches it's in.
struct Hit {
    uint64_t trailingIdx;
    uint32_t batch;
    uint32_t unused;
};

// Everything a launch reports, in one device buffer: how many hashA hits it
// had - which may exceed MAX_MATCHES, when the engine searches its batches
// again in smaller pieces - and the first MAX_MATCHES of them. The host reads
// the count and the first kHitsReadWithCount hits back in one copy, queued
// behind the kernel, so a launch with a hit or two costs the GPU no more idle
// time than one without; the host then checks every hit itself (see
// runBatches).
struct BatchResults {
    // 32 bits, though a launch of 16 leading values covers about 4.5 billion
    // candidates - more than it can count. Past 2^31 hits it would go negative
    // (and `slot < MAX_MATCHES` would let the kernel write outside `hits`),
    // past 2^32 wrap to a small count, and hits could be missed. But a real
    // launch has about one (4.5 billion / 2^32): overflowing this would take
    // half of all its candidates sharing one 32-bit hashA, which only a broken
    // hash or kernel could produce - and that would fail far more than this.
    // A 64-bit count would rule it out by construction, at no measurable
    // cost; it was left as it is. The other backends' batches (at most 50 *
    // 2^23 candidates) can't overflow theirs.
    int matchCount;
    int unused;
    Hit hits[MAX_MATCHES];
};
constexpr int kHitsReadWithCount = MAX_MATCHES < 16 ? MAX_MATCHES : 16;

struct DeviceBuffers {
    BatchResults* results;
    // This search's lookup filter: kLowBitsFilterEntries entries, see
    // buildLowBitsFilterTable (backends/common/lowbits_filter.h).
    uint64_t* filterTable;
    // The row groups to search - RowPruning::arena() (backends/common/row_pruning.h),
    // of which each batch has a slice.
    const uint32_t* groups;
};

static_assert(kThreadsPerBlock >= MAX_ALPHABET_SIZE, "filteredRowsKernel needs one thread per alphabet entry to fill its shared tables");

// One batch of a launch, as filteredRowsKernel needs it: its rows
// firstRow..lastRow of the trailing space, cut to [firstRowStartK,
// lastRowEndK) in its first and last row (see runBatches), the hash state
// after its prefix, and the row groups among them to search -
// DeviceBuffers::groups[groupsOffset .. groupsOffset + groupCount).
struct LaunchBatch {
    uint32_t firstRow;
    uint32_t lastRow;
    int firstRowStartK;
    int lastRowEndK;
    uint32_t seed1Start;
    uint32_t seed2Start;
    uint32_t groupsOffset;
    uint32_t groupCount;
};
// A launch's batches - a kernel argument, so passed by value.
struct LaunchBatches {
    LaunchBatch batch[kMaxBatchesPerLaunch];
};

// Suffix lengths 0-8 (see dispatchSuffixLen) get their own compile-time
// instantiation of filteredRowsKernel, with the suffix loop fully unrolled;
// anything longer falls back to kRuntimeSuffix.
constexpr int kRuntimeSuffix = -1;

// One step of the MPQ hash recurrence (hashA table, offset 0x100) for a
// character whose crypt-table key `key` and value `ord` are already known.
// Must match mpqHashCandidateAndSuffix/mpq_hash.cpp exactly, or a match
// found on one side would never reproduce on the other.
__device__ __forceinline__ void mpqStep(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = ord + seed1 + seed2 + (seed2 << 5) + 3;
}

// Hashes the N characters of `row` (its digits in base AlphabetSize, most
// significant first - filteredRowsKernel passes it a row group) into
// (seed1, seed2), via the block's shared tables. The lanes of a warp have
// consecutive groups (a few lanes each), so they read the same or
// neighbouring shared-memory entries here (no bank conflicts) - unlike the
// __constant__ d_cryptTable, where every lane needing a different entry is
// serialized.
template<int AlphabetSize, int N>
__device__ __forceinline__ void hashRowDigits(uint32_t row, uint32_t& seed1, uint32_t& seed2,
                                                const uint32_t* sKey, const uint32_t* sOrd) {
    if constexpr (N > 0) {
        unsigned digit[N];
        #pragma unroll
        for (int i = N - 1; i >= 0; --i) {
            uint32_t q = row / AlphabetSize;
            digit[i] = row - q * AlphabetSize;
            row = q;
        }
        #pragma unroll
        for (int i = 0; i < N; ++i)
            mpqStep(seed1, seed2, sKey[digit[i]], sOrd[digit[i]]);
    }
}

// A row *group* is the AlphabetSize consecutive rows that share every row
// character but the last: rows g * AlphabetSize .. g * AlphabetSize +
// AlphabetSize - 1 make up group g, and row g * AlphabetSize + d is the one
// whose last row character is d. filteredRowsKernel splits every group into
// kChunksPerGroup<AlphabetSize> chunks of consecutive rows, of about
// NAMEBREAK_ROWS_PER_THREAD rows each (tuning.h), and gives each chunk to a
// thread - so its rows share everything but their last row character, and
// only that changes from one to the next. Chunk c of a group is its rows
// d = c * AlphabetSize / kChunks .. (c + 1) * AlphabetSize / kChunks - 1:
// the chunks cover every row of the group exactly once, never cross into
// another group, and differ in size by at most one row.
template<int AlphabetSize>
constexpr uint32_t kChunksPerGroup = (AlphabetSize + NAMEBREAK_ROWS_PER_THREAD - 1) / NAMEBREAK_ROWS_PER_THREAD;

// One thread's work in filteredRowsKernel: chunk `chunk` of row group `group` -
// and if Listed, only the rows whose last row character is in `rowMask` (see
// backends/common/row_pruning.h).
template<int AlphabetSize, int SuffixLen, bool Listed>
__device__ __forceinline__ void searchChunk(uint32_t group, uint64_t rowMask, uint32_t chunk, int trailingLen, uint32_t firstRow,
                                            uint32_t lastRow, int firstRowStartK, int lastRowEndK, uint32_t targetA,
                                            uint32_t seed1Start, uint32_t seed2Start, uint32_t batch, const DeviceBuffers& bufs,
                                            const uint32_t* sKey, const uint32_t* sOrd) {
    constexpr uint32_t kChunks = kChunksPerGroup<AlphabetSize>;
    // The chunk's rows, as last row characters d of `group`, cut to the
    // launch's range in its first and last group.
    const uint64_t groupStart = (uint64_t) group * AlphabetSize;
    int dBegin = (int) (chunk * AlphabetSize / kChunks);
    int dEnd = (int) ((chunk + 1) * AlphabetSize / kChunks);
    if (groupStart + dBegin < firstRow)
        dBegin = (int) (firstRow - groupStart);
    if (groupStart + dEnd > (uint64_t) lastRow + 1)
        dEnd = (int) ((uint64_t) lastRow + 1 - groupStart);
    if (dBegin >= dEnd)
        return;
    // The d of the launch's first and last row, if they're in this group (-1
    // if not) - the rows firstRowStartK and lastRowEndK cut short.
    const int firstRowD = (group == firstRow / AlphabetSize) ? (int) (firstRow % AlphabetSize) : -1;
    const int lastRowD = (group == lastRow / AlphabetSize) ? (int) (lastRow % AlphabetSize) : -1;

    // The group's characters: every row character but the last, i.e. the
    // first trailingLen - 2 characters of the candidate - none when a row has
    // one character (trailingLen 2), and none at all with trailingLen 1, when
    // a row has no characters of its own and there's only row 0.
    uint32_t group1 = seed1Start;
    uint32_t group2 = seed2Start;
    switch (trailingLen - 2) {
        case 1: hashRowDigits<AlphabetSize, 1>(group, group1, group2, sKey, sOrd); break;
        case 2: hashRowDigits<AlphabetSize, 2>(group, group1, group2, sKey, sOrd); break;
        case 3: hashRowDigits<AlphabetSize, 3>(group, group1, group2, sKey, sOrd); break;
        case 4: hashRowDigits<AlphabetSize, 4>(group, group1, group2, sKey, sOrd); break;
        // trailingLen is validated against kMaxTrailingLen (== 6) by runSearch
    }
    const bool rowsHaveCharacters = trailingLen > 1;

    for (int d = dBegin; d < dEnd; ++d) {
        uint32_t seed1 = group1;
        uint32_t seed2 = group2;
        if (rowsHaveCharacters)
            mpqStep(seed1, seed2, sKey[d], sOrd[d]);

        // Bit k: the candidate with last character k is worth hashing.
        // Restricted to the launch's range in its first and last row - and to
        // the alphabet, which the table never exceeds anyway, but a stray bit
        // must not be able to index past sKey. The table doesn't change during
        // a launch, so it's read through the read-only data path (__ldg).
        //
        // A row the pruning leaves out (not in rowMask) gets no candidates:
        // the lanes of a warp step through their rows together, so skipping
        // one would save its lane nothing but idle time - while a branch
        // around every row cost about 4% more (measured with no row pruned).
        // Whole row groups are left out by the list the chunks come from.
        uint64_t mask;
        if constexpr (Listed)
            mask = ((rowMask >> d) & 1) ? __ldg(&bufs.filterTable[lowBitsFilterIndex(seed1, seed2)]) : 0;
        else
            mask = __ldg(&bufs.filterTable[lowBitsFilterIndex(seed1, seed2)]);
        mask &= (uint64_t(1) << AlphabetSize) - 1;
        if (d == firstRowD)
            mask &= ~uint64_t(0) << firstRowStartK;          // firstRowStartK is in [0, AlphabetSize)
        if (d == lastRowD)
            mask &= (uint64_t(1) << lastRowEndK) - 1;        // lastRowEndK is in [1, AlphabetSize]

        while (mask != 0) {
            const int k = __ffsll((long long) mask) - 1;
            mask &= mask - 1;
            uint32_t a = seed1, b = seed2;
            mpqStep(a, b, sKey[k], sOrd[k]);
            if constexpr (SuffixLen == kRuntimeSuffix) {
                const int n = d_suffix_size;
                for (int i = 0; i < n; ++i)
                    mpqStep(a, b, d_suffixKey[i], (unsigned char) d_suffix[i]);
            } else {
                #pragma unroll
                for (int i = 0; i < SuffixLen; ++i)
                    mpqStep(a, b, d_suffixKey[i], (unsigned char) d_suffix[i]);
            }
            if (hashAMatches(a, targetA)) {
                int slot = atomicAdd(&bufs.results->matchCount, 1);
                if (slot < MAX_MATCHES)
                    bufs.results->hits[slot] = Hit{(groupStart + d) * AlphabetSize + k, batch, 0};
            }
        }
    }
}

// No maxBackslashCount, pruneSymbolRuns or pruneUnopenedBrackets check here:
// the leading characters are checked on the CPU, before this kernel is ever
// launched (see the leadingIdx loop in runSearch), and when a search prunes
// the whole candidate, the host works out which row groups and rows survive
// (RowPruning, backends/common/row_pruning.h) - the kernel only walks its
// batch's list of them. See README.md's "Design decisions" section for why.
//
// A launch searches up to kMaxBatchesPerLaunch batches - usually each a whole
// leading value's trailing space, with its own seeds - one per blockIdx.y
// (see runBatches). Each thread handles one chunk (see kChunksPerGroup above -
// searchChunk) of its batch's rows firstRow..lastRow, and within each row
// every candidate whose last character index k is in [kBegin, kEnd) - all of
// them for every row but the (at most two) at the edges of the batch's range
// (firstRowStartK / lastRowEndK). The characters its rows share - the group's - are hashed
// once, and each row then costs one more step, for its own last row
// character. Then, instead of
// hashing every candidate of the row, one lookup in this search's filter
// table (bufs.filterTable, see backends/common/lowbits_filter.h) gives the set
// of last characters whose hashA has the target's low bits - on average
// alphabetSize / 2^kLowBitsFilterBits of them, and always including any
// candidate that matches the target. Only those are hashed in full. README.md's
// "The lookup filter" has why the lookup can never leave a match out.
//
// A hashA hit only records its trailing index and batch (BatchResults): the
// host builds the filename and checks it - hashA again, from scratch, and
// hashB - so none of that bloats this hot kernel.
template<int AlphabetSize, int SuffixLen, bool Listed>
__global__ void filteredRowsKernel(
    int trailingLen,
    LaunchBatches batches,
    uint32_t targetA,
    DeviceBuffers bufs
) {
    static_assert(AlphabetSize < 64, "a row's candidates, and one past the last of them, must fit a 64-bit mask");
    constexpr uint32_t kChunks = kChunksPerGroup<AlphabetSize>;
    __shared__ uint32_t sKey[AlphabetSize];
    __shared__ uint32_t sOrd[AlphabetSize];
    // In shared memory for the same reason as sKey: the lanes of a warp read
    // different entries.
    __shared__ uint64_t sRowMasks[kRowFlagCount];
    if (threadIdx.x < AlphabetSize) {
        sKey[threadIdx.x] = d_alphabetKey[threadIdx.x];
        sOrd[threadIdx.x] = d_alphabetOrd[threadIdx.x];
    }
    if (Listed && threadIdx.x < kRowFlagCount)
        sRowMasks[threadIdx.x] = d_rowMasks[threadIdx.x];
    __syncthreads();

    // This block's batch - the same for the whole block, so reading it from
    // the kernel's arguments with blockIdx.y costs nothing per thread.
    const uint32_t batchIndex = blockIdx.y;
    const LaunchBatch batch = batches.batch[batchIndex];

    // Every chunk of every group the batch touches - or if Listed, of every
    // group in the batch's list, those of them that aren't pruned - whatever
    // the grid's size: the number of threads launched only decides how the
    // chunks are shared out (normally one each), never which of them get
    // searched.
    const uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
    if constexpr (Listed) {
        const uint32_t* groups = bufs.groups + batch.groupsOffset;
        const uint64_t chunkCount = (uint64_t) batch.groupCount * kChunks;
        for (uint64_t c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; c < chunkCount; c += stride) {
            const uint32_t entry = __ldg(&groups[c / kChunks]);
            searchChunk<AlphabetSize, SuffixLen, true>(entry >> kRowFlagBits, sRowMasks[entry & (kRowFlagCount - 1)], (uint32_t) (c % kChunks),
                                                       trailingLen, batch.firstRow, batch.lastRow, batch.firstRowStartK, batch.lastRowEndK,
                                                       targetA, batch.seed1Start, batch.seed2Start, batchIndex, bufs, sKey, sOrd);
        }
    } else {
        const uint32_t firstGroup = batch.firstRow / AlphabetSize;
        const uint64_t chunkCount = (uint64_t) (batch.lastRow / AlphabetSize - firstGroup + 1) * kChunks;
        for (uint64_t c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; c < chunkCount; c += stride) {
            searchChunk<AlphabetSize, SuffixLen, false>(firstGroup + (uint32_t) (c / kChunks), 0, (uint32_t) (c % kChunks), trailingLen,
                                                        batch.firstRow, batch.lastRow, batch.firstRowStartK, batch.lastRowEndK, targetA,
                                                        batch.seed1Start, batch.seed2Start, batchIndex, bufs, sKey, sOrd);
        }
    }
}

// alphabetSize/suffix length have to be dispatched to one of a fixed set of
// compile-time template instantiations (see filteredRowsKernel's comment
// for why) - dispatchAlphabetSize/
// dispatchSuffixLen call `f` with a std::integral_constant of the matching
// value.
//
// The alphabet sizes this backend supports, ascending: add one here (and
// recompile/redistribute namebreak to volunteers) to support a new size.
// Must all be <= MAX_ALPHABET_SIZE.
template<int... Sizes>
struct AlphabetSizeList {};
using SupportedAlphabetSizes = AlphabetSizeList<29, 30, 40, 41, 42, 43, 47, 48, 49, 50>;

template<typename F, int... Sizes>
bool dispatchAlphabetSizeIn(AlphabetSizeList<Sizes...>, int alphabetSize, F&& f) {
    static_assert(((Sizes <= MAX_ALPHABET_SIZE) && ...), "an alphabet size exceeds MAX_ALPHABET_SIZE");
    return ((alphabetSize == Sizes ? (f(std::integral_constant<int, Sizes>{}), true) : false) || ...);
}

template<int... Sizes>
std::vector<int> alphabetSizesIn(AlphabetSizeList<Sizes...>) {
    return {Sizes...};
}

// Returns false if `alphabetSize` isn't in SupportedAlphabetSizes.
template<typename F>
bool dispatchAlphabetSize(int alphabetSize, F&& f) {
    return dispatchAlphabetSizeIn(SupportedAlphabetSizes{}, alphabetSize, f);
}

template<typename F>
void dispatchSuffixLen(int suffixLen, F&& f) {
    switch (suffixLen) {
        case 0: f(std::integral_constant<int, 0>{}); break;
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: f(std::integral_constant<int, kRuntimeSuffix>{}); break;
    }
}

namespace {

class CudaBackend : public SearchBackend {
public:
    // Not CUDA_CHECK'd, and its result deliberately ignored: this can run as
    // the process exits, after the CUDA runtime itself has shut down.
    ~CudaBackend() override {
        if (pinnedResults_)
            (void) cudaFreeHost(pinnedResults_);
        if (groups_)
            (void) cudaFree(groups_);
    }

#ifdef NAMEBREAK_HIP
    const char* name() const override { return "hip"; }
#else
    const char* name() const override { return "cuda"; }
#endif
    std::vector<int> supportedAlphabetSizes() const override { return alphabetSizesIn(SupportedAlphabetSizes{}); }
    int windowChars() const override { return NAMEBREAK_GPU_WINDOW_CHARS; }
    int maxTrailingLen() const override { return kMaxTrailingLen; }
    // A whole number of rows (see the terminology comment above
    // filteredRowsKernel), NAMEBREAK_ROWS_PER_LAUNCH of them (tuning.h). Launch
    // boundaries then always land on row boundaries (except at a range's own
    // start/end), so only a launch's first and last row can be partial.
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * NAMEBREAK_ROWS_PER_LAUNCH; }

    void beginSearch(const SearchConstants& constants) override;
    BatchOutcome runBatch(int trailingLen, uint64_t startIdx, uint64_t count, const BatchParams& params) override;
    void endSearch() override;
    // NAMEBREAK_BATCHES_PER_LAUNCH (tuning.h) batches, each up to
    // NAMEBREAK_ROWS_PER_LAUNCH rows, in one launch - see filteredRowsKernel.
    int maxBatchesPerCall() const override { return NAMEBREAK_BATCHES_PER_LAUNCH; }
    BatchOutcome runBatches(int trailingLen, const std::vector<BatchRequest>& batches) override;

private:
    int alphabetSize_ = 0;
    int suffixLen_ = 0;
    uint32_t targetA_ = 0;
    // A launch's results, copied back into pinned host memory by a copy
    // queued behind the kernel (see runBatches). Allocated once, by the first
    // search - it holds nothing from one launch to the next.
    BatchResults* pinnedResults_ = nullptr;
    DeviceBuffers bufs_ = {};
    // Checks every hit the kernel reports, on the CPU (see runBatches).
    HitVerifier verifier_;

    // Sleeps through most of each launch, instead of spinning in
    // cudaDeviceSynchronize for all of it (see backends/common/launch_waiter.h).
    LaunchWaiter waiter_;

    // The row groups each batch searches (see backends/common/row_pruning.h),
    // and their copy on the GPU: the first groupsUploaded_ entries of
    // rowPruning_.arena(), of generation groupsGeneration_, in a buffer of
    // groupsCapacity_ entries. Kept from one search to the next, as the
    // lists are.
    RowPruning rowPruning_;
    uint32_t* groups_ = nullptr;
    size_t groupsCapacity_ = 0;
    size_t groupsUploaded_ = 0;
    uint64_t groupsGeneration_ = 0;
    void uploadGroups();
    // Whether this search prunes the whole candidate, and the kernel walks
    // lists of row groups: only then - a kernel that walks every group
    // measured about 5% faster than one walking a list of them all.
    bool listed_ = false;
};

// How many of the lookup filter's entries beginSearch checks against their
// definition before every search (see checkLowBitsFilterTable), each with two
// different random high bits - about a millisecond. tests/lowbits_filter_test.cpp
// checks every entry of many tables.
constexpr uint32_t kFilterEntriesCheckedPerSearch = 1024;

// A search must not start with a filter table that is wrong: it could drop a
// match without any other sign. Like CUDA_CHECK, this ends the process - a
// coordinator range is then reassigned when its lease runs out.
[[noreturn]] void refuseFilterTable(const char* what, const std::string& detail) {
    fprintf(stderr, "INTERNAL ERROR: the lookup filter table %s (%s) - refusing to search with it\n", what, detail.c_str());
    exit(1);
}

void CudaBackend::beginSearch(const SearchConstants& constants) {
    alphabetSize_ = (int) constants.alphabet.size();
    suffixLen_ = (int) constants.suffix.size();
    targetA_ = constants.targetHashA;

    short suffixSize = (short) suffixLen_;
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix_size, &suffixSize, sizeof(suffixSize)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix, constants.suffix.c_str(), suffixLen_ + 1));

    // The per-search tables filteredRowsKernel reads - see their declaration
    // in hash_kernels.cuh.
    {
        uint32_t h_alphabetKey[MAX_ALPHABET_SIZE] = {0};
        uint32_t h_alphabetOrd[MAX_ALPHABET_SIZE] = {0};
        uint32_t h_suffixKey[kMaxSuffixSize] = {0};
        for (int k = 0; k < alphabetSize_; ++k) {
            h_alphabetOrd[k] = (unsigned char) constants.alphabet[k];
            h_alphabetKey[k] = constants.cryptTable[0x100 + h_alphabetOrd[k]];
        }
        for (int i = 0; i < suffixLen_; ++i)
            h_suffixKey[i] = constants.cryptTable[0x100 + (unsigned char) constants.suffix[i]];
        CUDA_CHECK(cudaMemcpyToSymbol(d_alphabetKey, h_alphabetKey, sizeof(h_alphabetKey)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_alphabetOrd, h_alphabetOrd, sizeof(h_alphabetOrd)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_suffixKey, h_suffixKey, sizeof(h_suffixKey)));
    }

    // Allocated once per search and reused for every batch, instead of
    // malloc/free per call - these buffers are always the same size, so
    // there's no reason to pay driver allocation overhead on every single
    // kernel launch.
    CUDA_CHECK(cudaMalloc(&bufs_.results, sizeof(BatchResults)));
    // A long-lived process (coordinator mode) runs many searches over its
    // lifetime, one per claimed range - zeroing this fresh allocation every
    // search (rather than relying on any process-lifetime state) is what
    // keeps a previous range's match from making every subsequent range
    // falsely report "found" immediately. It also establishes the
    // "matchCount is 0 at launch" invariant runBatch maintains from here.
    CUDA_CHECK(cudaMemset(bufs_.results, 0, sizeof(BatchResults)));
    if (!pinnedResults_)
        CUDA_CHECK(cudaMallocHost((void**) &pinnedResults_, sizeof(BatchResults)));
    waiter_.beginSearch();
    verifier_.begin(constants);
    listed_ = constants.trailingRules.any();
    if (listed_) {
        rowPruning_.begin(constants);
        CUDA_CHECK(cudaMemcpyToSymbol(d_rowMasks, rowPruning_.rowMasks(), kRowFlagCount * sizeof(uint64_t)));
    }

    // This search's lookup filter - it depends on the alphabet, the suffix and
    // the target, so it's built for every search. Checked against its
    // definition before it's used, and read back after the upload to make
    // sure the GPU has exactly what was checked.
    const std::vector<uint64_t> table = buildLowBitsFilterTable(constants);
    std::string error;
    if (!checkLowBitsFilterTable(table, constants, kFilterEntriesCheckedPerSearch, 2, std::random_device{}(), error))
        refuseFilterTable("failed its check", error);
    const size_t tableBytes = table.size() * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&bufs_.filterTable, tableBytes));
    CUDA_CHECK(cudaMemcpy(bufs_.filterTable, table.data(), tableBytes, cudaMemcpyHostToDevice));
    std::vector<uint64_t> readBack(table.size());
    CUDA_CHECK(cudaMemcpy(readBack.data(), bufs_.filterTable, tableBytes, cudaMemcpyDeviceToHost));
    if (readBack != table)
        refuseFilterTable("read back from the GPU differs from the one uploaded", std::to_string(tableBytes) + " bytes");
}

void CudaBackend::endSearch() {
    CUDA_CHECK(cudaFree(bufs_.results));
    CUDA_CHECK(cudaFree(bufs_.filterTable));
    bufs_ = {};
}

// Brings the GPU's copy of the row groups' lists up to date with
// rowPruning_.arena(), which only grows within a generation: only what was
// added since goes up, unless the buffer is too small or the lists were
// cleared. Nothing is running on the GPU here (runBatches waits for every
// launch before it returns), so a plain copy is safe.
void CudaBackend::uploadGroups() {
    const std::vector<uint32_t>& arena = rowPruning_.arena();
    if (rowPruning_.generation() != groupsGeneration_) {
        groupsGeneration_ = rowPruning_.generation();
        groupsUploaded_ = 0;
    }
    if (arena.size() > groupsCapacity_) {
        if (groups_)
            CUDA_CHECK(cudaFree(groups_));
        groupsCapacity_ = std::max(arena.size(), 2 * groupsCapacity_);
        CUDA_CHECK(cudaMalloc(&groups_, groupsCapacity_ * sizeof(uint32_t)));
        groupsUploaded_ = 0;
    }
    if (arena.size() > groupsUploaded_) {
        CUDA_CHECK(cudaMemcpy(groups_ + groupsUploaded_, arena.data() + groupsUploaded_, (arena.size() - groupsUploaded_) * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        groupsUploaded_ = arena.size();
    }
    bufs_.groups = groups_;
}

BatchOutcome CudaBackend::runBatch(int trailingLen, uint64_t startIdx, uint64_t count, const BatchParams& params) {
    return runBatches(trailingLen, {BatchRequest{startIdx, count, params}});
}

BatchOutcome CudaBackend::runBatches(int trailingLen, const std::vector<BatchRequest>& requests) {
    const int alphabetSize = alphabetSize_;
    const int batchCount = (int) requests.size();
    BatchOutcome outcome;
    if (batchCount < 1 || batchCount > NAMEBREAK_BATCHES_PER_LAUNCH) {
        fprintf(stderr, "INTERNAL ERROR: %d batches for one launch (1 to %d allowed) - exiting\n", batchCount, NAMEBREAK_BATCHES_PER_LAUNCH);
        exit(1);
    }

    // bufs_.results->matchCount is already 0 here: beginSearch zeroes it, and a
    // launch with hits queues its reset (see below), so there's no per-launch
    // memset in the gap between two kernels.

    // Each batch's range [start, start + count) covers its rows
    // firstRow..lastRow; only the first and last row can be partial (see
    // filteredRowsKernel). With the whole candidate pruned, each also gets
    // the list of its row groups that survive.
    const bool listed = listed_;
    LaunchBatches batches = {};
    uint64_t maxGroups = 0;
    for (int b = 0; b < batchCount; ++b) {
        const uint64_t startIdx = requests[b].start;
        const uint64_t endIdx = startIdx + requests[b].count;
        const uint64_t firstRow = startIdx / alphabetSize;
        const uint64_t lastRow = (endIdx - 1) / alphabetSize;
        const uint64_t rowCount = lastRow - firstRow + 1;
        if (requests[b].count == 0 || lastRow > UINT32_MAX || rowCount > (1ull << 31)) {
            fprintf(stderr, "Batch too large for the kernel's 32-bit row index (rows %llu..%llu) - exiting\n",
                    (unsigned long long) firstRow, (unsigned long long) lastRow);
            exit(1);
        }
        LaunchBatch& batch = batches.batch[b];
        batch.firstRow = (uint32_t) firstRow;
        batch.lastRow = (uint32_t) lastRow;
        batch.firstRowStartK = (int) (startIdx - firstRow * alphabetSize);
        batch.lastRowEndK = (int) (endIdx - lastRow * alphabetSize); // in [1, alphabetSize]
        batch.seed1Start = requests[b].params.seed1Start;
        batch.seed2Start = requests[b].params.seed2Start;
        if (listed) {
            const RowPruning::Slice groups =
                rowPruning_.groupsFor(trailingLen, requests[b].params.pruneEntry, firstRow / alphabetSize, lastRow / alphabetSize);
            batch.groupsOffset = groups.offset;
            batch.groupCount = groups.count;
            maxGroups = std::max<uint64_t>(maxGroups, groups.count);
        } else {
            maxGroups = std::max(maxGroups, lastRow / alphabetSize - firstRow / alphabetSize + 1);
        }
    }
    // Every row of every batch pruned: nothing to launch.
    if (maxGroups == 0)
        return outcome;
    if (listed)
        uploadGroups();

    const auto launched = std::chrono::steady_clock::now();
    bool supported = dispatchAlphabetSize(alphabetSize, [&](auto alphabetC) {
        // A type alias rather than a constexpr local: MSVC treats a constexpr
        // local of this lambda read from the nested [&] lambda below as a
        // capture, so it's no longer a constant expression there (C2672).
        using AlphabetC = decltype(alphabetC);
        // One row of blocks per batch (blockIdx.y), and in it one GPU thread
        // per chunk of every row group the batch touches (see
        // kChunksPerGroup) - about NAMEBREAK_ROWS_PER_THREAD rows each - as
        // many as the largest batch needs. The kernel works out each batch's
        // chunks itself and covers all of them whatever the grid, so this
        // count only spreads the work. Every 32 consecutive threads form a
        // "warp" that the hardware runs in lockstep (SIMT) - that grouping is
        // automatic (256 threads/block = 8 warps/block here), not something
        // chosen at this call site.
        const uint64_t threads = maxGroups * kChunksPerGroup<AlphabetC::value>;
        const dim3 blocks((unsigned) ((threads + kThreadsPerBlock - 1) / kThreadsPerBlock), (unsigned) batchCount);
        dispatchSuffixLen(suffixLen_, [&](auto suffixC) {
            constexpr int SuffixLen = decltype(suffixC)::value;
            if (listed)
                filteredRowsKernel<AlphabetC::value, SuffixLen, true><<<blocks, kThreadsPerBlock>>>(trailingLen, batches, targetA_, bufs_);
            else
                filteredRowsKernel<AlphabetC::value, SuffixLen, false><<<blocks, kThreadsPerBlock>>>(trailingLen, batches, targetA_, bufs_);
        });
    });
    if (!supported) {
        // runSearch checks supportedAlphabetSizes() before any batch.
        fprintf(stderr, "Unsupported alphabet size: %d\n", alphabetSize);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    // The results' copy is queued right behind the kernel, into pinned memory,
    // so the GPU starts it the moment the kernel ends and one wait covers
    // both - a blocking cudaMemcpy after the wait cost the GPU another round
    // trip, idle, between every two launches. It brings the hit count and the
    // first kHitsReadWithCount hits, so a launch with a few hits needs no
    // second round trip either.
    const size_t readWithCount = offsetof(BatchResults, hits) + kHitsReadWithCount * sizeof(Hit);
    CUDA_CHECK(cudaMemcpyAsync(pinnedResults_, bufs_.results, readWithCount, cudaMemcpyDeviceToHost, 0));
    uint64_t candidates = 0;
    for (const BatchRequest& request : requests)
        candidates += request.count;
    waiter_.wait(candidates, launched, [] { CUDA_CHECK(cudaDeviceSynchronize()); });

    const int hitCount = pinnedResults_->matchCount;
    outcome.hitCount = hitCount;
    if (hitCount == 0)
        return outcome;

    // Hits are rare (a launch of 16 leading values has one about two times in
    // three), so everything below is the exception path.
    if (hitCount <= MAX_MATCHES && hitCount > kHitsReadWithCount) {
        CUDA_CHECK(cudaMemcpy(pinnedResults_->hits + kHitsReadWithCount, bufs_.results->hits + kHitsReadWithCount,
                              (hitCount - kHitsReadWithCount) * sizeof(Hit), cudaMemcpyDeviceToHost));
    }
    // Restores "matchCount is 0 at launch" for the next launch - queued, so it
    // runs before that without anything waiting for it here.
    CUDA_CHECK(cudaMemsetAsync(&bufs_.results->matchCount, 0, sizeof(int), 0));
    if (hitCount > MAX_MATCHES) {
        // More hits than could be recorded: nothing from this launch has been
        // checked, so leave it all to the engine to search again in smaller
        // pieces.
        return outcome;
    }

    // Every hit, checked on the CPU with its own batch's prefix: the filename
    // is built from the trailing index again, and hashed from scratch - an
    // independent implementation on a different processor, which prints a
    // WARNING if it doesn't get the target's hashA (the tests treat that as a
    // failure) - and hashB is checked. A match found by *this* launch is
    // reported now, not by whatever call comes after it, which may never come
    // (a bounded search's last launch).
    for (int i = 0; i < hitCount; ++i) {
        const Hit& hit = pinnedResults_->hits[i];
        if (hit.batch >= (uint32_t) batchCount) {
            fprintf(stderr, "INTERNAL ERROR: the kernel reported a hit in batch %u of a launch of %d - exiting\n", hit.batch, batchCount);
            exit(1);
        }
        verifier_.addHits({hit.trailingIdx}, trailingLen, requests[hit.batch].params, outcome);
    }
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeCudaBackend(std::string& error) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        error = std::string("no usable " NAMEBREAK_GPU_RUNTIME_NAME " device: ") + cudaGetErrorString(err);
        return nullptr;
    }
    if (deviceCount == 0) {
        error = "no " NAMEBREAK_GPU_RUNTIME_NAME " device";
        return nullptr;
    }
    return std::make_unique<CudaBackend>();
}
