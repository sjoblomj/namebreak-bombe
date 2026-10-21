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
#include "backends/common/meet_in_the_middle_launch.h"
#include "backends/common/launch_waiter.h"
#include "backends/common/lowbits_filter.h"
#include "backends/common/row_batch.h"
#include "backends/common/row_pruning.h"
#include "backends/cuda/hash_kernels.cuh"
#include "backends/cuda/tuning.h"
#include "engine/backend.h"
#include "engine/dictionary_search.h"
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

// One hashA hit, as filteredRowsKernel records it: the candidate's row and
// last character, and which of the launch's batches it's in. The host works
// out its trailing index - a multiplication by the alphabet's size the kernel
// would otherwise do for every row with a flagged candidate.
struct Hit {
    uint64_t row;   // the candidate's trailing index is row * alphabetSize + k
    uint32_t batch;
    uint32_t k;
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
    // cost; it was left as it is. The other backends' batches (at most 63 *
    // 2^23 candidates) can't overflow theirs.
    int matchCount;
    int unused;
    Hit hits[MAX_MATCHES];
};
constexpr int kHitsReadWithCount = MAX_MATCHES < 16 ? MAX_MATCHES : 16;

struct DeviceBuffers {
    BatchResults* results;
    // This search's lookup filter: lowBitsFilterEntries(kCudaLowBitsFilterBits)
    // entries, see buildLowBitsFilterTable (backends/common/lowbits_filter.h).
    uint64_t* filterTable;
    // The row groups to search - RowPruning::arena() (backends/common/row_pruning.h),
    // of which each batch has a slice.
    const uint32_t* groups;
    // The row masks of this search's row pruning (RowPruning::rowMasks): the
    // last row characters an entry of a group list allows, by the entry's
    // classes - kRowFlagCount of them, 1 KB. Read from here by each chunk,
    // through the read-only cache: copying them into shared memory first, as
    // sKey is, doubled the kernel's shared memory and cost a search pruning
    // the whole candidate about 7%.
    const uint64_t* rowMasks;
};

static_assert(kThreadsPerBlock >= MAX_ALPHABET_SIZE, "filteredRowsKernel needs one thread per alphabet entry to fill its shared tables");
static_assert(sizeof(d_insertKey) >= 2 * kMaxInsertLen * sizeof(uint32_t), "d_insertKey must hold both insertions' text");

// One batch of a launch, as filteredRowsKernel needs it (see runBatches): its
// rows, from row firstRowD of row group firstGroup to row lastRowD of group
// lastGroup (see kChunksPerGroup below for groups), cut to [firstRowStartK,
// lastRowEndK) in its first and last row, the hash state after its prefix,
// and the row groups among them to search -
// DeviceBuffers::groups[groupsOffset .. groupsOffset + groupCount). The
// divisions that split the batch's first and last row into a group and a
// row are done here, on the host.
struct LaunchBatch {
    uint32_t firstGroup;
    uint32_t lastGroup;
    int firstRowD;
    int lastRowD;
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

// Suffix lengths 0-12 (see dispatchSuffixLen) get their own compile-time
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

// Division by a number known only at runtime, as a multiplication and a
// shift - what a compiler does for a constant divisor, and much cheaper on a
// GPU than a division: Granlund and Montgomery's method ("Division by
// invariant integers using multiplication", 1994), exact for every 32-bit
// numerator. makeFastDivisor (host) works out the multiplier and shift.
struct FastDivisor {
    uint32_t multiplier;
    uint32_t shift;
};

FastDivisor makeFastDivisor(uint32_t divisor) {
    // shift = ceil(log2(divisor)); multiplier = floor(2^32 * (2^shift -
    // divisor) / divisor) + 1, which fits 32 bits for any divisor of 1 to 2^31.
    uint32_t shift = 0;
    while ((uint64_t(1) << shift) < divisor)
        ++shift;
    const uint64_t multiplier = (uint64_t(1) << 32) * ((uint64_t(1) << shift) - divisor) / divisor + 1;
    return FastDivisor{(uint32_t) multiplier, shift};
}

__host__ __device__ __forceinline__ uint32_t divide(uint32_t n, FastDivisor by) {
#ifdef __CUDA_ARCH__
    const uint32_t high = __umulhi(n, by.multiplier);
#else
    const uint32_t high = (uint32_t) (((uint64_t) n * by.multiplier) >> 32);
#endif
    return (uint32_t) (((uint64_t) high + n) >> by.shift);
}

// The alphabet's size, and what the kernel divides by - a kernel argument,
// the same for a whole search. See kChunksPerGroup below for chunks.
struct AlphabetShape {
    uint32_t size;
    uint32_t chunksPerGroup;
    FastDivisor bySize;
    FastDivisor byChunksPerGroup;
};

// A row *group* is the alphabetSize consecutive rows that share every row
// character but the last: rows g * alphabetSize .. g * alphabetSize +
// alphabetSize - 1 make up group g, and row g * alphabetSize + d is the one
// whose last row character is d. filteredRowsKernel splits every group into
// kChunksPerGroup(alphabetSize) chunks of consecutive rows, of about
// NAMEBREAK_ROWS_PER_THREAD rows each (tuning.h), and gives each chunk to a
// thread - so its rows share everything but their last row character, and
// only that changes from one to the next. Chunk c of a group is its rows
// d = c * alphabetSize / kChunks .. (c + 1) * alphabetSize / kChunks - 1:
// the chunks cover every row of the group exactly once, never cross into
// another group, differ in size by at most one row, and have at most
// NAMEBREAK_ROWS_PER_THREAD rows.
__host__ __device__ constexpr uint32_t kChunksPerGroup(uint32_t alphabetSize) {
    return (alphabetSize + NAMEBREAK_ROWS_PER_THREAD - 1) / NAMEBREAK_ROWS_PER_THREAD;
}

// The alphabet's size, and what's divided by it, either compiled into the
// kernel (FixedSize, see CompiledAlphabetSizes) or taken from `shape` at
// runtime (FixedSize 0). A size the compiler knows saves a few instructions a
// row - masks folded together, values it would otherwise work out again -
// which made the kernel walking lists of row groups about 2% faster.
template<int FixedSize>
__device__ __forceinline__ uint32_t alphabetSizeOf(const AlphabetShape& shape) {
    if constexpr (FixedSize != 0)
        return FixedSize;
    else
        return shape.size;
}
template<int FixedSize>
__device__ __forceinline__ uint32_t chunksPerGroupOf(const AlphabetShape& shape) {
    if constexpr (FixedSize != 0)
        return kChunksPerGroup(FixedSize);
    else
        return shape.chunksPerGroup;
}
template<int FixedSize>
__device__ __forceinline__ uint32_t divideBySize(uint32_t n, const AlphabetShape& shape) {
    if constexpr (FixedSize != 0)
        return n / FixedSize;
    else
        return divide(n, shape.bySize);
}
template<int FixedSize>
__device__ __forceinline__ uint32_t divideByChunksPerGroup(uint32_t n, const AlphabetShape& shape) {
    if constexpr (FixedSize != 0)
        return n / kChunksPerGroup(FixedSize);
    else
        return divide(n, shape.byChunksPerGroup);
}

// Makes `value` opaque to the compiler, which then keeps it in a register
// rather than work it out again wherever it's used - which it would rather do
// with some values of searchChunk, every row. NVIDIA's compiler only: nothing
// has been measured on AMD's, which may not take the "r" constraint.
template<typename T>
__device__ __forceinline__ void keepInRegister(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
    (void) value;
#else
    asm("" : "+r"(value));
#endif
}

// Hashes the text inserted before a row group's character `at` (of
// groupDigits; groupDigits: after the last one) - see d_insertLayout. The
// same for every thread, and once per chunk.
__device__ __forceinline__ void hashGroupInsertions(int at, int groupDigits, uint32_t& seed1, uint32_t& seed2) {
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        if (groupDigits + 2 - d_insertLayout.groupCharsAfter[n] == at) {
            const int start = d_insertLayout.groupStart[n], len = d_insertLayout.groupLen[n];
            for (int i = 0; i < len; ++i)
                mpqStep(seed1, seed2, d_insertKey[start + i], d_insertOrd[start + i]);
        }
    }
}

// Hashes the N characters of `row` (its digits in base alphabetSize, most
// significant first - filteredRowsKernel passes it a row group) - and if
// Inserted, the text inserted between and after them - into (seed1, seed2),
// via the block's shared tables. The lanes of a warp have
// consecutive groups (a few lanes each), so they read the same or
// neighbouring shared-memory entries here (no bank conflicts) - unlike the
// __constant__ d_cryptTable, where every lane needing a different entry is
// serialized.
template<int FixedSize, int N, bool Inserted>
__device__ __forceinline__ void hashRowDigits(uint32_t row, const AlphabetShape& shape, uint32_t& seed1, uint32_t& seed2,
                                                const uint32_t* sKey, const uint32_t* sOrd) {
    if constexpr (N > 0) {
        unsigned digit[N];
        #pragma unroll
        for (int i = N - 1; i >= 0; --i) {
            uint32_t q = divideBySize<FixedSize>(row, shape);
            digit[i] = row - q * alphabetSizeOf<FixedSize>(shape);
            row = q;
        }
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            if constexpr (Inserted) {
                if (i > 0)
                    hashGroupInsertions(i, N, seed1, seed2);
            }
            mpqStep(seed1, seed2, sKey[digit[i]], sOrd[digit[i]]);
        }
        if constexpr (Inserted)
            hashGroupInsertions(N, N, seed1, seed2);
    }
}

// One thread's work in filteredRowsKernel: chunk `chunk` of row group `group` -
// and if Listed, only the rows whose last row character is in `rowMask` (see
// backends/common/row_pruning.h).
//
// firstRowD and lastRowD are the d of the launch's first and last row, if
// they're in this group (-1 if not) - the rows firstRowStartK and
// lastRowEndK cut short, and the chunk's rows with them.
template<int FixedSize, int SuffixLen, bool Listed, bool Inserted>
__device__ __forceinline__ void searchChunk(uint32_t group, uint64_t rowMask, uint32_t chunk, int trailingLen, const AlphabetShape& shape,
                                            int firstRowD, int lastRowD, int firstRowStartK, int lastRowEndK, uint32_t targetA,
                                            uint32_t seed1Start, uint32_t seed2Start, uint32_t batch, const DeviceBuffers& bufs,
                                            const uint32_t* sKey, const uint32_t* sOrd) {
    // The chunk's rows, as last row characters d of `group`, cut to the
    // launch's range in its first and last group.
    const uint32_t alphabetSize = alphabetSizeOf<FixedSize>(shape);
    const uint64_t groupStart = (uint64_t) group * alphabetSize;
    int dBegin = (int) divideByChunksPerGroup<FixedSize>(chunk * alphabetSize, shape);
    int dEnd = (int) divideByChunksPerGroup<FixedSize>((chunk + 1) * alphabetSize, shape);
    if (dBegin < firstRowD)
        dBegin = firstRowD;
    if (lastRowD >= 0 && dEnd > lastRowD + 1)
        dEnd = lastRowD + 1;
    if (dBegin >= dEnd)
        return;
    // Kept, rather than worked out again for every row - which the compiler
    // did with the alphabet's size taken at runtime: 25% more instructions.
    keepInRegister(firstRowD);
    keepInRegister(lastRowD);

    // The group's characters: every row character but the last, i.e. the
    // first trailingLen - 2 characters of the candidate - none when a row has
    // one character (trailingLen 2), and none at all with trailingLen 1, when
    // a row has no characters of its own and there's only row 0.
    uint32_t group1 = seed1Start;
    uint32_t group2 = seed2Start;
    switch (trailingLen - 2) {
        case 1: hashRowDigits<FixedSize, 1, Inserted>(group, shape, group1, group2, sKey, sOrd); break;
        case 2: hashRowDigits<FixedSize, 2, Inserted>(group, shape, group1, group2, sKey, sOrd); break;
        case 3: hashRowDigits<FixedSize, 3, Inserted>(group, shape, group1, group2, sKey, sOrd); break;
        case 4: hashRowDigits<FixedSize, 4, Inserted>(group, shape, group1, group2, sKey, sOrd); break;
        // trailingLen is validated against kMaxTrailingLen (== 6) by runSearch
    }
    const bool rowsHaveCharacters = trailingLen > 1;
    // If Listed, bit i: the chunk's i-th row isn't pruned - in the narrowest
    // integer a chunk's rows fit, shifted one row on at a time, which costs
    // less than testing bit d of the 64-bit rowMask every row.
    using ChunkRowBits = std::conditional_t<NAMEBREAK_ROWS_PER_THREAD <= 32, uint32_t, uint64_t>;
    ChunkRowBits rowBits = (ChunkRowBits) (rowMask >> dBegin);

    // The candidates a row may have: those of the alphabet - and in the
    // launch's first and last row, of its range.
    const uint64_t alphabetMask = (uint64_t(1) << alphabetSize) - 1;
    const uint64_t firstRowMask = alphabetMask & (~uint64_t(0) << firstRowStartK); // firstRowStartK is in [0, alphabetSize)
    const uint64_t lastRowMask = (uint64_t(1) << lastRowEndK) - 1;                  // lastRowEndK is in [1, alphabetSize]

    // A row's own step from the group's state is mpqStep with these parts
    // the same for every row - worked out once here, and kept (as above).
    uint32_t groupSum = group1 + group2;
    uint32_t group2Term = group2 + (group2 << 5) + 3;
    keepInRegister(groupSum);
    keepInRegister(group2Term);
    // The text inserted after a row's own character, if any - hashed for
    // every row.
    int rowInsertStart = 0, rowInsertLen = 0;
    if constexpr (Inserted) {
        rowInsertStart = d_insertLayout.rowStart;
        rowInsertLen = d_insertLayout.rowLen;
    }

    // A row's state: the group's, one step on for the row's own character d
    // (and the text inserted after it, if any).
    auto rowState = [&](int d, uint32_t& seed1, uint32_t& seed2) {
        seed1 = group1;
        seed2 = group2;
        if (rowsHaveCharacters) {
            seed1 = sKey[d] ^ groupSum;
            seed2 = sOrd[d] + seed1 + group2Term;
        }
        if constexpr (Inserted) {
            for (int i = 0; i < rowInsertLen; ++i)
                mpqStep(seed1, seed2, d_insertKey[rowInsertStart + i], d_insertOrd[rowInsertStart + i]);
        }
    };
    // Bit k: the candidate of row d with last character k is worth hashing -
    // the row's table entry, restricted to the launch's range in its first and
    // last row, and to the alphabet, which the table never exceeds anyway, but
    // a stray bit must not be able to index past sKey. The table doesn't
    // change during a launch, so it's read through the read-only data path
    // (__ldg).
    auto rowCandidates = [&](int d, uint32_t seed1, uint32_t seed2) {
        uint64_t mask = __ldg(&bufs.filterTable[lowBitsFilterIndex(seed1, seed2, kCudaLowBitsFilterBits)]);
        mask &= (d == firstRowD) ? firstRowMask : alphabetMask;
        if (d == lastRowD)
            mask &= lastRowMask;
        return mask;
    };

    // First the rows, noting only which of them have candidates worth
    // hashing (bit i: the chunk's i-th row) - then those candidates, one per
    // round of the loop below. A warp goes round a loop as many times as its
    // busiest lane needs: hashing each row's candidates right after its
    // lookup took as many rounds as the busiest lane's candidates in *each
    // row*, about twice a row, where this takes as many as its candidates in
    // the whole chunk.
    //
    // A row the pruning leaves out (not in rowMask) gets no candidates: the
    // lanes of a warp step through their rows together, so skipping one would
    // save its lane nothing but idle time - while a branch around every row
    // cost about 4% more (measured with no row pruned). Whole row groups are
    // left out by the list the chunks come from.
    ChunkRowBits flaggedRows = 0;
    for (int d = dBegin; d < dEnd; ++d) {
        uint32_t seed1, seed2;
        rowState(d, seed1, seed2);
        uint64_t mask;
        if constexpr (Listed) {
            mask = (rowBits & 1) ? rowCandidates(d, seed1, seed2) : 0;
            rowBits >>= 1;
        } else
            mask = rowCandidates(d, seed1, seed2);
        flaggedRows |= (ChunkRowBits) (mask != 0) << (d - dBegin);
    }

    // The flagged rows' candidates: a lane that has hashed its row's takes
    // the next flagged row, and works its state and candidates out again.
    uint64_t mask = 0;
    uint32_t seed1 = 0, seed2 = 0;
    int d = 0;
    while ((flaggedRows | mask) != 0) {
        if (mask == 0) {
            d = dBegin + __ffsll((long long) flaggedRows) - 1;
            flaggedRows &= flaggedRows - 1;
            rowState(d, seed1, seed2);
            mask = rowCandidates(d, seed1, seed2);
            if (mask == 0)
                continue; // can't happen: the same row's candidates as above
        }
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
                bufs.results->hits[slot] = Hit{groupStart + d, batch, (uint32_t) k};
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
// alphabetSize / 2^kCudaLowBitsFilterBits of them, and always including any
// candidate that matches the target. Only those are hashed in full. README.md's
// "The lookup filter" has why the lookup can never leave a match out.
//
// A hashA hit only records its row, last character and batch (BatchResults): the
// host builds the filename and checks it - hashA again, from scratch, and
// hashB - so none of that bloats this hot kernel.
template<int FixedSize, int SuffixLen, bool Listed, bool Inserted>
__global__ void filteredRowsKernel(
    int trailingLen,
    AlphabetShape shape,
    LaunchBatches batches,
    uint32_t targetA,
    DeviceBuffers bufs
) {
    static_assert(MAX_ALPHABET_SIZE < 64, "a row's candidates, and one past the last of them, must fit a 64-bit mask");
    __shared__ uint32_t sKey[MAX_ALPHABET_SIZE];
    __shared__ uint32_t sOrd[MAX_ALPHABET_SIZE];
    if (threadIdx.x < alphabetSizeOf<FixedSize>(shape)) {
        sKey[threadIdx.x] = d_alphabetKey[threadIdx.x];
        sOrd[threadIdx.x] = d_alphabetOrd[threadIdx.x];
    }
    __syncthreads();

    // This block's batch - the same for the whole block, so reading it from
    // the kernel's arguments with blockIdx.y costs nothing per thread.
    const uint32_t batchIndex = blockIdx.y;
    const LaunchBatch batch = batches.batch[batchIndex];

    // Every chunk of every group the batch touches - or if Listed, of every
    // group in the batch's list, those of them that aren't pruned - whatever
    // the grid's size: the number of threads launched only decides how the
    // chunks are shared out (normally one each), never which of them get
    // searched. A batch has at most 2^31 rows, so fewer chunks than that.
    const uint32_t chunks = chunksPerGroupOf<FixedSize>(shape);
    const uint32_t stride = gridDim.x * blockDim.x;
    if constexpr (Listed) {
        const uint32_t* groups = bufs.groups + batch.groupsOffset;
        const uint32_t chunkCount = batch.groupCount * chunks;
        for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunkCount; c += stride) {
            const uint32_t groupIndex = divideByChunksPerGroup<FixedSize>(c, shape);
            const uint32_t entry = __ldg(&groups[groupIndex]);
            const uint32_t group = entry >> kRowFlagBits;
            searchChunk<FixedSize, SuffixLen, true, Inserted>(group, __ldg(&bufs.rowMasks[entry & (kRowFlagCount - 1)]), c - groupIndex * chunks, trailingLen, shape,
                                         group == batch.firstGroup ? batch.firstRowD : -1, group == batch.lastGroup ? batch.lastRowD : -1,
                                         batch.firstRowStartK, batch.lastRowEndK, targetA, batch.seed1Start, batch.seed2Start, batchIndex,
                                         bufs, sKey, sOrd);
        }
    } else {
        const uint32_t chunkCount = (batch.lastGroup - batch.firstGroup + 1) * chunks;
        for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunkCount; c += stride) {
            const uint32_t groupIndex = divideByChunksPerGroup<FixedSize>(c, shape);
            const uint32_t group = batch.firstGroup + groupIndex;
            searchChunk<FixedSize, SuffixLen, false, Inserted>(group, 0, c - groupIndex * chunks, trailingLen, shape, groupIndex == 0 ? batch.firstRowD : -1,
                                          group == batch.lastGroup ? batch.lastRowD : -1, batch.firstRowStartK, batch.lastRowEndK, targetA,
                                          batch.seed1Start, batch.seed2Start, batchIndex, bufs, sKey, sOrd);
        }
    }
}

// The suffix length has to be dispatched to one of a fixed set of
// compile-time template instantiations, so that the suffix loop is unrolled
// with its keys as constant operands - dispatchSuffixLen calls `f` with a
// std::integral_constant of the matching value. The alphabet's size needs no
// such thing: what the kernel divides by it, it divides with FastDivisor.
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
        case 9: f(std::integral_constant<int, 9>{}); break;
        case 10: f(std::integral_constant<int, 10>{}); break;
        case 11: f(std::integral_constant<int, 11>{}); break;
        case 12: f(std::integral_constant<int, 12>{}); break;
        default: f(std::integral_constant<int, kRuntimeSuffix>{}); break;
    }
}

// The alphabet sizes filteredRowsKernel has compiled in, rather than taking
// them at runtime (see alphabetSizeOf): 40, 42 and 43 - those of the
// coordinator's targets. Every other size is taken at runtime. Adding one is
// a number here - and 28 more instantiations of the kernel to compile. 40,
// with a whole row group per thread (NAMEBREAK_ROWS_PER_THREAD), issued 4.4%
// fewer instructions than at runtime walking every row group (Nsight
// Compute: 149M instead of 156M, 1.50 ms instead of 1.53, at the profiler's
// fixed clocks), and searched about 4% faster at the GPU's own clocks.
template<int... Sizes>
struct AlphabetSizeList {};
using CompiledAlphabetSizes = AlphabetSizeList<40, 42, 43>;

// Launches filteredRowsKernel, walking every row group or lists of them -
// with the text inserted into the trailing part hashed if Inserted.
template<int FixedSize, int SuffixLen, bool Inserted>
void launchSearch(bool listed, dim3 blocks, int trailingLen, const AlphabetShape& shape, const LaunchBatches& batches, uint32_t targetA,
                  const DeviceBuffers& bufs) {
    if (listed)
        filteredRowsKernel<FixedSize, SuffixLen, true, Inserted><<<blocks, kThreadsPerBlock>>>(trailingLen, shape, batches, targetA, bufs);
    else
        filteredRowsKernel<FixedSize, SuffixLen, false, Inserted><<<blocks, kThreadsPerBlock>>>(trailingLen, shape, batches, targetA, bufs);
}

// launchSearch with the size compiled in if it's one of Sizes, and taken at
// runtime if not. A search that inserts text into the trailing part gets a
// kernel of its own - the insertions in every kernel cost a search without
// any about 4% - but only with the size taken at runtime, which saves
// compiling 40 more.
template<int SuffixLen, int... Sizes>
void launchSearchFor(AlphabetSizeList<Sizes...>, bool listed, bool inserted, dim3 blocks, int trailingLen, const AlphabetShape& shape,
                     const LaunchBatches& batches, uint32_t targetA, const DeviceBuffers& bufs) {
    static_assert(((Sizes >= 1 && Sizes <= MAX_ALPHABET_SIZE) && ...), "a compiled-in alphabet size is out of range");
    if (inserted) {
        launchSearch<0, SuffixLen, true>(listed, blocks, trailingLen, shape, batches, targetA, bufs);
        return;
    }
    const bool compiled =
        ((shape.size == (uint32_t) Sizes ? (launchSearch<Sizes, SuffixLen, false>(listed, blocks, trailingLen, shape, batches, targetA, bufs), true)
                                         : false) ||
         ...);
    if (!compiled)
        launchSearch<0, SuffixLen, false>(listed, blocks, trailingLen, shape, batches, targetA, bufs);
}


// --- Dictionary searches -----------------------------------------------------
//
// A dictionary search (engine/dictionary_search.h) hands the backend batches:
// a leading part's hash states, followed by a run of the word list's words
// and then the suffix - one candidate per word. dictionaryKernel hashes each
// of them on from its batch's states: hashA, or the basename hash instead,
// or both (see dictionaryHashes, and backends/common/dictionary_batch.h for
// the host's side of it).

// The word list and the separators, as DictionaryWordTable has them, on the
// GPU - a kernel argument.
struct DictionaryWords {
    const uint32_t* chars;
    const DictionaryWordEntry* entries;
    const uint32_t* positions;
    uint32_t count;
    uint32_t separatorCount;
};

// What a dictionary launch reports: how many hits of each kind it had -
// counts[0] hashA hits, counts[1] basename hits, every one of them, past
// `capacity` too - and the first `capacity` of each. A launch that has more
// is searched again, with room for them all (see runDictionaryBatches).
struct DictionaryResults {
    int* counts;
    DictionaryHit* hashAHits;
    DictionaryHit* basenameHits;
    uint32_t capacity;
};

// The suffix's keys (dictionarySuffixKeys), for the kernels that have its
// length compiled in - read at compile-time indices, which fold into their
// instructions as constant-bank operands. A longer suffix's are read from
// a buffer of the search's.
__device__ __constant__ uint32_t d_dictionarySuffixKeys[3 * kMaxSuffixSize];

// The batch block `segment` is in: the last of the launch's batches whose
// first segment is at most `segment` (dictionaryBatchOfSegment on the host).
__device__ __forceinline__ uint32_t dictionaryBatchOf(const DictionaryLaunchBatch* batches, uint32_t batchCount, uint32_t segment) {
    uint32_t lo = 0, hi = batchCount - 1;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1) / 2;
        if (__ldg(&batches[mid].firstSegment) <= segment)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// One step of the hash, for a character whose key and value plus 3 are
// known - mpqStep, its sums in another order (see dictionarySuffixKeys).
__device__ __forceinline__ void mpqStepPlus3(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ordPlus3) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = seed1 + 33 * seed2 + ordPlus3;
}

// Byte b (0: the lowest) of `four`. On NVIDIA's GPUs one instruction (PRMT)
// rather than a shift and a mask.
__device__ __forceinline__ uint32_t extractByte(uint32_t four, int b) {
#if defined(__HIP_PLATFORM_AMD__)
    return (four >> (8 * b)) & 0xFF;
#else
    return __byte_perm(four, 0, 0x4440 | b);
#endif
}

// Whether the candidate whose word leaves the state at (seed1, seed2) - of
// hashA, or if Basename, of the basename hash - matches `target`: if the
// suffix filter `filter` lets it through, the suffix hashed in full.
template<int SuffixLen, bool Basename>
__device__ __forceinline__ bool dictionarySuffixMatches(uint32_t seed1, uint32_t seed2, const uint32_t* filter, const uint32_t* suffixKeys,
                                                        int suffixLen, uint32_t target) {
    const uint32_t index = lowBitsFilterIndex(seed1, seed2, kDictionaryFilterBits);
    if (!(filter[index / 32] >> (index % 32) & 1))
        return false;
    if constexpr (SuffixLen == kRuntimeSuffix) {
        for (int i = 0; i < suffixLen; ++i)
            mpqStepPlus3(seed1, seed2, __ldg(&suffixKeys[3 * i + (Basename ? 1 : 0)]), __ldg(&suffixKeys[3 * i + 2]));
    } else {
        #pragma unroll
        for (int i = 0; i < SuffixLen; ++i)
            mpqStepPlus3(seed1, seed2, d_dictionarySuffixKeys[3 * i + (Basename ? 1 : 0)], d_dictionarySuffixKeys[3 * i + 2]);
    }
    return Basename ? basenameKeyMatches(seed1, target) : hashAMatches(seed1, target);
}

// The states after `batch`'s leading part and pair `pair` - its word, then
// its separator (see DictionaryBatch) - as (seed1, seed2, basenameSeed1,
// basenameSeed2): hashA's, the basename hash's (which a '\' starts over),
// or both, as HashA and Basename say. Once a pair, not once a candidate, so
// a character at a time.
template<bool HashA, bool Basename>
__device__ uint4 dictionaryPairStates(const DictionaryLaunchBatch& batch, const DictionaryWords& words, const uint2* sKeys, uint32_t pair) {
    uint32_t seed1 = batch.seed1, seed2 = batch.seed2, key1 = batch.basenameSeed1, key2 = batch.basenameSeed2;
    const uint32_t word = pair / words.separatorCount;
    const uint32_t at[2] = {__ldg(&words.positions[word]), words.count + (pair - word * words.separatorCount)};
    for (int text = 0; text < 2; ++text) {
        const uint4 entry = __ldg(reinterpret_cast<const uint4*>(&words.entries[at[text]]));
        uint32_t four = 0;
        for (uint32_t c = 0; c < entry.y; ++c) {
            if ((c & 3) == 0)
                four = __ldg(&words.chars[entry.x + c / 4]);
            const uint32_t ch = extractByte(four, c & 3);
            const uint2 keys = sKeys[ch];
            if constexpr (HashA)
                mpqStepPlus3(seed1, seed2, keys.x, ch + 3);
            if constexpr (Basename) {
                if (ch == '\\') {
                    key1 = 0x7FED7FED;
                    key2 = 0xEEEEEEEE;
                } else {
                    mpqStepPlus3(key1, key2, keys.y, ch + 3);
                }
            }
        }
    }
    return make_uint4(seed1, seed2, key1, key2);
}

// One block per segment of the launch (see planDictionaryLaunch): up to
// kDictionaryWordsPerThread * blockDim.x cells of one batch - a cell a word
// and a chunk of up to kDictionaryTailsPerCell of the batch's tails - of one
// pair or of several whole ones (see DictionaryLaunchBatch). The block's
// threads take neighbouring cells, a run each group of them
// (dictionaryLanesPerRun): neighbouring words with the same chunk of tails,
// the words in the word table's order (by length, see DictionaryWordTable)
// if the batch has every word, in the list's if not. A thread hashes its
// word from the states after the batch's leading part and the cell's pair -
// which the block works out first - or without pairs, the batch's, then
// each tail of its chunk from there - with Tails; without, there's only ""
// - and then the suffix, where its filter lets it through: hashA, the
// basename hash, or both, as Hashes says. Records every hashA hit and every
// basename hit, as (batch, word, tail, pair): the host rebuilds and checks
// them - and, of a basename hit with the basename hash alone, its hashA and
// hashB (DictionaryHitVerifier).
//
// `cryptKeys` is the crypt table's hashA part (0x100) and then its basename
// hash's part (0x300), 256 entries each, which the block copies into shared
// memory, a character's two keys side by side: a word's characters are a
// lookup each, different for every lane. `filters` is the suffix filters
// (buildDictionarySuffixFilter), hashA's and then the basename hash's,
// kDictionaryFilterWords each - the ones it uses into shared memory too.
// `tails` is the tails (dictionaryTailTable), which a warp's lanes read
// together.
template<int SuffixLen, DictionaryHashes Hashes, bool Tails>
#if defined(__HIP_PLATFORM_AMD__)
__global__ void __launch_bounds__(kDictionaryThreadsPerBlock)
#else
__global__ void __launch_bounds__(kDictionaryThreadsPerBlock, kDictionaryMinBlocksPerSm)
#endif
dictionaryKernel(const DictionaryLaunchBatch* __restrict__ batches, uint32_t batchCount, DictionaryWords words, const uint4* __restrict__ tails,
                 const uint32_t* __restrict__ cryptKeys, const uint32_t* __restrict__ filters, const uint32_t* __restrict__ suffixKeys,
                 int suffixLen, uint32_t targetA, uint32_t basenameKey, DictionaryResults results) {
    constexpr bool HashA = Hashes != DictionaryHashes::Basename;
    constexpr bool Basename = Hashes != DictionaryHashes::HashA;
    constexpr uint32_t kLanes = kDictionaryLanesPerRun;
    __shared__ uint2 sKeys[256];
    __shared__ uint32_t sFilterA[HashA ? kDictionaryFilterWords : 1];
    __shared__ uint32_t sFilterBasename[Basename ? kDictionaryFilterWords : 1];
    __shared__ uint4 sPairs[kDictionaryPairSlots];
    // The block's batch, its segment's first pair (of the batch's, counted
    // from its first) and how many of its pairs it has, and its cells, from
    // its first pair's first: [sCellFirst, sCellEnd).
    __shared__ uint32_t sBatch, sPairFirst, sPairCount, sCellFirst, sCellEnd;
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        sKeys[i] = make_uint2(__ldg(&cryptKeys[i]), __ldg(&cryptKeys[256 + i]));
    for (int i = threadIdx.x; i < (int) kDictionaryFilterWords; i += blockDim.x) {
        if constexpr (HashA)
            sFilterA[i] = __ldg(&filters[i]);
        if constexpr (Basename)
            sFilterBasename[i] = __ldg(&filters[kDictionaryFilterWords + i]);
    }
    if (threadIdx.x == 0) {
        const uint32_t b = dictionaryBatchOf(batches, batchCount, blockIdx.x);
        const DictionaryLaunchBatch& batch = batches[b];
        const uint32_t segment = blockIdx.x - batch.firstSegment, pairCells = batch.wordCount * batch.tailChunks;
        const uint32_t pairs = max(batch.pairCount, 1u), cellsPerSegment = kDictionaryWordsPerThread * kDictionaryThreadsPerBlock;
        uint32_t pairFirst, pairCount, cellFirst = 0, cellEnd;
        if (batch.segmentsPerPair > 1) {
            // A segment of a pair (see DictionaryLaunchBatch).
            pairFirst = segment / batch.segmentsPerPair;
            pairCount = 1;
            cellFirst = (segment - pairFirst * batch.segmentsPerPair) * cellsPerSegment;
            cellEnd = min(cellFirst + cellsPerSegment, pairCells);
        } else {
            // Pairs of a segment.
            pairFirst = segment * batch.pairsPerSegment;
            pairCount = pairFirst < pairs ? min(batch.pairsPerSegment, pairs - pairFirst) : 0;
            cellEnd = pairCount * pairCells;
        }
        // A segment past the batch's last pair has none of its cells.
        if (pairFirst >= pairs) {
            pairCount = 0;
            cellEnd = cellFirst;
        }
        sBatch = b;
        sPairFirst = pairFirst;
        sPairCount = pairCount;
        sCellFirst = cellFirst;
        sCellEnd = cellEnd;
        // Without pairs, the batch's states are those every cell's word is
        // hashed on from - as its one pair's.
        sPairs[0] = make_uint4(batch.seed1, batch.seed2, batch.basenameSeed1, batch.basenameSeed2);
    }
    __syncthreads();

    const uint32_t batchIndex = sBatch;
    const DictionaryLaunchBatch batch = batches[batchIndex];
    const bool wholeList = batch.wordCount == words.count;
    const uint32_t pairCells = batch.wordCount * batch.tailChunks; // a pair's (all, without pairs)
    const uint32_t cellFirst = sCellFirst, cellEnd = sCellEnd;
    if (batch.pairCount) {
        // The states after the leading part and each of the segment's pairs.
        const uint32_t pairFirst = batch.firstPair + sPairFirst, pairs = sPairCount;
        for (uint32_t p = threadIdx.x; p < pairs; p += blockDim.x)
            sPairs[p] = dictionaryPairStates<HashA, Basename>(batch, words, sKeys, pairFirst + p);
        __syncthreads();
    }
    // The segment's cells, in runs of perThread * kLanes, each of kLanes
    // threads: the thread's are `left` cells, kLanes apart, from
    // threadFirst on.
    const uint32_t perThread = (cellEnd - cellFirst + kDictionaryThreadsPerBlock - 1) / kDictionaryThreadsPerBlock;
    const uint32_t threadFirst = cellFirst + threadIdx.x / kLanes * kLanes * perThread + threadIdx.x % kLanes;
    uint32_t left = threadFirst < cellEnd ? min(perThread, (cellEnd - threadFirst + kLanes - 1) / kLanes) : 0;
    // Its cells pair by pair: the pair's slot of sPairs, and the cell of that
    // pair's.
    uint32_t slot = 0, within = threadFirst;
    if (within >= pairCells) {
        slot = within / pairCells;
        within -= slot * pairCells;
    }
    const uint32_t hitPairFirst = batch.firstPair + sPairFirst;
    while (left) {
        const uint4 states = sPairs[slot];
        const uint32_t pairSeed1 = states.x, pairSeed2 = states.y, pairKey1 = states.z, pairKey2 = states.w;
        const uint32_t hitPair = hitPairFirst + slot;
        // The thread's cells of the pair: `n` of them, from `within`.
        const uint32_t n = min(left, (pairCells - within + kLanes - 1) / kLanes);
        left -= n;
        for (const uint32_t stop = within + n * kLanes; within < stop; within += kLanes) {
            // The cell's word, and its chunk of the tails.
            uint32_t i = within, tailFirst = batch.firstTail, tailEnd = batch.firstTail + batch.tailCount;
            if (Tails && batch.tailChunks > 1) {
                const uint32_t chunk = within / batch.wordCount;
                i = within - chunk * batch.wordCount;
                tailFirst += chunk * kDictionaryTailsPerCell;
                tailEnd = min(tailFirst + kDictionaryTailsPerCell, tailEnd);
            }
            const uint4 entry = __ldg(reinterpret_cast<const uint4*>(&words.entries[wholeList ? i : __ldg(&words.positions[batch.firstWord + i])]));
            const uint32_t offset = entry.x, length = entry.y, basenameStart = entry.z, index = entry.w;
            uint32_t seed1 = pairSeed1, seed2 = pairSeed2;
            uint32_t key1 = pairKey1, key2 = pairKey2;
            auto step = [&](uint32_t four, int b) {
                const uint32_t ch = extractByte(four, b);
                const uint2 keys = sKeys[ch];
                if constexpr (HashA)
                    mpqStepPlus3(seed1, seed2, keys.x, ch + 3);
                if constexpr (Basename)
                    mpqStepPlus3(key1, key2, keys.y, ch + 3);
            };
            if (!Basename || basenameStart == 0) {
                // Four characters at a time, as they're stored, and then the
                // rest. The lanes of a warp mostly have words of the same
                // length (see DictionaryWordTable), so they go round
                // together.
                const uint32_t* chars = words.chars + offset;
                const uint32_t* const fullEnd = chars + length / 4;
                for (; chars != fullEnd; ++chars) {
                    const uint32_t four = __ldg(chars);
                    step(four, 0);
                    step(four, 1);
                    step(four, 2);
                    step(four, 3);
                }
                const uint32_t rest = length % 4;
                if (rest != 0) {
                    const uint32_t four = __ldg(chars);
                    step(four, 0);
                    if (rest > 1)
                        step(four, 1);
                    if (rest > 2)
                        step(four, 2);
                }
            } else {
                // A word with a '\': its basename starts over after the
                // last one - and without hashA, what comes before that isn't
                // hashed at all.
                key1 = 0x7FED7FED;
                key2 = 0xEEEEEEEE;
                const uint32_t from = HashA ? 0 : basenameStart;
                uint32_t four = 0;
                for (uint32_t c = from; c < length; ++c) {
                    if (c == from || (c & 3) == 0)
                        four = __ldg(&words.chars[offset + c / 4]);
                    const uint32_t ch = extractByte(four, c & 3);
                    const uint2 keys = sKeys[ch];
                    if constexpr (HashA)
                        mpqStepPlus3(seed1, seed2, keys.x, ch + 3);
                    if (c >= basenameStart)
                        mpqStepPlus3(key1, key2, keys.y, ch + 3);
                }
            }
            // The suffix after tail `tail`, which has left the states at
            // (s1, s2) and (k1, k2).
            auto check = [&](uint32_t s1, uint32_t s2, uint32_t k1, uint32_t k2, uint32_t tail) {
                if constexpr (HashA) {
                    if (dictionarySuffixMatches<SuffixLen, false>(s1, s2, sFilterA, suffixKeys, suffixLen, targetA)) {
                        const int slot = atomicAdd(&results.counts[0], 1);
                        if ((uint32_t) slot < results.capacity)
                            results.hashAHits[slot] = DictionaryHit{batchIndex, index, tail, hitPair};
                    }
                }
                if constexpr (Basename) {
                    if (dictionarySuffixMatches<SuffixLen, true>(k1, k2, sFilterBasename, suffixKeys, suffixLen, basenameKey)) {
                        const int slot = atomicAdd(&results.counts[1], 1);
                        if ((uint32_t) slot < results.capacity)
                            results.basenameHits[slot] = DictionaryHit{batchIndex, index, tail, hitPair};
                    }
                }
            };
            if constexpr (!Tails) {
                check(seed1, seed2, key1, key2, tailFirst);
            } else {
                // Each tail on from the word's states: its characters,
                // eight at most, two uint32s of them.
                for (uint32_t t = tailFirst; t < tailEnd; ++t) {
                    const uint4 tail = __ldg(&tails[t]);
                    uint32_t s1 = seed1, s2 = seed2, k1 = key1, k2 = key2;
                    for (uint32_t c = 0; c < tail.z; ++c) {
                        const uint32_t ch = extractByte(c < 4 ? tail.x : tail.y, c & 3);
                        const uint2 keys = sKeys[ch];
                        if constexpr (HashA)
                            mpqStepPlus3(s1, s2, keys.x, ch + 3);
                        if constexpr (Basename)
                            mpqStepPlus3(k1, k2, keys.y, ch + 3);
                    }
                    check(s1, s2, k1, k2, t);
                }
            }
        }
        // On to the thread's next pair - or one after that, if a pair has
        // fewer cells than kLanes.
        if (!left)
            break;
        within -= pairCells;
        ++slot;
        if (within >= pairCells) {
            const uint32_t pairs = within / pairCells;
            slot += pairs;
            within -= pairs * pairCells;
        }
    }
}

namespace {

// --- Meet-in-the-middle searches -------------------------------------------------------
//
// The backward half of a meet-in-the-middle search
// (engine/meet_in_the_middle.h): every tail of a batch rewound from the
// meeting point, and the state it reaches looked up in the table the engine
// built of the forward half. The table is the one thing here that doesn't fit
// in a kernel argument - up to a gigabyte and more of keys, uploaded once per
// search by beginMeetInTheMiddleSearch.
//
// One thread takes one *row*: the alphabetSize tails that share every
// character but the one rewound last - the first character of the tail,
// which is the directory's own. Everything before it, and the pruning rules
// over it, is then done once for the whole row, and a tail costs one step of
// the hash undone and one lookup. (A row here is the mirror of the search
// kernel's, whose rows share everything but the candidate's *last*
// character: the walk rewinds a tail's last character first, so it's that
// one the most tails have in common.)

// What the pruning rules make of one character - as much of them as a tail
// can be judged by (see tailPruneStep, engine/meet_in_the_middle.h).
constexpr uint32_t kMeetInTheMiddleSymbol = 1;    // neither alphanumeric nor a space
constexpr uint32_t kMeetInTheMiddleBackslash = 2;

// The alphabet, as the meet-in-the-middle kernel needs each character: its
// hashA crypt-table key, its own value, and those flags. In constant memory,
// as the search kernel's d_alphabetKey/d_alphabetOrd are - every lane of a
// warp usually reads the same entry (the rows of a warp differ in their high
// digits, and their k loops run in step), which the constant cache serves
// without serializing.
struct MeetInTheMiddleChar {
    uint32_t key;
    uint32_t ord;
    uint32_t flags;
};
__device__ __constant__ MeetInTheMiddleChar d_meetInTheMiddleChars[MAX_ALPHABET_SIZE];

// The rules themselves - PruneRules, less the two only a whole directory
// settles - and the state of them over the characters rewound so far
// (TailPruneState).
struct MeetInTheMiddleRules {
    int symbolRuns;
    int adjacentBackslashes;
    int maxBackslashCount;
    int any;
};
struct MeetInTheMiddlePrune {
    int symbolRun;
    int backslashes;
    int nextIsBackslash;
};

// Steps `state` over a character, to the left of the ones so far; false if it
// breaks one of `rules`. Must agree with tailPruneStep
// (engine/meet_in_the_middle.cpp) exactly, or the GPU would search other
// tails than the CPU.
__device__ __forceinline__ bool meetInTheMiddlePruneStep(const MeetInTheMiddleRules& rules, MeetInTheMiddlePrune& state, uint32_t flags) {
    if (flags & kMeetInTheMiddleSymbol) {
        if (++state.symbolRun >= 3 && rules.symbolRuns)
            return false;
        state.symbolRun = state.symbolRun < 2 ? state.symbolRun : 2;
    } else {
        state.symbolRun = 0;
    }
    if (flags & kMeetInTheMiddleBackslash) {
        if (state.nextIsBackslash && rules.adjacentBackslashes)
            return false;
        if (++state.backslashes > rules.maxBackslashCount && rules.maxBackslashCount != 0)
            return false;
    }
    state.nextIsBackslash = (flags & kMeetInTheMiddleBackslash) != 0;
    return true;
}

// One step of the hash undone: the state the character `ord`, whose crypt
// table key is `key`, was hashed onto. Must match hashStepBack
// (engine/meet_in_the_middle.h) exactly - 0x3E0F83E1 is 33's inverse modulo
// 2^32.
__device__ __forceinline__ void mpqStepBack(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    const uint32_t before2 = (seed2 - ord - seed1 - 3) * 0x3E0F83E1u;
    seed1 = (seed1 ^ key) - before2;
    seed2 = before2;
}

// What a meet-in-the-middle launch reports: how many tails it looked up in
// the table (the rules skipped the rest) and how many of them the table held
// - every one of them, past `capacity` too - and the first `capacity` hits,
// as the tails' indices. A launch with more hits than it had room for is
// launched again with room for them all (see runMeetInTheMiddleBatch), as a
// dictionary launch is.
struct MeetInTheMiddleResults {
    unsigned int* counts; // counts[0]: tails looked up, counts[1]: hits
    uint64_t* hits;
    uint32_t capacity;
};

// Everything else one launch needs - a kernel argument, so passed by value.
struct MeetInTheMiddleLaunch {
    const uint64_t* keys; // the table's, sorted
    const uint64_t* filter;
    uint64_t keyCount;
    uint64_t firstRow;
    uint64_t lastRow;
    uint32_t meetingPointSeed1;
    uint32_t meetingPointSeed2;
    uint32_t alphabetSize;
    uint32_t firstRowStartK; // the batch starts mid-row at its first row, ends mid-row at its last
    uint32_t lastRowEndK;
    int filterShift;
    int tailLength;
    MeetInTheMiddleRules rules;
    MeetInTheMiddlePrune entry; // the rules' state to the directory's right (tailPruneStart)
};

// Whether the table holds `key`: the filter first - all but about one lookup
// in MeetInTheMiddleTable::kFilterBitsPerKey stops there, without touching
// the keys
// - and then a binary search of the keys themselves.
__device__ __forceinline__ bool meetInTheMiddleTableHolds(const MeetInTheMiddleLaunch& p, uint64_t key) {
    const uint64_t bit = (key * 0x9E3779B97F4A7C15ull) >> p.filterShift;
    if (!((__ldg(&p.filter[bit >> 6]) >> (bit & 63)) & 1))
        return false;
    uint64_t lo = 0, hi = p.keyCount;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (__ldg(&p.keys[mid]) < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < p.keyCount && __ldg(&p.keys[lo]) == key;
}

__global__ void meetInTheMiddleKernel(MeetInTheMiddleLaunch p, MeetInTheMiddleResults results) {
    // The launch's tails looked up, counted per block and added to the
    // global count once - an atomic per tail would cost more than the
    // lookups do.
    __shared__ unsigned int sProbed;
    if (threadIdx.x == 0)
        sProbed = 0;
    __syncthreads();

    const uint64_t row = p.firstRow + (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int probed = 0;
    if (row <= p.lastRow) {
        uint32_t seed1 = p.meetingPointSeed1, seed2 = p.meetingPointSeed2;
        MeetInTheMiddlePrune prune = p.entry;
        bool pruned = false;
        // The characters rewound before the row's own: the row's digits, most
        // significant first - the directory's last character first (see
        // MeetInTheMiddleSpace for the numbering).
        if (p.tailLength > 1) {
            uint32_t digit[MAX_CANDIDATE_LEN];
            uint64_t rest = row;
            for (int i = p.tailLength - 2; i >= 0; --i) {
                digit[i] = (uint32_t) (rest % p.alphabetSize);
                rest /= p.alphabetSize;
            }
            for (int i = 0; i < p.tailLength - 1; ++i) {
                const MeetInTheMiddleChar c = d_meetInTheMiddleChars[digit[i]];
                if (p.rules.any && !meetInTheMiddlePruneStep(p.rules, prune, c.flags)) {
                    // Every tail of the row breaks the rules, and so does
                    // every row whose digits start like this one's - the
                    // engine's own walk skips them all at once; here they
                    // are threads that have nothing to do.
                    pruned = true;
                    break;
                }
                mpqStepBack(seed1, seed2, c.key, c.ord);
            }
        }
        if (!pruned && p.tailLength == 0) {
            // The one tail of no characters: the meeting point itself, which is
            // where a directory of exactly the table's length ends.
            probed = 1;
            if (meetInTheMiddleTableHolds(p, meetingPointKey(seed1, seed2))) {
                const unsigned int slot = atomicAdd(&results.counts[1], 1u);
                if (slot < results.capacity)
                    results.hits[slot] = 0;
            }
        } else if (!pruned) {
            const uint32_t fromK = row == p.firstRow ? p.firstRowStartK : 0;
            const uint32_t toK = row == p.lastRow ? p.lastRowEndK : p.alphabetSize;
            for (uint32_t k = fromK; k < toK; ++k) {
                const MeetInTheMiddleChar c = d_meetInTheMiddleChars[k];
                MeetInTheMiddlePrune last = prune;
                if (p.rules.any && !meetInTheMiddlePruneStep(p.rules, last, c.flags))
                    continue;
                uint32_t s1 = seed1, s2 = seed2;
                mpqStepBack(s1, s2, c.key, c.ord);
                ++probed;
                if (meetInTheMiddleTableHolds(p, meetingPointKey(s1, s2))) {
                    const unsigned int slot = atomicAdd(&results.counts[1], 1u);
                    if (slot < results.capacity)
                        results.hits[slot] = row * p.alphabetSize + k;
                }
            }
        }
    }
    atomicAdd(&sProbed, probed);
    __syncthreads();
    if (threadIdx.x == 0 && sProbed != 0)
        atomicAdd(&results.counts[0], sProbed);
}

class CudaBackend : public SearchBackend {
public:
    // Not CUDA_CHECK'd, and its result deliberately ignored: this can run as
    // the process exits, after the CUDA runtime itself has shut down.
    ~CudaBackend() override {
        if (pinnedResults_)
            (void) cudaFreeHost(pinnedResults_);
        if (groups_)
            (void) cudaFree(groups_);
        if (rowMasks_)
            (void) cudaFree(rowMasks_);
        freeDictionaryBuffers();
        if (dictionaryCounts_)
            (void) cudaFree(dictionaryCounts_);
        if (pinnedDictionaryCounts_)
            (void) cudaFreeHost(pinnedDictionaryCounts_);
        if (dictionaryBatches_)
            (void) cudaFree(dictionaryBatches_);
        for (DictionaryHit* hits : {dictionaryHashAHits_, dictionaryBasenameHits_}) {
            if (hits)
                (void) cudaFree(hits);
        }
        freeMeetInTheMiddleBuffers();
        if (pinnedMeetInTheMiddleCounts_)
            (void) cudaFreeHost(pinnedMeetInTheMiddleCounts_);
    }

#ifdef NAMEBREAK_HIP
    const char* name() const override { return "hip"; }
#else
    const char* name() const override { return "cuda"; }
#endif
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

    // A dictionary search: one launch of dictionaryKernel per call.
    bool supportsDictionary() const override { return true; }
    uint64_t dictionaryCandidatesPerCall() const override { return NAMEBREAK_DICTIONARY_CANDIDATES_PER_LAUNCH; }
    void beginDictionarySearch(const DictionaryConstants& constants) override;
    DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches) override;
    void endDictionarySearch() override;

    // A meet-in-the-middle search: one launch of meetInTheMiddleKernel per
    // call, over about NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_LAUNCH tails
    // (tuning.h).
    bool supportsMeetInTheMiddle() const override { return true; }
    uint64_t meetInTheMiddleTailsPerCall() const override { return NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_LAUNCH; }
    void beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) override;
    MeetInTheMiddleOutcome runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) override;
    void endMeetInTheMiddleSearch() override;

private:
    int alphabetSize_ = 0;
    // The alphabet's size, and the divisions the kernel does by it - see
    // AlphabetShape.
    AlphabetShape shape_ = {};
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
    uint64_t* rowMasks_ = nullptr; // DeviceBuffers::rowMasks, allocated by the first search that prunes the whole candidate
    size_t groupsCapacity_ = 0;
    size_t groupsUploaded_ = 0;
    uint64_t groupsGeneration_ = 0;
    void uploadGroups();
    // Whether this search prunes the whole candidate, and so has lists of row
    // groups: only then - a kernel that walks every group measured about 5%
    // faster than one walking a list of them all, so a launch walks them
    // only if they leave out enough groups (see runBatches) - or as
    // SearchConstants::listWalking says.
    bool listed_ = false;
    SearchConstants::ListWalking listWalking_ = SearchConstants::ListWalking::Auto;
    // Whether this search inserts text into the trailing part, and gets the
    // kernel that hashes it.
    bool inserted_ = false;

    // A dictionary search's: the word list, the crypt table's keys and the
    // suffix's (DictionaryWordTable, dictionarySuffixKeys) - uploaded by
    // beginDictionarySearch and freed by endDictionarySearch - and what the
    // kernel is launched with.
    void freeDictionaryBuffers();
    void launchDictionary(uint32_t batchCount, uint64_t segments);
    DictionaryHitVerifier dictionaryVerifier_;
    DictionaryWords dictionaryWords_ = {};
    uint32_t* dictionaryCryptKeys_ = nullptr;
    uint32_t* dictionaryTails_ = nullptr; // dictionaryTailTable
    bool dictionaryHasTails_ = false;
    uint32_t* dictionaryFilters_ = nullptr;
    uint32_t* dictionarySuffixKeys_ = nullptr;
    int dictionarySuffixLen_ = 0;
    DictionaryHashes dictionaryHashes_ = DictionaryHashes::HashA;
    uint32_t dictionaryTargetA_ = 0;
    uint32_t dictionaryBasenameKey_ = 0;
    // Kept from one search to the next, grown as a launch needs: its
    // batches (DictionaryLaunchBatch), the hit counts (and their copy in
    // pinned host memory, read back behind the kernel), and the hits, room
    // for dictionaryHitCapacity_ of each kind.
    std::vector<DictionaryLaunchBatch> dictionaryLaunch_;
    DictionaryLaunchBatch* dictionaryBatches_ = nullptr;
    size_t dictionaryBatchesCapacity_ = 0;
    int* dictionaryCounts_ = nullptr;
    int* pinnedDictionaryCounts_ = nullptr;
    DictionaryHit* dictionaryHashAHits_ = nullptr;
    DictionaryHit* dictionaryBasenameHits_ = nullptr;
    uint32_t dictionaryHitCapacity_ = 0;

    // A meet-in-the-middle search's: the engine's table, copied to the GPU by
    // beginMeetInTheMiddleSearch and freed by endMeetInTheMiddleSearch, and
    // what the kernel is launched with. The counts are read back behind the
    // kernel into pinned host memory, as a dictionary launch's are.
    void freeMeetInTheMiddleBuffers();
    MeetInTheMiddleConstants meet_in_the_middle_ = {};
    uint64_t* meetInTheMiddleKeys_ = nullptr;
    uint64_t* meetInTheMiddleFilter_ = nullptr;
    uint64_t meetInTheMiddleKeyCount_ = 0;
    int meetInTheMiddleFilterShift_ = 64;
    MeetInTheMiddleRules meetInTheMiddleRules_ = {};
    MeetInTheMiddlePrune meetInTheMiddleEntry_ = {};
    unsigned int* meetInTheMiddleCounts_ = nullptr;
    unsigned int* pinnedMeetInTheMiddleCounts_ = nullptr;
    uint64_t* meetInTheMiddleHits_ = nullptr;
    uint32_t meetInTheMiddleHitCapacity_ = 0;
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
    if (alphabetSize_ < 1 || alphabetSize_ > MAX_ALPHABET_SIZE) {
        // runSearch checks the alphabet's size before it begins a search.
        fprintf(stderr, "INTERNAL ERROR: an alphabet of %d characters (1 to %d allowed) - exiting\n", alphabetSize_, MAX_ALPHABET_SIZE);
        exit(1);
    }
    shape_.size = (uint32_t) alphabetSize_;
    shape_.chunksPerGroup = kChunksPerGroup(shape_.size);
    shape_.bySize = makeFastDivisor(shape_.size);
    shape_.byChunksPerGroup = makeFastDivisor(shape_.chunksPerGroup);
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

        // The text inserted into the trailing part - see d_insertLayout.
        const InsertLayout inserted = insertLayoutFor(constants);
        inserted_ = !constants.trailingInsertions.empty();
        TrailingInsertLayout layout = {};
        uint32_t h_insertKey[2 * kMaxInsertLen] = {0};
        uint32_t h_insertOrd[2 * kMaxInsertLen] = {0};
        for (int n = 0; n < 2; ++n) {
            layout.groupCharsAfter[n] = inserted.groupCharsAfter[n];
            layout.groupStart[n] = inserted.groupStart[n];
            layout.groupLen[n] = inserted.groupLen[n];
        }
        layout.rowStart = inserted.rowStart;
        layout.rowLen = inserted.rowLen;
        std::copy(inserted.key.begin(), inserted.key.end(), h_insertKey);
        std::copy(inserted.ord.begin(), inserted.ord.end(), h_insertOrd);
        CUDA_CHECK(cudaMemcpyToSymbol(d_insertLayout, &layout, sizeof(layout)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_insertKey, h_insertKey, sizeof(h_insertKey)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_insertOrd, h_insertOrd, sizeof(h_insertOrd)));
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
    listWalking_ = constants.listWalking;
    if (listed_) {
        rowPruning_.begin(constants);
        if (!rowMasks_)
            CUDA_CHECK(cudaMalloc(&rowMasks_, kRowFlagCount * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(rowMasks_, rowPruning_.rowMasks(), kRowFlagCount * sizeof(uint64_t), cudaMemcpyHostToDevice));
        bufs_.rowMasks = rowMasks_;
    }

    // This search's lookup filter - it depends on the alphabet, the suffix and
    // the target, so it's built for every search. Checked against its
    // definition before it's used, and read back after the upload to make
    // sure the GPU has exactly what was checked.
    const std::vector<uint64_t> table = buildLowBitsFilterTable(constants, kCudaLowBitsFilterBits);
    std::string error;
    if (!checkLowBitsFilterTable(table, constants, kFilterEntriesCheckedPerSearch, 2, std::random_device{}(), error, kCudaLowBitsFilterBits))
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
    // firstRow..lastRow - passed as their groups and rows within them (see
    // LaunchBatch); only the first and last row can be partial (see
    // filteredRowsKernel). With the whole candidate pruned, each also gets
    // the list of its row groups that survive.
    LaunchBatches batches = {};
    uint64_t maxGroups = 0, maxListedGroups = 0;
    uint64_t groups = 0, listedGroups = 0;
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
        batch.firstGroup = (uint32_t) (firstRow / alphabetSize);
        batch.lastGroup = (uint32_t) (lastRow / alphabetSize);
        batch.firstRowD = (int) (firstRow % alphabetSize);
        batch.lastRowD = (int) (lastRow % alphabetSize);
        batch.firstRowStartK = (int) (startIdx - firstRow * alphabetSize);
        batch.lastRowEndK = (int) (endIdx - lastRow * alphabetSize); // in [1, alphabetSize]
        batch.seed1Start = requests[b].params.seed1Start;
        batch.seed2Start = requests[b].params.seed2Start;
        const uint64_t batchGroups = batch.lastGroup - batch.firstGroup + 1;
        maxGroups = std::max(maxGroups, batchGroups);
        groups += batchGroups;
        if (listed_) {
            const RowPruning::Slice list = rowPruning_.groupsFor(trailingLen, requests[b].params.pruneEntry, batch.firstGroup, batch.lastGroup);
            batch.groupsOffset = list.offset;
            batch.groupCount = list.count;
            maxListedGroups = std::max<uint64_t>(maxListedGroups, list.count);
            listedGroups += list.count;
        }
    }
    // Every row of every batch pruned: nothing to launch.
    if (listed_ && maxListedGroups == 0)
        return outcome;
    // Whether this launch walks its batches' lists, or every group they
    // touch - when the lists leave out too few of them to pay for walking
    // them (see NAMEBREAK_LIST_MIN_PRUNED_PERCENT), and the hits they'd have
    // left out are dropped below instead.
    using ListWalking = SearchConstants::ListWalking;
    const bool listed = listed_ && listWalking_ != ListWalking::Never &&
                        (listWalking_ == ListWalking::Always || (groups - listedGroups) * 100 >= groups * NAMEBREAK_LIST_MIN_PRUNED_PERCENT);
    if (listed) {
        maxGroups = maxListedGroups;
        uploadGroups();
    }

    const auto launched = std::chrono::steady_clock::now();
    // One row of blocks per batch (blockIdx.y), and in it one GPU thread per
    // chunk of every row group the batch touches (see kChunksPerGroup) -
    // about NAMEBREAK_ROWS_PER_THREAD rows each - as many as the largest
    // batch needs. The kernel works out each batch's chunks itself and covers
    // all of them whatever the grid, so this count only spreads the work.
    // Every 32 consecutive threads form a "warp" that the hardware runs in
    // lockstep (SIMT) - that grouping is automatic (256 threads/block = 8
    // warps/block here), not something chosen at this call site.
    const uint64_t threads = maxGroups * shape_.chunksPerGroup;
    const dim3 blocks((unsigned) ((threads + kThreadsPerBlock - 1) / kThreadsPerBlock), (unsigned) batchCount);
    dispatchSuffixLen(suffixLen_, [&](auto suffixC) {
        using SuffixC = decltype(suffixC);
        launchSearchFor<SuffixC::value>(CompiledAlphabetSizes{}, listed, inserted_, blocks, trailingLen, shape_, batches, targetA_, bufs_);
    });
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
    // (a bounded search's last launch). If the launch searched every row
    // group where it had lists, the hits in a row the lists leave out are
    // dropped first, as walking the lists would have.
    int kept = 0;
    for (int i = 0; i < hitCount; ++i) {
        const Hit& hit = pinnedResults_->hits[i];
        if (hit.batch >= (uint32_t) batchCount) {
            fprintf(stderr, "INTERNAL ERROR: the kernel reported a hit in batch %u of a launch of %d - exiting\n", hit.batch, batchCount);
            exit(1);
        }
        const LaunchBatch& batch = batches.batch[hit.batch];
        if (listed_ && !listed && !rowPruning_.survives(RowPruning::Slice{batch.groupsOffset, batch.groupCount}, hit.row))
            continue;
        verifier_.addHits({hit.row * alphabetSize_ + hit.k}, trailingLen, requests[hit.batch].params, outcome);
        ++kept;
    }
    outcome.hitCount = kept;
    return outcome;
}


void CudaBackend::beginDictionarySearch(const DictionaryConstants& constants) {
    freeDictionaryBuffers();
    dictionaryVerifier_.begin(constants);
    dictionaryHashes_ = dictionaryVerifier_.hashes();
    dictionaryTargetA_ = constants.targetHashA;
    dictionaryBasenameKey_ = constants.basenameKey;
    dictionarySuffixLen_ = (int) constants.suffix.size();

    DictionaryWordTable table;
    if (!makeDictionaryWordTable(constants.words, constants.separators, table)) {
        // runDictionarySearch's word lists are far smaller.
        fprintf(stderr, "INTERNAL ERROR: the word list is too long for the GPU's 32-bit offsets - exiting\n");
        exit(1);
    }
    // At least an entry each, so that an empty table has its buffers too.
    auto upload = [](const auto& host, auto*& device) {
        using T = std::remove_const_t<std::remove_reference_t<decltype(*device)>>;
        T* buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&buffer, std::max<size_t>(1, host.size()) * sizeof(T)));
        if (!host.empty())
            CUDA_CHECK(cudaMemcpy(buffer, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
        device = buffer;
    };
    upload(table.chars, dictionaryWords_.chars);
    upload(table.entries, dictionaryWords_.entries);
    upload(table.positions, dictionaryWords_.positions);
    dictionaryWords_.count = (uint32_t) constants.words.size();
    dictionaryWords_.separatorCount = (uint32_t) constants.separators.size();
    // The tails - only "" without any, which the kernel built without
    // Tails doesn't read.
    dictionaryHasTails_ = constants.tails != std::vector<std::string>{""};
    upload(dictionaryTailTable(constants.tails), dictionaryTails_);

    std::vector<uint32_t> cryptKeys(constants.cryptTable + kHashAOffset, constants.cryptTable + kHashAOffset + 256);
    cryptKeys.insert(cryptKeys.end(), constants.cryptTable + kFileKeyOffset, constants.cryptTable + kFileKeyOffset + 256);
    upload(cryptKeys, dictionaryCryptKeys_);
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
    upload(filters, dictionaryFilters_);
    // The suffix's keys: in a buffer for any length, and in
    // d_dictionarySuffixKeys for the kernels that have its length compiled
    // in (see dispatchSuffixLen).
    const std::vector<uint32_t> suffixKeys = dictionarySuffixKeys(constants.suffix, constants.cryptTable);
    upload(suffixKeys, dictionarySuffixKeys_);
    if (!suffixKeys.empty() && suffixKeys.size() <= 3 * (size_t) kMaxSuffixSize)
        CUDA_CHECK(cudaMemcpyToSymbol(d_dictionarySuffixKeys, suffixKeys.data(), suffixKeys.size() * sizeof(uint32_t)));

    // The counts are 0 at every launch: zeroed here, and after every launch
    // that had hits (see runDictionaryBatches).
    if (!dictionaryCounts_)
        CUDA_CHECK(cudaMalloc(&dictionaryCounts_, 2 * sizeof(int)));
    if (!pinnedDictionaryCounts_)
        CUDA_CHECK(cudaMallocHost((void**) &pinnedDictionaryCounts_, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(dictionaryCounts_, 0, 2 * sizeof(int)));
    waiter_.beginSearch();
}

void CudaBackend::freeDictionaryBuffers() {
    for (const void* buffer : {(const void*) dictionaryWords_.chars, (const void*) dictionaryWords_.entries,
                               (const void*) dictionaryWords_.positions, (const void*) dictionaryCryptKeys_,
                               (const void*) dictionaryFilters_, (const void*) dictionarySuffixKeys_, (const void*) dictionaryTails_}) {
        if (buffer)
            (void) cudaFree((void*) buffer);
    }
    dictionaryWords_ = {};
    dictionaryCryptKeys_ = nullptr;
    dictionaryFilters_ = nullptr;
    dictionarySuffixKeys_ = nullptr;
    dictionaryTails_ = nullptr;
}

void CudaBackend::endDictionarySearch() {
    CUDA_CHECK(cudaDeviceSynchronize());
    freeDictionaryBuffers();
}

void CudaBackend::beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) {
    meet_in_the_middle_ = constants;
    const MeetInTheMiddleTable& table = *constants.table;

    // The alphabet, as the kernel reads it.
    MeetInTheMiddleChar chars[MAX_ALPHABET_SIZE] = {};
    for (size_t k = 0; k < constants.alphabet.size(); ++k) {
        const unsigned char ch = (unsigned char) constants.alphabet[k];
        chars[k].key = constants.cryptTable[kHashAOffset + ch];
        chars[k].ord = ch;
        chars[k].flags = (isRunSymbol_CPU((char) ch) ? kMeetInTheMiddleSymbol : 0) | (ch == '\\' ? kMeetInTheMiddleBackslash : 0);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_meetInTheMiddleChars, chars, sizeof chars));
    meetInTheMiddleRules_.symbolRuns = constants.rules.symbolRuns;
    meetInTheMiddleRules_.adjacentBackslashes = constants.rules.adjacentBackslashes;
    meetInTheMiddleRules_.maxBackslashCount = constants.rules.maxBackslashCount;
    meetInTheMiddleRules_.any = constants.rules.any();
    meetInTheMiddleEntry_.symbolRun = constants.tailEntry.symbolRun;
    meetInTheMiddleEntry_.backslashes = constants.tailEntry.backslashes;
    meetInTheMiddleEntry_.nextIsBackslash = constants.tailEntry.nextIsBackslash;

    // The table: the one thing a meet-in-the-middle search has that a kernel
    // argument can't carry, and the one thing that can be too big for the GPU
    // - the engine sizes it to the memory it was given, which is the
    // machine's, not this device's.
    meetInTheMiddleKeyCount_ = table.size();
    meetInTheMiddleFilterShift_ = table.filterShift();
    const size_t keyBytes = std::max<size_t>(1, table.size() * sizeof(uint64_t));
    const size_t filterBytes = std::max<size_t>(1, (size_t) table.filterWords() * sizeof(uint64_t));
    if (cudaMalloc(&meetInTheMiddleKeys_, keyBytes) != cudaSuccess || cudaMalloc(&meetInTheMiddleFilter_, filterBytes) != cudaSuccess) {
        size_t freeBytes = 0, totalBytes = 0;
        (void) cudaGetLastError();
        (void) cudaMemGetInfo(&freeBytes, &totalBytes);
        fprintf(stderr,
                "The meet-in-the-middle search's table needs %zu MB on the GPU, and this one has %zu MB free of %zu MB.\n"
                "Give the table less memory (config.conf's table_memory, a character shorter is %u times smaller),\n"
                "or search on the cpu backend (--backend cpu), which uses the machine's own memory.\n",
                (keyBytes + filterBytes) / (1024 * 1024), freeBytes / (1024 * 1024), totalBytes / (1024 * 1024),
                (unsigned) constants.alphabet.size());
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(meetInTheMiddleKeys_, table.keys(), table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(meetInTheMiddleFilter_, table.filter(), (size_t) table.filterWords() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    if (!meetInTheMiddleCounts_) {
        CUDA_CHECK(cudaMalloc(&meetInTheMiddleCounts_, 2 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost((void**) &pinnedMeetInTheMiddleCounts_, 2 * sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaMemset(meetInTheMiddleCounts_, 0, 2 * sizeof(unsigned int)));
    if (!meetInTheMiddleHits_) {
        meetInTheMiddleHitCapacity_ = NAMEBREAK_MEET_IN_THE_MIDDLE_HIT_CAPACITY;
        CUDA_CHECK(cudaMalloc(&meetInTheMiddleHits_, meetInTheMiddleHitCapacity_ * sizeof(uint64_t)));
    }
    waiter_.beginSearch();
}

void CudaBackend::freeMeetInTheMiddleBuffers() {
    for (uint64_t** buffer : {&meetInTheMiddleKeys_, &meetInTheMiddleFilter_}) {
        if (*buffer) {
            (void) cudaFree(*buffer);
            *buffer = nullptr;
        }
    }
}

void CudaBackend::endMeetInTheMiddleSearch() {
    CUDA_CHECK(cudaDeviceSynchronize());
    freeMeetInTheMiddleBuffers();
}

MeetInTheMiddleOutcome CudaBackend::runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) {
    MeetInTheMiddleOutcome outcome;
    if (batch.count == 0)
        return outcome;
    const uint64_t alphabetSize = meet_in_the_middle_.alphabet.size();

    MeetInTheMiddleLaunch p = {};
    p.keys = meetInTheMiddleKeys_;
    p.filter = meetInTheMiddleFilter_;
    p.keyCount = meetInTheMiddleKeyCount_;
    p.meetingPointSeed1 = meet_in_the_middle_.meetingPoint.first;
    p.meetingPointSeed2 = meet_in_the_middle_.meetingPoint.second;
    p.alphabetSize = (uint32_t) alphabetSize;
    p.filterShift = meetInTheMiddleFilterShift_;
    p.tailLength = batch.tailLength;
    p.rules = meetInTheMiddleRules_;
    p.entry = meetInTheMiddleEntry_;
    const MeetInTheMiddleRows rows = meetInTheMiddleRowsOf(batch, alphabetSize);
    p.firstRow = rows.firstRow;
    p.lastRow = rows.lastRow;
    p.firstRowStartK = rows.firstRowStartK;
    p.lastRowEndK = rows.lastRowEndK;
    const uint64_t blocks = (rows.rowCount() + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > UINT32_MAX || batch.count > UINT32_MAX) {
        // runMeetInTheMiddleSearch's calls are far smaller (meetInTheMiddleTailsPerCall).
        fprintf(stderr, "INTERNAL ERROR: a meet-in-the-middle launch of %llu tails, %llu blocks - exiting\n", (unsigned long long) batch.count,
                (unsigned long long) blocks);
        exit(1);
    }

    // Launched again, with room for every hit, as long as it had more than it
    // had room for - which a real search, comparing the meeting point's whole
    // 64-bit state, practically never does.
    unsigned int counts[2];
    for (;;) {
        const MeetInTheMiddleResults results{meetInTheMiddleCounts_, meetInTheMiddleHits_, meetInTheMiddleHitCapacity_};
        const auto launched = std::chrono::steady_clock::now();
        meetInTheMiddleKernel<<<(unsigned) blocks, kThreadsPerBlock>>>(p, results);
        CUDA_CHECK(cudaGetLastError());
        // The counts' copy is queued right behind the kernel, as a
        // dictionary launch's are (see launchDictionary).
        CUDA_CHECK(cudaMemcpyAsync(pinnedMeetInTheMiddleCounts_, meetInTheMiddleCounts_, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost, 0));
        waiter_.wait(batch.count, launched, [] { CUDA_CHECK(cudaDeviceSynchronize()); });
        counts[0] = pinnedMeetInTheMiddleCounts_[0];
        counts[1] = pinnedMeetInTheMiddleCounts_[1];
        if (counts[0] > batch.count || counts[1] > batch.count) {
            fprintf(stderr, "INTERNAL ERROR: a meet-in-the-middle launch of %llu tails looked up %u of them and found %u - exiting\n",
                    (unsigned long long) batch.count, counts[0], counts[1]);
            exit(1);
        }
        if (counts[1] <= meetInTheMiddleHitCapacity_)
            break;
        CUDA_CHECK(cudaFree(meetInTheMiddleHits_));
        meetInTheMiddleHitCapacity_ = std::max(counts[1], 2 * meetInTheMiddleHitCapacity_);
        CUDA_CHECK(cudaMalloc(&meetInTheMiddleHits_, meetInTheMiddleHitCapacity_ * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(meetInTheMiddleCounts_, 0, 2 * sizeof(unsigned int)));
    }

    outcome.probed = counts[0];
    if (counts[1] > 0) {
        std::vector<uint64_t> hits(counts[1]);
        CUDA_CHECK(cudaMemcpy(hits.data(), meetInTheMiddleHits_, counts[1] * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (uint64_t index : hits) {
            if (index < batch.start || index >= batch.start + batch.count) {
                fprintf(stderr, "INTERNAL ERROR: the meet-in-the-middle kernel reported the tail at %llu, which its batch doesn't have - exiting\n",
                        (unsigned long long) index);
                exit(1);
            }
            outcome.hits.push_back({batch.tailLength, index});
        }
    }
    if (counts[0] != 0 || counts[1] != 0)
        CUDA_CHECK(cudaMemsetAsync(meetInTheMiddleCounts_, 0, 2 * sizeof(unsigned int), 0));
    return outcome;
}

void CudaBackend::launchDictionary(uint32_t batchCount, uint64_t segments) {
    const DictionaryResults results{dictionaryCounts_, dictionaryHashAHits_, dictionaryBasenameHits_, dictionaryHitCapacity_};
    dispatchSuffixLen(dictionarySuffixLen_, [&](auto suffixC) {
        using SuffixC = decltype(suffixC);
        auto launch = [&](auto hashesC) {
            const uint4* tails = reinterpret_cast<const uint4*>(dictionaryTails_);
            if (dictionaryHasTails_)
                dictionaryKernel<SuffixC::value, decltype(hashesC)::value, true><<<(unsigned) segments, kDictionaryThreadsPerBlock>>>(
                    dictionaryBatches_, batchCount, dictionaryWords_, tails, dictionaryCryptKeys_, dictionaryFilters_, dictionarySuffixKeys_,
                    dictionarySuffixLen_, dictionaryTargetA_, dictionaryBasenameKey_, results);
            else
                dictionaryKernel<SuffixC::value, decltype(hashesC)::value, false><<<(unsigned) segments, kDictionaryThreadsPerBlock>>>(
                    dictionaryBatches_, batchCount, dictionaryWords_, tails, dictionaryCryptKeys_, dictionaryFilters_, dictionarySuffixKeys_,
                    dictionarySuffixLen_, dictionaryTargetA_, dictionaryBasenameKey_, results);
        };
        switch (dictionaryHashes_) {
            case DictionaryHashes::HashA: launch(std::integral_constant<DictionaryHashes, DictionaryHashes::HashA>()); break;
            case DictionaryHashes::Basename: launch(std::integral_constant<DictionaryHashes, DictionaryHashes::Basename>()); break;
            case DictionaryHashes::Both: launch(std::integral_constant<DictionaryHashes, DictionaryHashes::Both>()); break;
        }
    });
    CUDA_CHECK(cudaGetLastError());
    // The counts' copy is queued right behind the kernel - as the row
    // search's results are (see runBatches).
    CUDA_CHECK(cudaMemcpyAsync(pinnedDictionaryCounts_, dictionaryCounts_, 2 * sizeof(int), cudaMemcpyDeviceToHost, 0));
}

DictionaryOutcome CudaBackend::runDictionaryBatches(const std::vector<DictionaryBatch>& batches) {
    DictionaryOutcome outcome;
    if (batches.empty())
        return outcome;
    const uint32_t cellsPerSegment = (uint32_t) (kDictionaryWordsPerThread * kDictionaryThreadsPerBlock);
    const uint64_t segments = planDictionaryLaunch(batches, cellsPerSegment, dictionaryHasTails_ ? kDictionaryTailsPerCell : 1, dictionaryLaunch_);
    uint64_t candidates = 0;
    for (const DictionaryBatch& batch : batches)
        candidates += dictionaryBatchCandidates(batch);
    // The kernel counts a batch's cells in 32 bits.
    if (segments > INT32_MAX || batches.size() > UINT32_MAX || candidates > UINT32_MAX) {
        // runDictionarySearch's calls are far smaller (dictionaryCandidatesPerCall).
        fprintf(stderr, "INTERNAL ERROR: a dictionary launch of %llu segments, %llu candidates - exiting\n", (unsigned long long) segments,
                (unsigned long long) candidates);
        exit(1);
    }
    if (dictionaryLaunch_.size() > dictionaryBatchesCapacity_) {
        if (dictionaryBatches_)
            CUDA_CHECK(cudaFree(dictionaryBatches_));
        dictionaryBatchesCapacity_ = std::max(dictionaryLaunch_.size(), 2 * dictionaryBatchesCapacity_);
        CUDA_CHECK(cudaMalloc(&dictionaryBatches_, dictionaryBatchesCapacity_ * sizeof(DictionaryLaunchBatch)));
    }
    CUDA_CHECK(cudaMemcpy(dictionaryBatches_, dictionaryLaunch_.data(), dictionaryLaunch_.size() * sizeof(DictionaryLaunchBatch),
                          cudaMemcpyHostToDevice));
    if (!dictionaryHashAHits_) {
        dictionaryHitCapacity_ = NAMEBREAK_DICTIONARY_HIT_CAPACITY;
        CUDA_CHECK(cudaMalloc(&dictionaryHashAHits_, dictionaryHitCapacity_ * sizeof(DictionaryHit)));
        CUDA_CHECK(cudaMalloc(&dictionaryBasenameHits_, dictionaryHitCapacity_ * sizeof(DictionaryHit)));
    }

    // Launched again, with room for every hit, as long as it had more than
    // it had room for - which a real search, with a 32-bit hashA and key,
    // practically never does.
    int counts[2];
    for (;;) {
        const auto launched = std::chrono::steady_clock::now();
        launchDictionary((uint32_t) dictionaryLaunch_.size(), segments);
        waiter_.wait(candidates, launched, [] { CUDA_CHECK(cudaDeviceSynchronize()); });
        counts[0] = pinnedDictionaryCounts_[0];
        counts[1] = pinnedDictionaryCounts_[1];
        if (counts[0] < 0 || counts[1] < 0 || (uint64_t) counts[0] > candidates || (uint64_t) counts[1] > candidates) {
            fprintf(stderr, "INTERNAL ERROR: a dictionary launch of %llu candidates reported %d and %d hits - exiting\n",
                    (unsigned long long) candidates, counts[0], counts[1]);
            exit(1);
        }
        const uint32_t needed = (uint32_t) std::max(counts[0], counts[1]);
        if (needed <= dictionaryHitCapacity_)
            break;
        CUDA_CHECK(cudaFree(dictionaryHashAHits_));
        CUDA_CHECK(cudaFree(dictionaryBasenameHits_));
        dictionaryHitCapacity_ = std::max(needed, 2 * dictionaryHitCapacity_);
        CUDA_CHECK(cudaMalloc(&dictionaryHashAHits_, dictionaryHitCapacity_ * sizeof(DictionaryHit)));
        CUDA_CHECK(cudaMalloc(&dictionaryBasenameHits_, dictionaryHitCapacity_ * sizeof(DictionaryHit)));
        CUDA_CHECK(cudaMemset(dictionaryCounts_, 0, 2 * sizeof(int)));
    }
    if (counts[0] == 0 && counts[1] == 0) {
        dictionaryVerifier_.addHits(batches, {}, {}, outcome);
        return outcome;
    }

    // The exception: hits to bring back, and the counts to zero again.
    std::vector<DictionaryHit> hashAHits(counts[0]), basenameHits(counts[1]);
    if (counts[0] > 0)
        CUDA_CHECK(cudaMemcpy(hashAHits.data(), dictionaryHashAHits_, counts[0] * sizeof(DictionaryHit), cudaMemcpyDeviceToHost));
    if (counts[1] > 0)
        CUDA_CHECK(cudaMemcpy(basenameHits.data(), dictionaryBasenameHits_, counts[1] * sizeof(DictionaryHit), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemsetAsync(dictionaryCounts_, 0, 2 * sizeof(int), 0));
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
