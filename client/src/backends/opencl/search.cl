// The OpenCL backend's search kernel - the CUDA backend's filteredRowsKernel
// (backends/cuda/cuda_backend.cu), ported: one work-item per chunk of a row
// group (see CHUNKS_PER_GROUP below). Embedded into the program at build time and
// compiled by the OpenCL driver at runtime, once per combination of these
// (see opencl_backend.cpp), which makes them all compile-time constants the
// compiler can unroll and fold:
//   ALPHABET_SIZE  the alphabet's size
//   SUFFIX_LEN     the suffix's length
//   TRAILING_LEN   the candidate's trailing (GPU-enumerated) length, >= 1
//   MAX_MATCHES    how many hits one launch can record
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                  in the stress tests' builds (see engine/hash_match.h)
//   FILTER_BITS    kLowBitsFilterBits: how many low bits of each seed index
//                  the lookup filter's table (backends/common/lowbits_filter.h)
//   ROWS_PER_THREAD  about how many rows one work-item searches
//
// Rows are every value of the candidate's last character, for one
// combination of the other trailing characters (see the terminology in
// backends/common/row_batch.h). A work-item hashes the characters its rows
// share once, then one more for each row, and looks each row's state up in
// this search's filter table, which gives the set of last characters whose
// hashA has the target's low bits - always including any that matches the
// target (README.md's "The lookup filter" has why) - and hashes only those,
// in full. It records the trailing index and batch of every hashA hit; the
// host rebuilds and checks those.
//
// A launch searches up to 32 batches - usually each a whole leading value's
// trailing space, with its own seeds - one per row of work-groups
// (get_group_id(1); see runBatches in opencl_backend.cpp).

#ifndef HASHA_MATCH_MASK
#define HASHA_MATCH_MASK 0xFFFFFFFFu
#endif
#define HASHA_MATCHES(a, target) (((a) & HASHA_MATCH_MASK) == ((target) & HASHA_MATCH_MASK))

// lowBitsFilterIndex (backends/common/lowbits_filter.h), which this must
// match exactly: the low FILTER_BITS bits of seed1, then those of seed2.
#define FILTER_STATE_MASK ((1u << FILTER_BITS) - 1u)
#define FILTER_INDEX(seed1, seed2) (((seed1) & FILTER_STATE_MASK) | (((seed2) & FILTER_STATE_MASK) << FILTER_BITS))

#define MPQ_STEP(seed1, seed2, key, ord)                        \
    do {                                                        \
        seed1 = (key) ^ (seed1 + seed2);                        \
        seed2 = (ord) + seed1 + seed2 + (seed2 << 5) + 3;       \
    } while (0)

// A row *group* is the ALPHABET_SIZE consecutive rows that share every row
// character but the last (see kChunksPerGroup in backends/cuda/cuda_backend.cu,
// which this follows). Each group is split into CHUNKS_PER_GROUP chunks of
// about ROWS_PER_THREAD consecutive rows, one per work-item: chunk c is the
// group's rows d = c * ALPHABET_SIZE / CHUNKS_PER_GROUP .. (c + 1) *
// ALPHABET_SIZE / CHUNKS_PER_GROUP - 1, so the chunks cover every row of the
// group exactly once and never cross into another.
#define CHUNKS_PER_GROUP ((ALPHABET_SIZE + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD)

// These three must match their namesakes in opencl_backend.cpp.
// One batch of a launch: its rows firstRow..lastRow of the trailing space,
// cut to [firstRowStartK, lastRowEndK) in its first and last row, and the
// hash state after its prefix.
typedef struct {
    uint firstRow;
    uint lastRow;
    int firstRowStartK;
    int lastRowEndK;
    uint seed1Start;
    uint seed2Start;
} LaunchBatch;
// One hashA hit: the candidate's trailing index, and which batch it's in.
typedef struct {
    ulong trailingIdx;
    uint batch;
    uint unused;
} Hit;
// Everything a launch reports: how many hits it had - which may exceed
// MAX_MATCHES - and the first MAX_MATCHES of them. The host reads the count
// and the first few hits in one go.
typedef struct {
    int matchCount;
    int unused;
    Hit hits[MAX_MATCHES];
} BatchResults;

__kernel void searchRows(uint targetA,
                         __constant LaunchBatch* batches,      // the launch's batches
                         __constant uint* alphabetKey,         // crypt-table key of each alphabet character
                         __constant uint* alphabetOrd,         // each alphabet character itself
                         __constant uint* suffixKey,
                         __constant uint* suffixOrd,
                         __global const ulong* filterTable,    // this search's lookup filter
                         __global BatchResults* results) {
    // The rows' characters, and the few last characters the filter lets
    // through, differ between work-items, so they're looked up here rather
    // than in constant memory, where work-items reading different entries
    // would be serialized.
    __local uint sKey[ALPHABET_SIZE];
    __local uint sOrd[ALPHABET_SIZE];
    const uint lid = get_local_id(0);
    if (lid < ALPHABET_SIZE) {
        sKey[lid] = alphabetKey[lid];
        sOrd[lid] = alphabetOrd[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE); // before anything returns, so every work-item reaches it

    // This work-group's batch - the same for the whole work-group.
    const uint batchIndex = get_group_id(1);
    const LaunchBatch batch = batches[batchIndex];
    const uint firstRow = batch.firstRow, lastRow = batch.lastRow;
    const int firstRowStartK = batch.firstRowStartK, lastRowEndK = batch.lastRowEndK;
    const uint seed1Start = batch.seed1Start, seed2Start = batch.seed2Start;

    // Every chunk of every group the batch touches, whatever the global work
    // size: that only decides how the chunks are shared out (normally one
    // each), never which of them get searched.
    const uint firstGroup = firstRow / ALPHABET_SIZE;
    const ulong chunkCount = (ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;
    for (ulong c = get_global_id(0); c < chunkCount; c += get_global_size(0)) {
        const uint group = firstGroup + (uint) (c / CHUNKS_PER_GROUP);
        const uint chunk = (uint) (c % CHUNKS_PER_GROUP);
        // The chunk's rows, as last row characters d of `group`, cut to the
        // batch's range in its first and last group.
        const ulong groupStart = (ulong) group * ALPHABET_SIZE;
        int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP);
        int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP);
        if (groupStart + dBegin < firstRow)
            dBegin = (int) (firstRow - groupStart);
        if (groupStart + dEnd > (ulong) lastRow + 1)
            dEnd = (int) ((ulong) lastRow + 1 - groupStart);
        // The d of the batch's first and last row, if they're in this group
        // (-1 if not) - the rows firstRowStartK and lastRowEndK cut short.
        const int firstRowD = (group == firstRow / ALPHABET_SIZE) ? (int) (firstRow % ALPHABET_SIZE) : -1;
        const int lastRowD = (group == lastRow / ALPHABET_SIZE) ? (int) (lastRow % ALPHABET_SIZE) : -1;

        // The group's characters: every row character but the last, i.e. the
        // first TRAILING_LEN - 2 characters of the candidate.
        uint group1 = seed1Start, group2 = seed2Start;
#if TRAILING_LEN > 2
        uint digit[TRAILING_LEN - 2];
        uint rest = group;
#pragma unroll
        for (int i = TRAILING_LEN - 3; i >= 0; --i) {
            const uint q = rest / ALPHABET_SIZE;
            digit[i] = rest - q * ALPHABET_SIZE;
            rest = q;
        }
#pragma unroll
        for (int i = 0; i < TRAILING_LEN - 2; ++i)
            MPQ_STEP(group1, group2, sKey[digit[i]], sOrd[digit[i]]);
#endif

        for (int d = dBegin; d < dEnd; ++d) {
            uint seed1 = group1, seed2 = group2;
#if TRAILING_LEN > 1
            // The row's own last row character. (With TRAILING_LEN 1 a row has
            // no characters of its own, and there's only row 0.)
            MPQ_STEP(seed1, seed2, sKey[d], sOrd[d]);
#endif

            // Bit k: the candidate with last character k is worth hashing.
            // Restricted to the batch's range in its first and last row - and
            // to the alphabet, which the table never exceeds anyway, but a
            // stray bit must not be able to index past sKey.
            ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];
            mask &= (1UL << ALPHABET_SIZE) - 1UL;              // ALPHABET_SIZE is at most 50
            if (d == firstRowD)
                mask &= ~0UL << firstRowStartK;                // firstRowStartK is in [0, ALPHABET_SIZE)
            if (d == lastRowD)
                mask &= (1UL << lastRowEndK) - 1UL;            // lastRowEndK is in [1, ALPHABET_SIZE]

            while (mask != 0) {
                const int k = 63 - (int) clz(mask & (0UL - mask)); // the lowest bit set
                mask &= mask - 1UL;
                uint a = seed1, b = seed2;
                MPQ_STEP(a, b, sKey[k], sOrd[k]);
#pragma unroll
                for (int i = 0; i < SUFFIX_LEN; ++i)
                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);
                if (HASHA_MATCHES(a, targetA)) {
                    const int slot = atomic_inc(&results->matchCount);
                    if (slot < MAX_MATCHES) {
                        results->hits[slot].trailingIdx = (groupStart + d) * ALPHABET_SIZE + k;
                        results->hits[slot].batch = batchIndex;
                    }
                }
            }
        }
    }
}
