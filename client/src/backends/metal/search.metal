// The Metal backend's search kernel - the OpenCL one (backends/opencl/search.cl,
// itself the CUDA kernel ported) in Metal's shading language, lookup filter,
// row groups and all. Embedded into the program at build time and compiled by
// Metal at runtime, once per combination of these (see metal_backend.mm),
// which makes them all compile-time constants the compiler can unroll and
// fold:
//   ALPHABET_SIZE  the alphabet's size
//   SUFFIX_LEN     the suffix's length
//   TRAILING_LEN   the candidate's trailing (GPU-enumerated) length, >= 1
//   MAX_MATCHES    how many hits one batch can record
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                  in the stress tests' builds (see engine/hash_match.h)
//   FILTER_BITS    kLowBitsFilterBits: how many low bits of each seed index
//                  the lookup filter's table (backends/common/lowbits_filter.h)
//   ROWS_PER_THREAD  about how many rows one thread searches
//   LISTED         1 if the search prunes the whole candidate: the batch then
//                  searches the row groups of its list (see below), 0 if
//                  every group it touches
//   GROUP_INSERTn_AFTER/_START/_LEN, ROW_INSERT_START/_LEN  where the text
//                  inserted into the trailing part goes, as in search.cl
//
// Rows are every value of the candidate's last character, for one
// combination of the other trailing characters (see the terminology in
// backends/common/row_batch.h). A thread hashes the characters its rows
// share once, then one more for each row, and looks each row's state up in
// this search's filter table, which gives the set of last characters whose
// hashA has the target's low bits - always including any that matches the
// target (README.md's "The lookup filter" has why) - and hashes only those,
// in full. It records the trailing index of every hashA hit; the host
// rebuilds and checks those.
//
// With LISTED, the batch searches the row groups of its list - those the
// search's pruning leaves - and of each, the rows its entry's classes allow
// (see backends/common/row_pruning.h). Without it, every group it touches,
// every row of them - as the CUDA and OpenCL kernels do, where walking a list
// of every group measured 5-6% slower.

#include <metal_stdlib>
using namespace metal;

#ifndef HASHA_MATCH_MASK
#define HASHA_MATCH_MASK 0xFFFFFFFFu
#endif
#define HASHA_MATCHES(a, target) (((a) & (uint) (HASHA_MATCH_MASK)) == ((target) & (uint) (HASHA_MATCH_MASK)))

// lowBitsFilterIndex (backends/common/lowbits_filter.h), which this must
// match exactly: the low FILTER_BITS bits of seed1, then those of seed2.
#define FILTER_STATE_MASK ((1u << FILTER_BITS) - 1u)
#define FILTER_INDEX(seed1, seed2) (((seed1) & FILTER_STATE_MASK) | (((seed2) & FILTER_STATE_MASK) << FILTER_BITS))

// A row *group* is the ALPHABET_SIZE consecutive rows that share every row
// character but the last (see kChunksPerGroup in backends/cuda/cuda_backend.cu,
// which this follows). Each group is split into CHUNKS_PER_GROUP chunks of
// about ROWS_PER_THREAD consecutive rows, one per thread: chunk c is the
// group's rows d = c * ALPHABET_SIZE / CHUNKS_PER_GROUP .. (c + 1) *
// ALPHABET_SIZE / CHUNKS_PER_GROUP - 1, so the chunks cover every row of the
// group exactly once and never cross into another.
#define CHUNKS_PER_GROUP ((ALPHABET_SIZE + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD)

// An entry of a row groups' list: the group, shifted past its classes
// (ROW_FLAG_BITS, given by the host: kRowFlagBits in
// backends/common/row_pruning.h).
#define ROW_FLAG_COUNT (1 << ROW_FLAG_BITS)

#ifndef GROUP_INSERT0_AFTER
#define GROUP_INSERT0_AFTER 0
#define GROUP_INSERT0_START 0
#define GROUP_INSERT0_LEN 0
#define GROUP_INSERT1_AFTER 0
#define GROUP_INSERT1_START 0
#define GROUP_INSERT1_LEN 0
#define ROW_INSERT_START 0
#define ROW_INSERT_LEN 0
#endif
// The group character an insertion goes before - TRAILING_LEN - 2: after
// the last one; TRAILING_LEN, never reached: none.
#define GROUP_INSERT0_AT (TRAILING_LEN - GROUP_INSERT0_AFTER)
#define GROUP_INSERT1_AT (TRAILING_LEN - GROUP_INSERT1_AFTER)
// Hashes `len` characters of `inserted` from `start`.
#define HASH_INSERTED(seed1, seed2, start, len)                                                  \
    do {                                                                                         \
        for (int n_ = 0; n_ < (len); ++n_)                                                       \
            MPQ_STEP(seed1, seed2, inserted[2 * ((start) + n_)], inserted[2 * ((start) + n_) + 1]); \
    } while (0)

// Must match RowArgs in metal_backend.mm.
struct RowArgs {
    uint firstRow;
    uint lastRow;
    int firstRowStartK;
    int lastRowEndK;
    uint targetA;
    uint seed1Start;
    uint seed2Start;
    // With LISTED, the row groups to search: groups[groupsOffset ..
    // groupsOffset + groupCount).
    uint groupsOffset;
    uint groupCount;
};

#define MPQ_STEP(seed1, seed2, key, ord)                        \
    do {                                                        \
        seed1 = (key) ^ (seed1 + seed2);                        \
        seed2 = (ord) + seed1 + seed2 + (seed2 << 5) + 3;       \
    } while (0)

kernel void searchRows(constant RowArgs& args [[buffer(0)]],
                       constant uint* alphabetKey [[buffer(1)]],      // crypt-table key of each alphabet character
                       constant uint* alphabetOrd [[buffer(2)]],      // each alphabet character itself
                       constant uint* suffixKey [[buffer(3)]],
                       constant uint* suffixOrd [[buffer(4)]],
                       device atomic_int* matchCount [[buffer(5)]],   // hits this batch - may exceed MAX_MATCHES
                       device ulong* matchIdx [[buffer(6)]],          // the first MAX_MATCHES hits' trailing indices
                       device const ulong* filterTable [[buffer(7)]], // this search's lookup filter
                       device const uint* groups [[buffer(8)]],       // the row groups' lists (RowPruning::arena)
                       device const ulong* rowMasks [[buffer(9)]],    // the rows an entry's classes allow (RowPruning::rowMasks)
                       constant uint* inserted [[buffer(10)]],        // the inserted text's characters: key, value, key, ...
                       uint t [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_grid]]) {
    // The rows' characters, and the few last characters the filter lets
    // through, differ between threads, so they're looked up here rather than
    // in constant memory.
    threadgroup uint sKey[ALPHABET_SIZE];
    threadgroup uint sOrd[ALPHABET_SIZE];
    if (lid < ALPHABET_SIZE) {
        sKey[lid] = alphabetKey[lid];
        sOrd[lid] = alphabetOrd[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // before anything returns, so every thread reaches it

    // Every chunk of every group the batch touches - or with LISTED, of every
    // group in its list - whatever the grid's size: that only decides how the
    // chunks are shared out (normally one each), never which of them get
    // searched.
#if LISTED
    device const uint* batchGroups = groups + args.groupsOffset;
    const ulong chunkCount = (ulong) args.groupCount * CHUNKS_PER_GROUP;
#else
    const uint firstGroup = args.firstRow / ALPHABET_SIZE;
    const ulong chunkCount = (ulong) (args.lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;
#endif
    for (ulong c = t; c < chunkCount; c += threads) {
#if LISTED
        const uint entry = batchGroups[c / CHUNKS_PER_GROUP];
        const uint group = entry >> ROW_FLAG_BITS;
        // Bit d: row d of the group isn't pruned.
        // Read from device memory, as the CUDA and OpenCL kernels do: copying
        // all ROW_FLAG_COUNT into threadgroup memory first cost them about 7%.
        const ulong rowMask = rowMasks[entry & (ROW_FLAG_COUNT - 1)];
#else
        const uint group = firstGroup + (uint) (c / CHUNKS_PER_GROUP);
#endif
        const uint chunk = (uint) (c % CHUNKS_PER_GROUP);
        // The chunk's rows, as last row characters d of `group`, cut to the
        // batch's range in its first and last group.
        const ulong groupStart = (ulong) group * ALPHABET_SIZE;
        int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP);
        int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP);
        if (groupStart + dBegin < args.firstRow)
            dBegin = (int) (args.firstRow - groupStart);
        if (groupStart + dEnd > (ulong) args.lastRow + 1)
            dEnd = (int) ((ulong) args.lastRow + 1 - groupStart);
        // The d of the batch's first and last row, if they're in this group
        // (-1 if not) - the rows firstRowStartK and lastRowEndK cut short.
        const int firstRowD = (group == args.firstRow / ALPHABET_SIZE) ? (int) (args.firstRow % ALPHABET_SIZE) : -1;
        const int lastRowD = (group == args.lastRow / ALPHABET_SIZE) ? (int) (args.lastRow % ALPHABET_SIZE) : -1;

        // The group's characters: every row character but the last, i.e. the
        // first TRAILING_LEN - 2 characters of the candidate.
        uint group1 = args.seed1Start, group2 = args.seed2Start;
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
        for (int i = 0; i < TRAILING_LEN - 2; ++i) {
            if (i == GROUP_INSERT0_AT)
                HASH_INSERTED(group1, group2, GROUP_INSERT0_START, GROUP_INSERT0_LEN);
            if (i == GROUP_INSERT1_AT)
                HASH_INSERTED(group1, group2, GROUP_INSERT1_START, GROUP_INSERT1_LEN);
            MPQ_STEP(group1, group2, sKey[digit[i]], sOrd[digit[i]]);
        }
        if (TRAILING_LEN - 2 == GROUP_INSERT0_AT)
            HASH_INSERTED(group1, group2, GROUP_INSERT0_START, GROUP_INSERT0_LEN);
        if (TRAILING_LEN - 2 == GROUP_INSERT1_AT)
            HASH_INSERTED(group1, group2, GROUP_INSERT1_START, GROUP_INSERT1_LEN);
#endif

#if LISTED
        // Bit i: the chunk's i-th row isn't pruned - shifted on one row at a
        // time, which costs less than testing bit d of rowMask every row.
        ulong rowBits = rowMask >> dBegin;
#endif
        for (int d = dBegin; d < dEnd; ++d) {
            uint seed1 = group1, seed2 = group2;
#if TRAILING_LEN > 1
            // The row's own last row character. (With TRAILING_LEN 1 a row has
            // no characters of its own, and there's only row 0.)
            MPQ_STEP(seed1, seed2, sKey[d], sOrd[d]);
            HASH_INSERTED(seed1, seed2, ROW_INSERT_START, ROW_INSERT_LEN);
#endif

            // Bit k: the candidate with last character k is worth hashing.
            // Restricted to the batch's range in its first and last row - and
            // to the alphabet, which the table never exceeds anyway, but a
            // stray bit must not be able to index past sKey. None, in a row the
            // pruning leaves out: the threads of a SIMD-group step through
            // their rows together, so skipping one would save its thread
            // nothing (as in the CUDA kernel, whose comment has more).
#if LISTED
            ulong mask = (rowBits & 1ul) ? filterTable[FILTER_INDEX(seed1, seed2)] : 0ul;
            rowBits >>= 1;
#else
            ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];
#endif
            mask &= (1ul << ALPHABET_SIZE) - 1ul;               // ALPHABET_SIZE is at most 63
            if (d == firstRowD)
                mask &= ~0ul << args.firstRowStartK;            // firstRowStartK is in [0, ALPHABET_SIZE)
            if (d == lastRowD)
                mask &= (1ul << args.lastRowEndK) - 1ul;        // lastRowEndK is in [1, ALPHABET_SIZE]

            while (mask != 0) {
                // The lowest bit set, from 32-bit halves.
                const uint low = uint(mask), high = uint(mask >> 32);
                const int k = (low != 0) ? int(ctz(low)) : 32 + int(ctz(high));
                mask &= mask - 1ul;
                uint a = seed1, b = seed2;
                MPQ_STEP(a, b, sKey[k], sOrd[k]);
#pragma unroll
                for (int i = 0; i < SUFFIX_LEN; ++i)
                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);
                if (HASHA_MATCHES(a, args.targetA)) {
                    const int slot = atomic_fetch_add_explicit(matchCount, 1, memory_order_relaxed);
                    if (slot < MAX_MATCHES)
                        matchIdx[slot] = (groupStart + d) * ALPHABET_SIZE + k;
                }
            }
        }
    }
}
