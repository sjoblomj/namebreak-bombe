// The Metal backend's search kernel - the OpenCL one (backends/opencl/search.cl,
// itself the CUDA kernel ported) in Metal's shading language, lookup filter
// and all, with one thread per row. Embedded into the program at build time
// and compiled by Metal at runtime, once per combination of these (see
// metal_backend.mm), which makes them all compile-time constants the compiler
// can unroll and fold:
//   ALPHABET_SIZE  the alphabet's size
//   SUFFIX_LEN     the suffix's length
//   TRAILING_LEN   the candidate's trailing (GPU-enumerated) length, >= 1
//   MAX_MATCHES    how many hits one batch can record
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                  in the stress tests' builds (see engine/hash_match.h)
//   FILTER_BITS    kLowBitsFilterBits: how many low bits of each seed index
//                  the lookup filter's table (backends/common/lowbits_filter.h)
//
// One thread per *row* - every value of the candidate's last character, for
// one combination of the other trailing characters (see the terminology in
// backends/common/row_batch.h). It hashes the row's shared characters once,
// then looks the row's state up in this search's filter table, which gives
// the set of last characters whose hashA has the target's low bits - always
// including any that matches the target (README.md's "The lookup filter" has
// why) - and hashes only those, in full. It records the trailing index of
// every hashA hit; the host rebuilds and checks those.

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

// Must match RowArgs in metal_backend.mm.
struct RowArgs {
    uint firstRow;
    uint rowCount;
    int firstRowStartK;
    int lastRowEndK;
    uint targetA;
    uint seed1Start;
    uint seed2Start;
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
                       uint t [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]]) {
    // The row's characters, and the few last characters the filter lets
    // through, differ between threads, so they're looked up here rather than
    // in constant memory.
    threadgroup uint sKey[ALPHABET_SIZE];
    threadgroup uint sOrd[ALPHABET_SIZE];
    if (lid < ALPHABET_SIZE) {
        sKey[lid] = alphabetKey[lid];
        sOrd[lid] = alphabetOrd[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // before the bounds check, so every thread reaches it

    if (t >= args.rowCount)
        return;
    const uint row = args.firstRow + t;

    uint seed1 = args.seed1Start, seed2 = args.seed2Start;
#if TRAILING_LEN > 1
    uint digit[TRAILING_LEN - 1];
    uint rest = row;
#pragma unroll
    for (int i = TRAILING_LEN - 2; i >= 0; --i) {
        const uint q = rest / ALPHABET_SIZE;
        digit[i] = rest - q * ALPHABET_SIZE;
        rest = q;
    }
#pragma unroll
    for (int i = 0; i < TRAILING_LEN - 1; ++i)
        MPQ_STEP(seed1, seed2, sKey[digit[i]], sOrd[digit[i]]);
#endif

    // Bit k: the candidate with last character k is worth hashing. Restricted
    // to the batch's range in its first and last row - and to the alphabet,
    // which the table never exceeds anyway, but a stray bit must not be able
    // to index past sKey.
    ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];
    mask &= (1ul << ALPHABET_SIZE) - 1ul;                   // ALPHABET_SIZE is at most 50
    if (t == 0)
        mask &= ~0ul << args.firstRowStartK;                // firstRowStartK is in [0, ALPHABET_SIZE)
    if (t == args.rowCount - 1)
        mask &= (1ul << args.lastRowEndK) - 1ul;            // lastRowEndK is in [1, ALPHABET_SIZE]

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
                matchIdx[slot] = (ulong) row * ALPHABET_SIZE + k;
        }
    }
}
