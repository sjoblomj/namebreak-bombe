// The Metal backend's search kernel - the same algorithm as the OpenCL one
// (backends/opencl/search.cl, itself the CUDA kernel ported). Embedded into
// the program at build time and compiled by Metal at runtime, once per
// combination of these (see metal_backend.mm), which makes them all
// compile-time constants the compiler can unroll and fold:
//   ALPHABET_SIZE  the alphabet's size
//   SUFFIX_LEN     the suffix's length
//   TRAILING_LEN   the candidate's trailing (GPU-enumerated) length, >= 1
//   MAX_MATCHES    how many hits one batch can record
//
// One thread per *row* - every value of the candidate's last character, for
// one combination of the other trailing characters (see the terminology in
// backends/common/row_batch.h). It hashes the row's shared characters once,
// then only the last character and the suffix per candidate, and records the
// trailing index of every hashA hit; the host rebuilds and checks those.

#include <metal_stdlib>
using namespace metal;

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
                       constant uint* alphabetKey [[buffer(1)]], // crypt-table key of each alphabet character
                       constant uint* alphabetOrd [[buffer(2)]], // each alphabet character itself
                       constant uint* suffixKey [[buffer(3)]],
                       constant uint* suffixOrd [[buffer(4)]],
                       device atomic_int* matchCount [[buffer(5)]], // hits this batch - may exceed MAX_MATCHES
                       device ulong* matchIdx [[buffer(6)]],        // the first MAX_MATCHES hits' trailing indices
                       uint t [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]]) {
    // The row's shared characters differ between threads, so they're looked
    // up here rather than in constant memory.
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
    const int kBegin = (t == 0) ? args.firstRowStartK : 0;
    const int kEnd = (t == args.rowCount - 1) ? args.lastRowEndK : ALPHABET_SIZE;

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

    if (kBegin == 0 && kEnd == ALPHABET_SIZE) {
#pragma unroll
        for (int k = 0; k < ALPHABET_SIZE; ++k) {
            uint a = seed1, b = seed2;
            MPQ_STEP(a, b, alphabetKey[k], alphabetOrd[k]);
#pragma unroll
            for (int i = 0; i < SUFFIX_LEN; ++i)
                MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);
            if (a == args.targetA) {
                const int slot = atomic_fetch_add_explicit(matchCount, 1, memory_order_relaxed);
                if (slot < MAX_MATCHES)
                    matchIdx[slot] = (ulong) row * ALPHABET_SIZE + k;
            }
        }
    } else {
        // A partial row at the edge of the batch's range - only ever the
        // first and/or last thread, so only correctness matters here.
        for (int k = kBegin; k < kEnd; ++k) {
            uint a = seed1, b = seed2;
            MPQ_STEP(a, b, sKey[k], sOrd[k]);
            for (int i = 0; i < SUFFIX_LEN; ++i)
                MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);
            if (a == args.targetA) {
                const int slot = atomic_fetch_add_explicit(matchCount, 1, memory_order_relaxed);
                if (slot < MAX_MATCHES)
                    matchIdx[slot] = (ulong) row * ALPHABET_SIZE + k;
            }
        }
    }
}
