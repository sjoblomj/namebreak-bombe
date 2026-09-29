// The OpenCL backend's search kernel - the CUDA backend's kernel as it was
// before it got the lookup filter (backends/common/lowbits_filter.h), ported:
// every candidate of a row is hashed. Embedded into the program at
// build time and compiled by the OpenCL driver at runtime, once per
// combination of these (see opencl_backend.cpp), which makes them all
// compile-time constants the compiler can unroll and fold:
//   ALPHABET_SIZE  the alphabet's size
//   SUFFIX_LEN     the suffix's length
//   TRAILING_LEN   the candidate's trailing (GPU-enumerated) length, >= 1
//   MAX_MATCHES    how many hits one batch can record
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                  in the stress tests' builds (see engine/hash_match.h)
//
// One work-item per *row* - every value of the candidate's last character,
// for one combination of the other trailing characters (see the terminology
// in backends/common/row_batch.h). It hashes the row's shared characters
// once, then only the last character and the suffix per candidate, and
// records the trailing index of every hashA hit; the host rebuilds and
// checks those.

#ifndef HASHA_MATCH_MASK
#define HASHA_MATCH_MASK 0xFFFFFFFFu
#endif
#define HASHA_MATCHES(a, target) (((a) & HASHA_MATCH_MASK) == ((target) & HASHA_MATCH_MASK))

#define MPQ_STEP(seed1, seed2, key, ord)                        \
    do {                                                        \
        seed1 = (key) ^ (seed1 + seed2);                        \
        seed2 = (ord) + seed1 + seed2 + (seed2 << 5) + 3;       \
    } while (0)

__kernel void searchRows(uint firstRow, uint rowCount, int firstRowStartK, int lastRowEndK, uint targetA,
                         uint seed1Start, uint seed2Start,
                         __constant uint* alphabetKey,  // crypt-table key of each alphabet character
                         __constant uint* alphabetOrd,  // each alphabet character itself
                         __constant uint* suffixKey,
                         __constant uint* suffixOrd,
                         __global int* matchCount,      // hits this batch - may exceed MAX_MATCHES
                         __global ulong* matchIdx) {    // the first MAX_MATCHES hits' trailing indices
    // The row's shared characters differ between work-items, so they're
    // looked up here rather than in constant memory, where work-items
    // reading different entries would be serialized.
    __local uint sKey[ALPHABET_SIZE];
    __local uint sOrd[ALPHABET_SIZE];
    const uint lid = get_local_id(0);
    if (lid < ALPHABET_SIZE) {
        sKey[lid] = alphabetKey[lid];
        sOrd[lid] = alphabetOrd[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE); // before the bounds check, so every work-item reaches it

    const uint t = get_global_id(0);
    if (t >= rowCount)
        return;
    const uint row = firstRow + t;
    const int kBegin = (t == 0) ? firstRowStartK : 0;
    const int kEnd = (t == rowCount - 1) ? lastRowEndK : ALPHABET_SIZE;

    uint seed1 = seed1Start, seed2 = seed2Start;
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
        // Every index here is a compile-time constant once unrolled, so the
        // keys become constant-memory operands of the instructions themselves.
#pragma unroll
        for (int k = 0; k < ALPHABET_SIZE; ++k) {
            uint a = seed1, b = seed2;
            MPQ_STEP(a, b, alphabetKey[k], alphabetOrd[k]);
#pragma unroll
            for (int i = 0; i < SUFFIX_LEN; ++i)
                MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);
            if (HASHA_MATCHES(a, targetA)) {
                const int slot = atomic_inc(matchCount);
                if (slot < MAX_MATCHES)
                    matchIdx[slot] = (ulong) row * ALPHABET_SIZE + k;
            }
        }
    } else {
        // A partial row at the edge of the batch's range - only ever the
        // first and/or last work-item, so only correctness matters here.
        for (int k = kBegin; k < kEnd; ++k) {
            uint a = seed1, b = seed2;
            MPQ_STEP(a, b, sKey[k], sOrd[k]);
            for (int i = 0; i < SUFFIX_LEN; ++i)
                MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);
            if (HASHA_MATCHES(a, targetA)) {
                const int slot = atomic_inc(matchCount);
                if (slot < MAX_MATCHES)
                    matchIdx[slot] = (ulong) row * ALPHABET_SIZE + k;
            }
        }
    }
}
