#ifndef NAMEBREAK_CUDA_HASH_KERNELS_CUH
#define NAMEBREAK_CUDA_HASH_KERNELS_CUH

#include <cstdint>
#include <cstring>
#include "constants.h"

// Device-side hashing/candidate-decoding building blocks shared between
// namebreak.cu's real search kernel and tests/warp_hash_test.cu's correctness
// harness - kept in one header so the code under test is always exactly the
// code that ships, never a copy that could drift out of sync.

__device__ __constant__ char d_alphabet[MAX_ALPHABET_SIZE + 1];
__device__ __constant__ char d_prefix[64];
__device__ __constant__ char d_suffix[64];
__device__ __constant__ short d_prefix_size;
__device__ __constant__ short d_suffix_size;
__device__ __constant__ uint32_t d_seed1_start;
__device__ __constant__ uint32_t d_seed2_start;
// Max '\' occurrences allowed in a candidate before it's discarded unhashed;
// 0 means unlimited (no candidate is ever discarded on this basis - use an
// alphabet without '\' in it if none should ever appear at all). A plain
// runtime constant rather than a template parameter like AlphabetSize: this is
// just an integer compare, not a division, so there's no compile-time-constant
// codegen benefit to chase here.
__device__ __constant__ int d_maxBackslashCount;

__device__ __constant__ uint32_t d_cryptTable[0x500];

// Hashes `candidate` followed by d_suffix directly, without ever concatenating them
// into a scratch buffer first. Starts from d_seed1_start/d_seed2_start, which already
// account for the (extended) prefix's contribution - see mpqHashWithPrefixCache_CPU.
// This is the hot path (every thread runs it), so avoiding the extra buffer write+read
// that buildFilenameWithoutPrefix + a buffer-based hash would need is worth it; the
// full filename is only built (via buildCompleteFilename) on the rare hashA match below.
//
// Tempting-looking optimization that turned out not to be one: since
// indexToCandidate makes the *last* candidate character the fastest-changing
// digit, the 32 consecutive indices one GPU warp processes usually share
// every character but the last couple, which looks like a lot of redundant
// per-thread hashing to dedupe. It isn't redundant in practice - a warp's 32
// lanes already execute this loop in lockstep, one instruction per cycle
// across all lanes, so the "shared" work already costs the same regardless of
// whether the lanes' data happens to match. An earlier version of this
// function tried to have one lane hash the shared prefix and broadcast it via
// __shfl_sync; it was verified correct (bit-for-bit, exhaustively) but
// measured 5-15% *slower* end to end, because that lane's loop still costs
// the warp the same number of cycles it always did (masking off the other 31
// lanes doesn't make a loop finish faster), and the broadcast/uniformity-
// check machinery is pure overhead on top of that.
__device__ uint32_t mpqHashCandidateAndSuffix(const char* candidate, int candidateLen) {
    uint32_t seed1 = d_seed1_start;
    uint32_t seed2 = d_seed2_start;

    // unsigned so a byte >= 0x80 zero-extends into the crypt-table index/seed
    // arithmetic instead of sign-extending to a negative value - must match
    // cpu-utils.cpp's host-side hash exactly, or a match found on one side
    // would never reproduce on the other.
    for (int i = 0; i < candidateLen; ++i) {
        unsigned char ch = candidate[i];
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    for (int i = 0; i < d_suffix_size; ++i) {
        unsigned char ch = d_suffix[i];
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

__device__ uint32_t mpqHashSeed2(const char* str) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    // unsigned - see mpqHashCandidateAndSuffix above.
    unsigned char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x200 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

// AlphabetSize is a compile-time template parameter (mirroring PruneSymbolRuns
// in namebreak.cu) so this modulus/division - run once per candidate character, for every
// thread - stays a cheap compiler-optimized constant instead of a real (much
// slower) GPU integer division. See runCudaBatch in namebreak.cu for the fixed set of
// sizes this gets instantiated for and the runtime dispatch between them.
template<int AlphabetSize>
__device__ void indexToCandidate(uint64_t index, int candidateLen, char* outCandidate) {
    for (int i = candidateLen - 1; i >= 0; --i) {
        outCandidate[i] = d_alphabet[index % AlphabetSize];
        index /= AlphabetSize;
    }
}

__device__ __forceinline__ bool isAlnumMpq(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z');
}

// Real MPQ filename components essentially never contain three consecutive
// non-alphanumeric, non-space characters (e.g. "']&_") - used to prune obviously-
// implausible candidates before spending a hash chain on them. Spaces are exempted
// since " - " and " & " are common real word separators (e.g. "Arathi - Lake",
// "Gold Separates East & West") that would otherwise be wrongly pruned. Only
// inspects the candidate itself, not where it joins the (fixed, user-supplied)
// prefix/suffix.
__device__ __forceinline__ bool hasForbiddenSymbolRun(const char* candidate, int candidateLen) {
    int run = 0;
    for (int i = 0; i < candidateLen; ++i) {
        if (isAlnumMpq(candidate[i]) || candidate[i] == ' ') {
            run = 0;
        } else if (++run >= 3) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ int countBackslashes(const char* candidate, int candidateLen) {
    int count = 0;
    for (int i = 0; i < candidateLen; ++i) {
        if (candidate[i] == '\\') count++;
    }
    return count;
}

__device__ void buildCompleteFilename(const char* candidate, int candidateLen, char* out) {
    memcpy(out, d_prefix, d_prefix_size);
    short i = d_prefix_size;

    memcpy(out + i, candidate, candidateLen);
    i += candidateLen;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

#endif //NAMEBREAK_CUDA_HASH_KERNELS_CUH
