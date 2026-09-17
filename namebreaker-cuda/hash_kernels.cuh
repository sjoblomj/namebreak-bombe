#ifndef NAMEBREAK_CUDA_HASH_KERNELS_CUH
#define NAMEBREAK_CUDA_HASH_KERNELS_CUH

#include <cstdint>
#include <cstring>
#include "constants.h"

// Device-side hashing/candidate-decoding building blocks shared between
// namebreak.cu's real search kernel and this project's GPU-backed tests
// (tests/search_bench.cu, tests/search_integration_test.cu, ...) - kept in
// one header so the code under test is always exactly the code that ships,
// never a copy that could drift out of sync.

// Capacity of both d_prefix and BatchParams::prefix below (including the
// terminating NUL) - what runSearch checks the extended prefix against.
constexpr int kMaxPrefixSize = 64;

// Everything that changes from one leading value to the next (i.e. once per
// kernel launch in the common case), passed to bruteForceKernel by value as a
// kernel argument instead of being uploaded to the __constant__ symbols below
// with cudaMemcpyToSymbol. Those uploads were synchronous driver calls that
// each cost ~6us of GPU idle time between two consecutive kernels - a kernel
// argument rides along with the launch itself for free.
struct BatchParams {
    char prefix[kMaxPrefixSize]; // req.prefix + the leading characters, NUL-terminated
    short prefixSize;
    uint32_t seed1Start;         // hash state after the (extended) prefix - see
    uint32_t seed2Start;         // mpqHashWithPrefixCache_CPU / IncrementalPrefixHasher
};

__device__ __constant__ char d_alphabet[MAX_ALPHABET_SIZE + 1];
// d_prefix/d_prefix_size/d_seed*_start (below) are no longer what the search
// kernel reads - it takes a BatchParams instead. They remain for the
// symbol-based overloads of mpqHashCandidateAndSuffix/buildCompleteFilename,
// which tests/window_sweep_bench.cu still uses.
__device__ __constant__ char d_prefix[kMaxPrefixSize];
__device__ __constant__ char d_suffix[64];
__device__ __constant__ short d_prefix_size;
__device__ __constant__ short d_suffix_size;
__device__ __constant__ uint32_t d_seed1_start;
__device__ __constant__ uint32_t d_seed2_start;

__device__ __constant__ uint32_t d_cryptTable[0x500];

// Hashes `candidate` followed by d_suffix directly, without ever concatenating them
// into a scratch buffer first. Starts from seed1/seed2, which already
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
__device__ uint32_t mpqHashCandidateAndSuffix(const char* candidate, int candidateLen, uint32_t seed1, uint32_t seed2) {
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

// Same, starting from the d_seed1_start/d_seed2_start symbols.
__device__ uint32_t mpqHashCandidateAndSuffix(const char* candidate, int candidateLen) {
    return mpqHashCandidateAndSuffix(candidate, candidateLen, d_seed1_start, d_seed2_start);
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

// Same recurrence as mpqHashSeed2 (offset 0x200), but offset 0x100 - i.e. a
// from-scratch hashA over the complete filename, independent of the
// prefix-cache incremental path mpqHashCandidateAndSuffix uses. Only called
// once, on the rare candidate that already matched via that incremental
// path, as a correctness cross-check (see its call site in
// bruteForceKernel) - not on the hot per-candidate path, so recomputing from
// scratch here costs nothing that matters.
__device__ uint32_t mpqHashSeed1(const char* str) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    unsigned char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

// AlphabetSize is a compile-time template parameter so this modulus/division -
// run once per candidate character, for every thread - stays a cheap
// compiler-optimized constant instead of a real (much slower) GPU integer
// division. See runCudaBatch in namebreak.cu for the fixed set of sizes this
// gets instantiated for and the runtime dispatch between them.
template<int AlphabetSize>
__device__ void indexToCandidate(uint64_t index, int candidateLen, char* outCandidate) {
    for (int i = candidateLen - 1; i >= 0; --i) {
        outCandidate[i] = d_alphabet[index % AlphabetSize];
        index /= AlphabetSize;
    }
}

// No hasForbiddenSymbolRun/countBackslashes here (deliberately - see
// bruteForceKernel's doc comment in namebreak.cu and
// hasForbiddenSymbolRun_CPU's in cpu-utils.h): those checks only ever run on
// the CPU now, against the leading characters, before this candidate's batch
// is even launched.

__device__ void buildCompleteFilename(const char* prefix, short prefixSize, const char* candidate, int candidateLen, char* out) {
    memcpy(out, prefix, prefixSize);
    short i = prefixSize;

    memcpy(out + i, candidate, candidateLen);
    i += candidateLen;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

// Same, using the d_prefix/d_prefix_size symbols.
__device__ void buildCompleteFilename(const char* candidate, int candidateLen, char* out) {
    buildCompleteFilename(d_prefix, d_prefix_size, candidate, candidateLen, out);
}

#endif //NAMEBREAK_CUDA_HASH_KERNELS_CUH
