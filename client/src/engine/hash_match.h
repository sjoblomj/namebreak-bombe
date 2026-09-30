#ifndef NAMEBREAK_ENGINE_HASH_MATCH_H
#define NAMEBREAK_ENGINE_HASH_MATCH_H

#include <cstdint>

// Lets a small function be called both from host code and from the CUDA/HIP
// kernels, so the two can't drift apart.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define NAMEBREAK_HOST_DEVICE __host__ __device__
#else
#define NAMEBREAK_HOST_DEVICE
#endif

// How many of hashA's bits must equal the target's for a candidate to be a
// hashA hit. Always all 32 in the program. The dense-hit stress test
// (tests/search_stress_test.cpp) builds the search with fewer
// (-DNAMEBREAK_HASHA_MATCH_BITS=12, see CMakeLists.txt), so that about one
// candidate in 4096 is a "hit" - every path that finds and reports hits then
// runs hundreds of thousands of times per test, rather than the handful of
// times a real 32-bit target allows. Always the *low* bits: the lookup filter
// (backends/common/lowbits_filter.h) relies on that.
#ifndef NAMEBREAK_HASHA_MATCH_BITS
#define NAMEBREAK_HASHA_MATCH_BITS 32
#endif
static_assert(NAMEBREAK_HASHA_MATCH_BITS >= 1 && NAMEBREAK_HASHA_MATCH_BITS <= 32, "NAMEBREAK_HASHA_MATCH_BITS must be 1-32");
constexpr uint32_t kHashAMatchMask = 0xFFFFFFFFu >> (32 - NAMEBREAK_HASHA_MATCH_BITS);

// Whether a candidate whose hashA is `hashA` is a hit for `target`. hashB is
// always compared in full.
NAMEBREAK_HOST_DEVICE inline bool hashAMatches(uint32_t hashA, uint32_t target) {
    return (hashA & kHashAMatchMask) == (target & kHashAMatchMask);
}

#endif // NAMEBREAK_ENGINE_HASH_MATCH_H
