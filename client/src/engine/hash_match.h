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

// The same for a dictionary search's basename hash (hash type 3, what an
// encryption key is made with): all 32 bits in the program, fewer in the
// dictionary search test's builds (-DNAMEBREAK_BASENAME_MATCH_BITS=8), so
// that a candidate's basename matches the key by chance often enough to
// test every path a basename hit takes. The low bits again: the suffix
// filters rely on that.
#ifndef NAMEBREAK_BASENAME_MATCH_BITS
#define NAMEBREAK_BASENAME_MATCH_BITS 32
#endif
static_assert(NAMEBREAK_BASENAME_MATCH_BITS >= 1 && NAMEBREAK_BASENAME_MATCH_BITS <= 32, "NAMEBREAK_BASENAME_MATCH_BITS must be 1-32");
constexpr uint32_t kBasenameMatchMask = 0xFFFFFFFFu >> (32 - NAMEBREAK_BASENAME_MATCH_BITS);

NAMEBREAK_HOST_DEVICE inline bool basenameKeyMatches(uint32_t hash, uint32_t key) {
    return (hash & kBasenameMatchMask) == (key & kBasenameMatchMask);
}

// And the same for a junction search (engine/junction.h), where what must
// match isn't a hash but the whole hash state after the unknown directory:
// how many of the low bits *of each of seed1 and seed2* must equal the
// junction's. All 32 of each - the state entire - in the program. The
// junction search tests build with fewer
// (-DNAMEBREAK_JUNCTION_MATCH_BITS=3, so six bits in all), so that
// directories match the junction by chance all the time, and every path a
// match takes - looking its directories back up in the table, checking them
// in full, reporting them - runs thousands of times rather than once in a
// search that plants one.
//
// The low bits of both seeds, and the same number of each: no step of the
// hash lets a bit of the state affect any *lower* bit of the result (see
// backends/common/lowbits_filter.h for the argument, which holds for the
// step undone as well - it multiplies, subtracts and XORs, and a shift only
// moves bits up). So two states whose seeds agree in their low n bits still
// agree there after the same character is hashed onto them, or rewound off
// them, and a junction search's two halves meet on those bits exactly as
// they meet on the whole state. Taking the low bits of seed1 alone, or more
// of one seed than the other, would not carry through a step like that: the
// table and the walk would then be comparing different things.
#ifndef NAMEBREAK_JUNCTION_MATCH_BITS
#define NAMEBREAK_JUNCTION_MATCH_BITS 32
#endif
static_assert(NAMEBREAK_JUNCTION_MATCH_BITS >= 1 && NAMEBREAK_JUNCTION_MATCH_BITS <= 32, "NAMEBREAK_JUNCTION_MATCH_BITS must be 1-32");
constexpr uint32_t kJunctionSeedMask = 0xFFFFFFFFu >> (32 - NAMEBREAK_JUNCTION_MATCH_BITS);

// The two seeds of a hash state as one key, each masked to the bits that
// must match. Two states with the same key are a junction search's "same
// place".
NAMEBREAK_HOST_DEVICE inline uint64_t junctionKey(uint32_t seed1, uint32_t seed2) {
    return (((uint64_t) (seed2 & kJunctionSeedMask)) << 32) | (seed1 & kJunctionSeedMask);
}

#endif // NAMEBREAK_ENGINE_HASH_MATCH_H
