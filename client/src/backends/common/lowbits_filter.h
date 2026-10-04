#ifndef NAMEBREAK_BACKENDS_COMMON_LOWBITS_FILTER_H
#define NAMEBREAK_BACKENDS_COMMON_LOWBITS_FILTER_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/backend.h"
#include "engine/hash_match.h"
#include "engine/limits.h"

// The lookup filter a row-based backend can use to hash only the few
// candidates of a row that could possibly match, instead of all of them (see
// the terminology in row_batch.h for rows, and README.md's "The lookup
// filter" for the whole argument).
//
// Every step of the MPQ hash,
//     seed1 = key[ch] ^ (seed1 + seed2)
//     seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3
// is built from +, ^, << and constants, and the crypt table is indexed by the
// character, never by the state. None of those operations lets a bit of the
// state affect any *lower* bit of the result, so the low kLowBitsFilterBits
// bits of a candidate's hashA depend only on the low kLowBitsFilterBits bits
// of its row's state (seed1, seed2) - plus the last character and the suffix,
// which are the same for every row of a search. The table maps those
// 2 * kLowBitsFilterBits state bits to the set of last characters whose hashA
// has the target's low bits. A candidate that matches the target has all of
// its bits, so in particular those: it is always in the set. The filter can
// only skip candidates that cannot match.

// How many low bits of each of seed1 and seed2 index the table: 2^(2n)
// entries of 8 bytes (n = 7: 128 KB). About alphabetSize / 2^n of a row's
// candidates get past it. 7 is measured, not guessed: on the RTX 3080 Ti
// Laptop (search_bench --scale 20, interleaved runs) 6 bits did about 1,770
// G candidates/s, 7 about 1,800, 8 about 1,550, 9 about 1,580 and 10 about
// 390 - a smaller table caches better, and that outweighs hashing twice as
// many candidates in full. The OpenCL, Metal and CPU backends use this
// width; the CUDA backend has its own (kCudaLowBitsFilterBits, below).
// Overridable at compile time (-DNAMEBREAK_LOWBITS_FILTER_BITS=N) so the
// tests can run with a deliberately weak filter that lets most candidates
// through (see CMakeLists.txt) - which then sets the CUDA backend's too.
//
// The CUDA (and HIP) backend's width: 6 bits, a 32 KB table. Its kernel
// hashes a row's flagged candidates after all of the chunk's rows rather
// than after each, so each candidate costs it less than it did, and the
// smaller table's cheaper reads count for more: measured 13% faster than 7
// bits with a 4-character suffix, and the same with an 11-character one -
// where OpenCL and the CPU backend measured 3-17% slower at 6 bits.
// Overridable on its own with -DNAMEBREAK_CUDA_LOWBITS_FILTER_BITS=N.
#ifndef NAMEBREAK_CUDA_LOWBITS_FILTER_BITS
#ifdef NAMEBREAK_LOWBITS_FILTER_BITS
#define NAMEBREAK_CUDA_LOWBITS_FILTER_BITS NAMEBREAK_LOWBITS_FILTER_BITS
#else
#define NAMEBREAK_CUDA_LOWBITS_FILTER_BITS 6
#endif
#endif
#ifndef NAMEBREAK_LOWBITS_FILTER_BITS
#define NAMEBREAK_LOWBITS_FILTER_BITS 7
#endif
constexpr int kLowBitsFilterBits = NAMEBREAK_LOWBITS_FILTER_BITS;
constexpr int kCudaLowBitsFilterBits = NAMEBREAK_CUDA_LOWBITS_FILTER_BITS;
constexpr int kMaxLowBitsFilterBits = 10;
static_assert(kLowBitsFilterBits >= 0 && kLowBitsFilterBits <= kMaxLowBitsFilterBits, "NAMEBREAK_LOWBITS_FILTER_BITS must be 0-10 (a 2^20-entry, 8 MB table)");
static_assert(kCudaLowBitsFilterBits >= 0 && kCudaLowBitsFilterBits <= kMaxLowBitsFilterBits, "NAMEBREAK_CUDA_LOWBITS_FILTER_BITS must be 0-10");
static_assert(MAX_ALPHABET_SIZE <= 64, "a row's candidates must fit in one 64-bit table entry");

// Everything below takes the table's width, `bits`, defaulting to
// kLowBitsFilterBits.
NAMEBREAK_HOST_DEVICE constexpr uint32_t lowBitsFilterStateMask(int bits = kLowBitsFilterBits) { return (1u << bits) - 1; }
NAMEBREAK_HOST_DEVICE constexpr uint32_t lowBitsFilterEntries(int bits = kLowBitsFilterBits) { return 1u << (2 * bits); }
// The bits of hashA an entry is computed from: the low `bits`, but never more
// than a hit needs to match (fewer only in the stress tests' builds, see
// hash_match.h).
NAMEBREAK_HOST_DEVICE constexpr uint32_t lowBitsFilterHashMask(int bits = kLowBitsFilterBits) {
    return lowBitsFilterStateMask(bits) & kHashAMatchMask;
}
constexpr uint32_t kLowBitsFilterStateMask = lowBitsFilterStateMask();
constexpr uint32_t kLowBitsFilterEntries = lowBitsFilterEntries();
constexpr uint32_t kLowBitsFilterHashMask = lowBitsFilterHashMask();

// The table entry for a row whose shared characters leave the hash state at
// (seed1, seed2). The high bits of both are ignored - that is the point.
NAMEBREAK_HOST_DEVICE inline uint32_t lowBitsFilterIndex(uint32_t seed1, uint32_t seed2, int bits = kLowBitsFilterBits) {
    return (seed1 & lowBitsFilterStateMask(bits)) | ((seed2 & lowBitsFilterStateMask(bits)) << bits);
}

// The table for one search: lowBitsFilterEntries(bits) entries, where bit k
// of entry lowBitsFilterIndex(seed1, seed2, bits) is set exactly when the
// candidate that continues from (seed1, seed2) with alphabet character k and
// then the suffix has a hashA whose lowBitsFilterHashMask(bits) bits equal
// the target's. Bits at or past the alphabet's size are never set.
std::vector<uint64_t> buildLowBitsFilterTable(const SearchConstants& constants, int bits = kLowBitsFilterBits);

// Checks `table` against the definition above, independently of how
// buildLowBitsFilterTable computed it: for each checked entry, it picks a
// full 32-bit state with *random* high bits that lowBitsFilterIndex maps to
// that entry, hashes every last character followed by the suffix from it, and
// compares every bit of the entry with the result - so it checks the table's
// layout, the index function and the property the filter rests on, not only
// the builder. Checks `entriesToCheck` randomly chosen entries (every entry,
// if 0), each with `highBitRounds` different random high bits. False, with
// `error` describing the first mismatch, if any bit is wrong.
bool checkLowBitsFilterTable(const std::vector<uint64_t>& table, const SearchConstants& constants, uint32_t entriesToCheck,
                             int highBitRounds, uint64_t seed, std::string& error, int bits = kLowBitsFilterBits);

#endif // NAMEBREAK_BACKENDS_COMMON_LOWBITS_FILTER_H
