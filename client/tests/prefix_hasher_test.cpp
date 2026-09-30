// Correctness test for IncrementalPrefixHasher (engine/mpq_hash.h). Pure host
// C++, no CUDA/GPU involved - this class only ever runs on the CPU.
//
// The thing that could go wrong here is exactly the kind of bug that would
// silently drop candidates from a real search: an off-by-one in figuring out
// which digit position a carry actually reached, or in how much of the
// hash-state stack gets popped/rebuilt, would make advance() land on the
// wrong (seed1, seed2) for some leading values - meaning namebreak would
// search with a wrong starting hash state for whatever GPU batches follow,
// and simply never find a candidate that happens to live there. No crash,
// no error - just a gap in the search space. So this doesn't spot-check a
// few values: for a spread of alphabet sizes and leadingLens, it walks
// advance() through long contiguous runs (and specifically through the
// digit-carry boundaries, including cascading multi-digit carries) and
// requires it to agree, at every single step, with recomputing
// mpqHashWithPrefixCache_CPU(prefix + indexToString(idx, leadingLen, alphabet))
// completely from scratch.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "engine/candidate.h"
#include "engine/mpq_hash.h"

static uint64_t g_checked = 0;
static uint64_t g_failures = 0;

static void checkOne(IncrementalPrefixHasher& hasher, const std::string& prefix, int leadingLen,
                      const std::string& alphabet, const uint32_t* cryptTable, uint64_t idx) {
    std::string expectedLeading = indexToString(idx, leadingLen, alphabet);
    std::string expectedFull = prefix + expectedLeading;
    auto expected = mpqHashWithPrefixCache_CPU(expectedFull.c_str(), cryptTable);

    auto actual = hasher.state();
    g_checked++;

    if (hasher.leading() != expectedLeading || actual != expected) {
        g_failures++;
        if (g_failures <= 20) {
            fprintf(stderr, "MISMATCH idx=%llu leadingLen=%d alphabetSize=%zu\n",
                    (unsigned long long) idx, leadingLen, alphabet.size());
            fprintf(stderr, "  expected leading='%s' state=(0x%08X,0x%08X)\n",
                    expectedLeading.c_str(), expected.first, expected.second);
            fprintf(stderr, "  actual   leading='%s' state=(0x%08X,0x%08X)\n",
                    hasher.leading().c_str(), actual.first, actual.second);
        }
    }
}

// Walks reset(start) then advance() through `count` consecutive values,
// checking every single one - covers long dense runs (exercising every kind
// of carry, including deep cascades, somewhere in the run) rather than just
// isolated boundary points.
static void walkRun(const std::string& prefix, int leadingLen, const std::string& alphabet,
                     const uint32_t* cryptTable, uint64_t start, uint64_t count) {
    auto baseState = mpqHashWithPrefixCache_CPU(prefix.c_str(), cryptTable);
    IncrementalPrefixHasher hasher(baseState, leadingLen, alphabet, cryptTable);
    hasher.reset(start);
    checkOne(hasher, prefix, leadingLen, alphabet, cryptTable, start);
    for (uint64_t i = 1; i < count; ++i) {
        hasher.advance();
        checkOne(hasher, prefix, leadingLen, alphabet, cryptTable, start + i);
    }
}

int main() {
    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);

    const std::string real49 = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
    std::vector<std::pair<int, std::string>> alphabets = {
        {29, " -ABCDEFGHIJKLMNOPQRSTUVWXYZ_"},
        {30, " -ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_"},
        {40, " -.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"},
        {41, " -.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_"},
        {42, real49.substr(0, 42)},
        {43, real49.substr(0, 42) + "\\"},
        {47, real49.substr(0, 47)},
        {48, real49.substr(0, 48)},
        {49, real49},
        {50, real49 + "\\"},
    };

    std::vector<std::string> prefixes = {"", "REZ\\", "REZ\\SOME_LONGER_PREFIX_"};
    std::vector<int> leadingLens = {0, 1, 2, 3, 4, 5, 6};

    for (const auto& a : alphabets) {
        int size = a.first;
        const std::string& alphabet = a.second;
        uint64_t R = (uint64_t) size;

        for (const auto& prefix : prefixes) {
            for (int leadingLen : leadingLens) {
                // From the very start of the space - exercises reset() at 0
                // and a long run of advances with no carries at all beyond
                // the last digit for quite a while.
                walkRun(prefix, leadingLen, alphabet, cryptTable, 0, 3000);

                if (leadingLen == 0) continue; // no digits to carry

                uint64_t Rp = R; // R^1
                // Dense run straddling the first single-digit carry boundary.
                if (Rp > 20) walkRun(prefix, leadingLen, alphabet, cryptTable, Rp - 10, 20);

                if (leadingLen >= 2) {
                    uint64_t R2 = R * R;
                    // Straddling a two-digit carry boundary.
                    walkRun(prefix, leadingLen, alphabet, cryptTable, R2 - 10, 20);
                    // Cascading carry: R2 - 1 has both digits at max, so the
                    // very next value ripples through both of them at once.
                    walkRun(prefix, leadingLen, alphabet, cryptTable, R2 - 1, 5);
                }

                if (leadingLen >= 3) {
                    uint64_t R3 = R * R * R;
                    walkRun(prefix, leadingLen, alphabet, cryptTable, R3 - 10, 20);
                    // Deepest possible cascade for this leadingLen: every
                    // digit at max, rippling all the way to the front.
                    uint64_t allMax = R3 - 1; // for leadingLen==3 exactly
                    if (leadingLen == 3) walkRun(prefix, leadingLen, alphabet, cryptTable, allMax, 3);
                }

                // Right at the very top of the representable range for this
                // leadingLen (last few values before it stops making sense
                // to advance() further).
                uint64_t top = 1;
                bool overflowed = false;
                for (int i = 0; i < leadingLen; ++i) {
                    if (top > UINT64_MAX / R) { overflowed = true; break; }
                    top *= R;
                }
                if (!overflowed && top > 20) {
                    walkRun(prefix, leadingLen, alphabet, cryptTable, top - 20, 20);
                }
            }
        }
    }

    printf("checked:  %llu\n", (unsigned long long) g_checked);
    printf("failures: %llu\n", (unsigned long long) g_failures);

    if (g_failures > 0) {
        fprintf(stderr, "\nFAILED\n");
        return 1;
    }
    printf("\nPASSED: IncrementalPrefixHasher agrees with a from-scratch "
           "mpqHashWithPrefixCache_CPU on every one of %llu steps checked.\n", (unsigned long long) g_checked);
    return 0;
}
