// Isolates and measures IncrementalPrefixHasher's own claimed speedup - O(1)
// amortized per leadingIdx step vs. the old from-scratch
// mpqHashWithPrefixCache_CPU(prefix + indexToString(leadingIdx, ...)) every
// step - independent of the separate GPU-window-size question (that's
// tests/search_bench.cpp's job, with -DNAMEBREAK_BENCH_WINDOW=N). Pure CPU, no
// GPU involved, so unaffected by anything the GPU is doing.
//
// Both variants build the same `prefix + leading` string every iteration
// (matching what runSearch's leadingIdx loop actually does either way, for
// BatchParams::prefix) - only the hashing itself differs, so the measured
// ratio isolates exactly the thing IncrementalPrefixHasher changed, with the
// (identical, so if anything conservative) string-building cost counted in
// both.

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "engine/candidate.h"
#include "engine/mpq_hash.h"

static double benchNaive(const std::string& prefix, int leadingLen, const std::string& alphabet,
                          const uint32_t* cryptTable, uint64_t iterations) {
    volatile uint32_t sink = 0;
    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
        std::string leading = indexToString(i, leadingLen, alphabet);
        std::string extended = prefix + leading;
        auto h = mpqHashWithPrefixCache_CPU(extended.c_str(), cryptTable);
        sink ^= h.first;
    }
    auto end = std::chrono::steady_clock::now();
    (void) sink;
    return std::chrono::duration<double>(end - start).count();
}

static double benchIncremental(const std::string& prefix, int leadingLen, const std::string& alphabet,
                                const uint32_t* cryptTable, uint64_t iterations) {
    auto baseState = mpqHashWithPrefixCache_CPU(prefix.c_str(), cryptTable);
    IncrementalPrefixHasher hasher(baseState, leadingLen, alphabet, cryptTable);
    hasher.reset(0);
    volatile uint32_t sink = 0;
    auto start = std::chrono::steady_clock::now();
    // i=0's state came from reset() above (not timed, matching how runSearch
    // pays reset()'s one-time cost before the timed leadingIdx loop starts);
    // every iteration here is exactly what one leadingIdx++ step does.
    std::string extended = prefix + hasher.leading();
    sink ^= hasher.state().first;
    for (uint64_t i = 1; i < iterations; ++i) {
        hasher.advance();
        extended = prefix + hasher.leading();
        sink ^= hasher.state().first;
    }
    auto end = std::chrono::steady_clock::now();
    (void) sink;
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);
    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"; // real, 49 chars
    const std::string prefix = "REZ\\";

    struct Case { int leadingLen; uint64_t iterations; const char* label; };
    std::vector<Case> cases = {
        {6, 50'000'000, "leadingLen=6 (this project's real setup: candidateLen=10, gpuWindowChars=4)"},
        {2, 50'000'000, "leadingLen=2 (short candidate, e.g. candidateLen=6)"},
        {11, 20'000'000, "leadingLen=11 (near maxSafeIndexLen - the longest realistic candidates)"},
    };

    for (const auto& c : cases) {
        double tNaive = benchNaive(prefix, c.leadingLen, alphabet, cryptTable, c.iterations);
        double tIncr = benchIncremental(prefix, c.leadingLen, alphabet, cryptTable, c.iterations);
        printf("%s\n", c.label);
        printf("  naive (full rehash each leadingIdx step):  %8.3f s  (%6.1f M steps/sec)\n",
               tNaive, c.iterations / tNaive / 1e6);
        printf("  incremental (IncrementalPrefixHasher):     %8.3f s  (%6.1f M steps/sec)\n",
               tIncr, c.iterations / tIncr / 1e6);
        printf("  speedup: %.2fx\n\n", tNaive / tIncr);
    }
    return 0;
}
