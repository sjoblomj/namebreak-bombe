// Real-code throughput check: times the *actual* runSearch() (not a
// standalone proxy kernel) over a bounded candidateLen=10 range of about
// 28.8 billion candidates, using this project's real 49-character
// alphabet and an unreachable target hash so it runs to completion instead
// of stopping early on a match. Linked against namebreak.cu directly, so
// this measures the exact code that ships, not a reimplementation of it.
//
// Calls the real runSearch(), which does fopen("matches.txt", "a") relative
// to the current directory - `make search_bench` runs this from
// tests/.testrun/ to keep it away from the project's real matches.txt; run
// the binary directly from somewhere else disposable if not going through
// `make search_bench`.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include "../search.h"
#include "../cpu-utils.h"

int main(int argc, char** argv) {
    bool pruneSymbolRuns = !(argc >= 2 && std::string(argv[1]) == "noprune");
    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"; // real, 49 chars
    const std::string prefix = "REZ\\";
    const std::string suffix = ".WAV";
    const int candidateLen = 10;
    // Chosen directly in full-candidate space (not tied to any particular
    // leading/trailing split - namebreak.cu picks that internally via
    // gpuWindowChars) so this bound, and therefore this benchmark, works
    // unchanged regardless of what NAMEBREAK_GPU_WINDOW_CHARS is compiled
    // with - a fixed ~28.8B total, for direct comparability across runs.
    const uint64_t targetTotalCandidates = 28'800'000'000ULL;
    std::string upper = indexToString(targetTotalCandidates, candidateLen, alphabet);
    std::string lower(candidateLen, alphabet[0]);

    SearchRequest req;
    req.alphabet = alphabet;
    req.maxBackslashCount = 0;
    req.prefix = prefix;
    req.suffix = suffix;
    req.startCandidate = lower;
    req.lowerBound = lower;
    req.upperBound = upper;
    req.targetHashA = 0x00000000; // effectively unreachable within this range; just needs to run to completion
    req.targetHashB = 0x00000000;
    req.pruneSymbolRuns = pruneSymbolRuns; // pass "noprune" as argv[1] to disable, for comparison
    req.continuous = false;

    uint64_t matchCount = 0;
    auto onPartialMatch = [&](const std::string&) { matchCount++; };

    auto start = std::chrono::steady_clock::now();
    SearchResult result = runSearch(req, nullptr, onPartialMatch);
    auto end = std::chrono::steady_clock::now();

    if (!result.ok) {
        fprintf(stderr, "runSearch() failed: %s\n", result.error.c_str());
        return 1;
    }

    double seconds = std::chrono::duration<double>(end - start).count();

    // Recompute the exact candidate count the same way runSearch() itself
    // derives its range, rather than assuming the ~28.8B estimate above is
    // exact (it depends on leading/trailing bound edge effects).
    std::string err, upperBoundLimit;
    getUpperBound(upper, alphabet, upperBoundLimit, err);
    std::string end_full = make_bound_string(upperBoundLimit, candidateLen);
    uint64_t totalCandidates;
    stringToIndex(end_full, alphabet, totalCandidates, err);

    printf("\n=== search_bench: real runSearch(), candidateLen=%d, pruneSymbolRuns=%s ===\n",
           candidateLen, pruneSymbolRuns ? "true" : "false");
    printf("candidates: %llu\n", (unsigned long long) totalCandidates);
    printf("time:       %.3f s\n", seconds);
    printf("throughput: %.3f G candidates/sec\n", totalCandidates / seconds / 1e9);
    printf("(incidental hashA collisions along the way, harmless: %llu)\n", (unsigned long long) matchCount);
    return 0;
}
