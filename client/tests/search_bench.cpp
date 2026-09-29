// Real-code throughput check: times the *actual* runSearch() (not a
// standalone proxy kernel) over a bounded candidateLen=10 range of about
// 28.8 billion candidates (times --scale), using this project's real
// 49-character alphabet and an unreachable target hash so it runs to
// completion instead of stopping early on a match, on the build's default
// backend. Linked against the real search code, so this measures the exact
// code that ships, not a reimplementation of it.
//
// Creating the backend (for CUDA, the GPU context, and its self-test) and a
// first, small warm-up search happen before the clock starts: a fast GPU
// covers the default range in a few hundredths of a second, which that
// start-up cost would otherwise dominate. --scale 20 gives it a range long
// enough to time well.
//
// Calls the real runSearch(), which does fopen("matches.txt", "a") relative
// to the current directory - the run_search_bench target (CMakeLists.txt)
// runs this from build/testrun/search_bench/ to keep it away from any real
// matches; run the binary directly from somewhere else disposable if not
// going through that target.

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include "backends/backends.h"
#include "engine/search.h"
#include "engine/candidate.h"

int main(int argc, char** argv) {
    // [noprune] [--backend <name>] [--scale <n>] - the default backend is
    // the first that can run here (see backends/backends.h).
    bool pruneSymbolRuns = true;
    std::string backendName;
    uint64_t scale = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "noprune") {
            pruneSymbolRuns = false;
        } else if (arg == "--backend" && i + 1 < argc) {
            backendName = argv[++i];
        } else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoull(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [noprune] [--backend <name>] [--scale <n>]\n", argv[0]);
            return 1;
        }
    }
    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"; // real, 49 chars
    const std::string prefix = "REZ\\";
    const std::string suffix = ".WAV";
    const int candidateLen = 10;
    // Chosen directly in full-candidate space (not tied to any particular
    // leading/trailing split - runSearch() picks that internally via the
    // backend's windowChars()) so this bound, and therefore this benchmark,
    // works unchanged regardless of what NAMEBREAK_GPU_WINDOW_CHARS is
    // compiled with - a fixed ~28.8B total, for direct comparability across runs.
    const uint64_t targetTotalCandidates = 28'800'000'000ULL * scale;
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

    std::string error;
    std::unique_ptr<SearchBackend> backend = createBackend(backendName, error);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    SearchRequest warmUp = req;
    warmUp.upperBound = indexToString(targetTotalCandidates / scale / 100, candidateLen, alphabet);
    runSearch(*backend, warmUp);

    auto start = std::chrono::steady_clock::now();
    SearchResult result = runSearch(*backend, req, nullptr, onPartialMatch);
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
    std::string end_full = makeBoundString(upperBoundLimit, candidateLen);
    uint64_t totalCandidates;
    stringToIndex(end_full, alphabet, totalCandidates, err);

    printf("\n=== search_bench: real runSearch() on the %s backend, candidateLen=%d, pruneSymbolRuns=%s ===\n",
           backend->name(), candidateLen, pruneSymbolRuns ? "true" : "false");
    printf("candidates: %llu\n", (unsigned long long) totalCandidates);
    printf("time:       %.3f s\n", seconds);
    printf("throughput: %.3f G candidates/sec\n", totalCandidates / seconds / 1e9);
    printf("(incidental hashA collisions along the way, harmless: %llu)\n", (unsigned long long) matchCount);
    return 0;
}
