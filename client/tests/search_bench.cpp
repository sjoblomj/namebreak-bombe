// Real-code throughput check: times the *actual* runSearch() (not a
// standalone proxy kernel) over a bounded candidateLen=10 range of about
// 28.8 billion candidates (times --scale), using this project's real
// 49-character alphabet and an unreachable target hash so it runs to
// completion instead of stopping early on a match, on the build's default
// backend. Linked against the real search code, so this measures the exact
// code that ships, not a reimplementation of it.
//
// It prunes as the real configuration does (config.conf: symbol runs and
// unopened brackets), and reports the range covered per second, pruned
// candidates included, and the candidates the backend actually searched per
// second (the search rate), which tells a faster kernel apart from more
// pruning. But a range short enough to time only ever varies the last
// leading characters, so how much of *it* gets pruned says little about a
// real search - in the default range, symbol runs can't even occur. So it
// also estimates the share of the whole candidate space the engine would
// prune, from random leading values put through the engine's own checks,
// and projects a real search's rate from that and the search rate. Pruning
// only looks at the leading (CPU) characters, so the window changes both:
// compare windows by the projection.
// --prune symbols gives the configuration every figure measured before
// 2026-09-30 used (symbol runs only); --prune none (or `noprune`) none.
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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#endif
#include "backends/backends.h"
#include "engine/search.h"
#include "engine/candidate.h"

namespace {

// The backend, passed through untouched - every one of SearchBackend's
// functions, the ones with a default too - except that it counts the
// candidates it's asked to search: what's left of the range once the CPU has
// pruned.
class CountingBackend : public SearchBackend {
public:
    explicit CountingBackend(SearchBackend& inner) : inner_(inner) {}
    uint64_t searched = 0;

    const char* name() const override { return inner_.name(); }
    std::vector<int> supportedAlphabetSizes() const override { return inner_.supportedAlphabetSizes(); }
    int windowChars() const override { return inner_.windowChars(); }
    int maxTrailingLen() const override { return inner_.maxTrailingLen(); }
    uint64_t batchSize(int alphabetSize) const override { return inner_.batchSize(alphabetSize); }
    void beginSearch(const SearchConstants& constants) override { inner_.beginSearch(constants); }
    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override {
        searched += count;
        return inner_.runBatch(trailingLen, start, count, params);
    }
    void endSearch() override { inner_.endSearch(); }
    // Forwarded too, or the engine would search one batch at a time through
    // this wrapper, whatever the backend can do.
    int maxBatchesPerCall() const override { return inner_.maxBatchesPerCall(); }
    BatchOutcome runBatches(int trailingLen, const std::vector<BatchRequest>& batches) override {
        for (const BatchRequest& batch : batches)
            searched += batch.count;
        return inner_.runBatches(trailingLen, batches);
    }

private:
    SearchBackend& inner_;
};

// CPU time this process has used so far, in seconds, every thread's.
double processCpuSeconds() {
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
    auto seconds = [](const FILETIME& t) { return (double) (((uint64_t) t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-7; };
    return seconds(kernel) + seconds(user);
#else
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    auto seconds = [](const timeval& t) { return (double) t.tv_sec + (double) t.tv_usec * 1e-6; };
    return seconds(usage.ru_utime) + seconds(usage.ru_stime);
#endif
}

// The share of all leading values of `leadingLen` characters that the engine
// prunes under `req` - estimated from `samples` random ones, run through the
// same three checks as runSearch's leading loop, with the same prefix state.
double prunedShare(const SearchRequest& req, int leadingLen, int samples) {
    std::mt19937_64 random(20260930);
    std::uniform_int_distribution<size_t> character(0, req.alphabet.size() - 1);
    const OpenBrackets prefixOpenBrackets = openBracketsAfter_CPU(req.prefix);
    std::string leading(leadingLen, ' ');
    int pruned = 0;
    for (int n = 0; n < samples; ++n) {
        for (char& c : leading)
            c = req.alphabet[character(random)];
        if ((req.pruneSymbolRuns && hasForbiddenSymbolRun_CPU(leading)) ||
            (req.maxBackslashCount != 0 && countBackslashes_CPU(leading) > req.maxBackslashCount) ||
            (req.pruneUnopenedBrackets && hasUnopenedBracket_CPU(leading, prefixOpenBrackets)))
            ++pruned;
    }
    return (double) pruned / samples;
}

} // namespace

int main(int argc, char** argv) {
    // [--prune all|symbols|none] [noprune] [--backend <name>] [--scale <n>] -
    // the default backend is the first that can run here (see
    // backends/backends.h).
    std::string prune = "all";
    std::string backendName;
    uint64_t scale = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "noprune") {
            prune = "none";
        } else if (arg == "--prune" && i + 1 < argc) {
            prune = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            backendName = argv[++i];
        } else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoull(argv[++i]);
        } else {
            prune = "?";
        }
        if (prune != "all" && prune != "symbols" && prune != "none") {
            fprintf(stderr, "Usage: %s [--prune all|symbols|none] [noprune] [--backend <name>] [--scale <n>]\n", argv[0]);
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
    req.pruneSymbolRuns = prune != "none";
    req.pruneUnopenedBrackets = prune == "all";
    req.continuous = false;

    uint64_t matchCount = 0;
    auto onPartialMatch = [&](const std::string&) { matchCount++; };

    std::string error;
    std::unique_ptr<SearchBackend> realBackend = createBackend(backendName, error);
    if (!realBackend) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    CountingBackend backend(*realBackend);
    SearchRequest warmUp = req;
    warmUp.upperBound = indexToString(targetTotalCandidates / scale / 100, candidateLen, alphabet);
    runSearch(backend, warmUp);
    backend.searched = 0;

    const double cpuStart = processCpuSeconds();
    auto start = std::chrono::steady_clock::now();
    SearchResult result = runSearch(backend, req, nullptr, onPartialMatch);
    auto end = std::chrono::steady_clock::now();
    const double cpuSeconds = processCpuSeconds() - cpuStart;

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

    printf("\n=== search_bench: real runSearch() on the %s backend, candidateLen=%d, window %d, pruning: %s ===\n",
           backend.name(), candidateLen, backend.windowChars(),
           prune == "all" ? "symbol runs and brackets" : prune == "symbols" ? "symbol runs" : "none");
    printf("candidates: %llu\n", (unsigned long long) totalCandidates);
    printf("searched:   %llu (%.2f%% of the range; the rest pruned on the CPU)\n", (unsigned long long) backend.searched,
           100.0 * (double) backend.searched / (double) totalCandidates);
    printf("time:       %.3f s\n", seconds);
    // A GPU backend's host thread mostly waits for the GPU: how it waits shows here.
    printf("CPU time:   %.3f s (%.0f%% of a core)\n", cpuSeconds, 100.0 * cpuSeconds / seconds);
    printf("throughput: %.3f G candidates/sec (of the range, pruned ones included)\n", totalCandidates / seconds / 1e9);
    const double searchRate = backend.searched / seconds / 1e9;
    printf("search rate: %.3f G candidates/sec (searched by the backend)\n", searchRate);
    // How runSearch splits a candidate of this length (see its trailingLen).
    const int leadingLen = std::max(0, candidateLen - backend.windowChars());
    const double share = prunedShare(req, leadingLen, 4'000'000);
    printf("real search: %.2f%% of all %d-character leading values pruned (estimated) -> %.3f G candidates/sec projected\n",
           100.0 * share, leadingLen, searchRate / (1.0 - share));

    // With nothing pruned, the engine must have asked the backend for every
    // candidate of the range exactly once.
    if (prune == "none" && backend.searched != totalCandidates) {
        fprintf(stderr, "ERROR: the backend was asked for %llu candidates, but the range has %llu\n",
                (unsigned long long) backend.searched, (unsigned long long) totalCandidates);
        return 1;
    }
    printf("(incidental hashA collisions along the way, harmless: %llu)\n", (unsigned long long) matchCount);
    return 0;
}
