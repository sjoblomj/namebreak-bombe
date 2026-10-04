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
// --whole prunes at every character but the last
// (SearchRequest::pruneWholeCandidate): the backend then leaves out some of
// the rows it's asked for, so the search rate counts candidates it was asked
// for, and the projection those it searched - the share of the range's
// candidates that survive the rules, estimated too - over the share of all
// candidates of this length that survive them.
//
// --insert-from-start/--insert-from-end <text>,<position> insert text into
// every candidate (SearchRequest::insertFromStart/insertFromEnd), to time
// what it costs where it lands - see the README's "Inserted text".
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
#include "engine/limits.h"

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

// Whether `req` prunes the first `checkedLen` characters of `candidate` - the
// same three checks as runSearch's leading loop, with the same prefix state.
bool prunes(const SearchRequest& req, const std::string& candidate, int checkedLen) {
    const std::string_view checked(candidate.data(), checkedLen);
    return (req.pruneSymbolRuns && hasForbiddenSymbolRun_CPU(checked)) ||
           (req.maxBackslashCount != 0 && countBackslashes_CPU(checked) > req.maxBackslashCount) ||
           (req.pruneUnopenedBrackets && hasUnopenedBracket_CPU(checked, openBracketsAfter_CPU(req.prefix)));
}

// The share of all candidates of `len` characters that `req` prunes by their
// first `checkedLen` - estimated from `samples` random ones.
double prunedShare(const SearchRequest& req, int len, int checkedLen, int samples) {
    std::mt19937_64 random(20260930);
    std::uniform_int_distribution<size_t> character(0, req.alphabet.size() - 1);
    std::string candidate(len, ' ');
    int pruned = 0;
    for (int n = 0; n < samples; ++n) {
        for (char& c : candidate)
            c = req.alphabet[character(random)];
        pruned += prunes(req, candidate, checkedLen);
    }
    return (double) pruned / samples;
}

// Of the candidates in [0, total) that survive their first `leadingLen`
// characters' check - those the engine asks a backend for - the share that
// also survive `checkedLen`'s, estimated from `samples` random ones.
double survivingShare(const SearchRequest& req, uint64_t total, int len, int leadingLen, int checkedLen, int samples) {
    std::mt19937_64 random(20260930);
    std::uniform_int_distribution<uint64_t> index(0, total - 1);
    int asked = 0, survived = 0;
    while (asked < samples) {
        const std::string candidate = indexToString(index(random), len, req.alphabet);
        if (prunes(req, candidate, leadingLen))
            continue;
        ++asked;
        survived += !prunes(req, candidate, checkedLen);
    }
    return (double) survived / asked;
}

} // namespace

int main(int argc, char** argv) {
    // [--prune all|symbols|none] [noprune] [--whole] [--backend <name>]
    // [--scale <n>] [--size <n>] [--suffix <text>] - the default backend is the first that can
    // run here (see backends/backends.h).
    std::string prune = "all";
    bool whole = false;
    std::string backendName;
    uint64_t scale = 1;
    int alphabetSize = 49;
    std::string suffix = ".WAV";
    Insertion insertFromStart, insertFromEnd;
    // "<text>,<position>", split at the last comma.
    auto parseInsertion = [](const std::string& value, Insertion& out) {
        const size_t comma = value.rfind(',');
        if (comma == std::string::npos || comma == 0)
            return false;
        out.text = value.substr(0, comma);
        out.position = std::stoi(value.substr(comma + 1));
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "noprune") {
            prune = "none";
        } else if (arg == "--prune" && i + 1 < argc) {
            prune = argv[++i];
        } else if (arg == "--whole") {
            whole = true;
        } else if (arg == "--backend" && i + 1 < argc) {
            backendName = argv[++i];
        } else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoull(argv[++i]);
        } else if ((arg == "--insert-from-start" || arg == "--insert-from-end") && i + 1 < argc) {
            if (!parseInsertion(argv[++i], arg == "--insert-from-start" ? insertFromStart : insertFromEnd))
                prune = "?";
        } else if (arg == "--suffix" && i + 1 < argc) {
            suffix = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            alphabetSize = std::stoi(argv[++i]);
            if (alphabetSize < 1 || alphabetSize > MAX_ALPHABET_SIZE)
                prune = "?";
        } else {
            prune = "?";
        }
        if (prune != "all" && prune != "symbols" && prune != "none") {
            fprintf(stderr,
                    "Usage: %s [--prune all|symbols|none] [noprune] [--whole] [--backend <name>] [--scale <n>] [--size 1-%d]\n"
                    "       [--suffix <text>] [--insert-from-start <text>,<position>] [--insert-from-end <text>,<position>]\n",
                    argv[0], MAX_ALPHABET_SIZE);
            return 1;
        }
    }
    // The real 49 characters - or with --size, the first that many of these
    // (14 more after the real ones, up to MAX_ALPHABET_SIZE), to time other sizes.
    const std::string alphabet = std::string(" !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_#$%*:;<=>?@^`~").substr(0, alphabetSize);
    const std::string prefix = "REZ\\";
    const int candidateLen = 10;
    // Chosen directly in full-candidate space (not tied to any particular
    // leading/trailing split - runSearch() picks that internally via the
    // backend's windowChars()) so this bound, and therefore this benchmark,
    // works unchanged regardless of what NAMEBREAK_GPU_WINDOW_CHARS is
    // compiled with - a fixed ~28.8B total, for direct comparability across runs.
    // With a small alphabet, the whole length may be less than that.
    const uint64_t targetTotalCandidates = 28'800'000'000ULL * scale;
    uint64_t space = 1;
    for (int i = 0; i < candidateLen && space <= targetTotalCandidates; ++i)
        space *= alphabetSize;
    std::string upper = targetTotalCandidates < space ? indexToString(targetTotalCandidates, candidateLen, alphabet)
                                                      : std::string(candidateLen, alphabet.back());
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
    req.pruneWholeCandidate = whole;
    req.insertFromStart = insertFromStart;
    req.insertFromEnd = insertFromEnd;
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
    const uint64_t warmUpCandidates = targetTotalCandidates / scale / 100;
    warmUp.upperBound = warmUpCandidates < space ? indexToString(warmUpCandidates, candidateLen, alphabet) : upper;
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

    printf("\n=== search_bench: real runSearch() on the %s backend, candidateLen=%d, window %d, pruning: %s%s ===\n",
           backend.name(), candidateLen, backend.windowChars(),
           prune == "all" ? "symbol runs and brackets" : prune == "symbols" ? "symbol runs" : "none",
           prune != "none" ? (whole ? ", every character but the last" : ", leading characters") : "");
    printf("candidates: %llu\n", (unsigned long long) totalCandidates);
    printf("searched:   %llu (%.2f%% of the range; the rest pruned on the CPU)\n", (unsigned long long) backend.searched,
           100.0 * (double) backend.searched / (double) totalCandidates);
    printf("time:       %.3f s\n", seconds);
    // A GPU backend's host thread mostly waits for the GPU: how it waits shows here.
    printf("CPU time:   %.3f s (%.0f%% of a core)\n", cpuSeconds, 100.0 * cpuSeconds / seconds);
    printf("throughput: %.3f G candidates/sec (of the range, pruned ones included)\n", totalCandidates / seconds / 1e9);
    const double searchRate = backend.searched / seconds / 1e9;
    printf("search rate: %.3f G candidates/sec (asked of the backend)\n", searchRate);
    // How runSearch splits a candidate of this length (see its trailingLen),
    // and how much of it the rules look at.
    const int leadingLen = std::max(0, candidateLen - backend.windowChars());
    const int checkedLen = whole ? candidateLen - 1 : leadingLen;
    const double searchedShare = whole ? survivingShare(req, totalCandidates, candidateLen, leadingLen, checkedLen, 1'000'000) : 1.0;
    if (whole)
        printf("of which:   %.2f%% searched (estimated; the rest pruned by the backend)\n", 100.0 * searchedShare);
    const double share = prunedShare(req, candidateLen, checkedLen, 4'000'000);
    printf("real search: %.2f%% of all %d-character candidates pruned (estimated, by their first %d) -> %.3f G candidates/sec projected\n",
           100.0 * share, candidateLen, checkedLen, searchRate * searchedShare / (1.0 - share));

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
