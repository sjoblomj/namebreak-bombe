// End-to-end correctness test for the *actual* runSearch() (namebreak.cu),
// not a reimplementation of it - linked in directly (built with
// -DNAMEBREAK_NO_MAIN so this file's main() doesn't collide with its own).
//
// Three scenarios, each comparing runSearch()'s hashA-only matches against a
// CPU reference that brute-forces the same range independently:
//
//  1. Leading/trailing split correctness (pruning disabled): IncrementalPrefixHasher
//     itself is already verified in isolation (tests/prefix_hasher_test.cpp);
//     what that can't catch is a mistake in how it's *wired into* runSearch's
//     leadingIdx loop - an off-by-one in when reset()/advance() get called
//     relative to leadingIdx, or a boundary mistake in trailStart/trailEnd.
//  2. pruneSymbolRuns, now CPU-side only (see cpu-utils.h's doc comment on
//     hasForbiddenSymbolRun_CPU for why): confirms runSearch() actually skips
//     every leading value whose first leadingLen characters contain a
//     forbidden run, and still finds real matches under leading values that
//     don't.
//  3. maxBackslashCount, same shape as #2.
//
// A bug in any of these would look exactly like the failure mode this whole
// exercise has been worried about from the start: candidates silently
// missing from a real search, no crash, no error.
//
// Calls the real runSearch(), which does fopen("matches.txt", "a") relative
// to the current directory - `make test` runs this from tests/.testrun/ to
// keep it away from the project's real matches.txt; run the binary directly
// from somewhere else disposable if not going through `make test`.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <string_view>
#include <set>
#include "../search.h"
#include "../cpu-utils.h"
#include "../constants.h"

// Must match namebreak.cu's compiled gpuWindowChars - this test's Makefile
// target doesn't override NAMEBREAK_GPU_WINDOW_CHARS, so that's the default (4).
static const int kGpuWindowChars = 4;

// Mirrors exactly what runSearch()'s leadingIdx loop now does: prune based
// only on the candidate's first leadingLen characters.
static bool isPruned(const std::string& candidate, int leadingLen, bool pruneSymbolRuns, int maxBackslashCount) {
    std::string_view leading(candidate.data(), leadingLen); // view, not a copy - this runs hundreds of millions of times
    if (pruneSymbolRuns && hasForbiddenSymbolRun_CPU(leading))
        return true;
    if (maxBackslashCount != 0 && countBackslashes_CPU(leading) > maxBackslashCount)
        return true;
    return false;
}

// Runs one scenario; returns true on success. `mustBeFound`, if given, is a
// candidate this scenario's range and settings guarantee should NOT be
// pruned and should hashA-match - a sanity check that the scenario is
// exercising a real match, not vacuously passing on an empty result set.
static bool runScenario(const std::string& label, const std::string& alphabet, const std::string& prefix,
                         const std::string& suffix, int candidateLen, const std::string& lower, const std::string& upper,
                         bool pruneSymbolRuns, int maxBackslashCount, uint32_t targetHashA, uint32_t targetHashB,
                         const std::string* mustBeFound) {
    printf("=== %s ===\n", label.c_str());

    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);

    // Mirrors exactly what runSearch() itself does to turn req.lowerBound/
    // req.upperBound into the [start, end) range it will enumerate -
    // bootstrapping the reference range from the same (unmodified, not-under-
    // test) boundary logic, rather than a hand-computed magic number. Using
    // `lower` directly (not getLowerBound(), which computes a *predecessor*
    // for continuous-mode growth - not what a single bounded-mode pass
    // enumerates) matches how runSearch() seeds start_candidate for the one
    // candidateLen these scenarios ever run.
    std::string err, upperBoundLimit;
    if (!getUpperBound(upper, alphabet, upperBoundLimit, err)) {
        fprintf(stderr, "getUpperBound failed: %s\n", err.c_str());
        return false;
    }
    std::string end_full = make_bound_string(upperBoundLimit, candidateLen);
    uint64_t endIdxExclusive;
    if (!stringToIndex(end_full, alphabet, endIdxExclusive, err)) {
        fprintf(stderr, "stringToIndex failed: %s\n", err.c_str());
        return false;
    }
    uint64_t startIdx;
    if (!stringToIndex(lower, alphabet, startIdx, err)) {
        fprintf(stderr, "stringToIndex(lower) failed: %s\n", err.c_str());
        return false;
    }

    int leadingLen = candidateLen - kGpuWindowChars;

    // --- CPU reference: brute-force every candidate in [lower, endIdxExclusive)
    // directly (no leading/trailing split), skipping ones the same
    // leading-only pruning rule runSearch() now applies would skip. Walks
    // candidates via an odometer increment on persistent buffers
    // (candidate/digitIdx/filename), not a fresh indexToString + string
    // concatenation every iteration - at the hundreds of millions of
    // candidates these scenarios cover, that naive form spent nearly all its
    // time in the allocator rather than hashing (the same "incremental beats
    // from-scratch" lesson as IncrementalPrefixHasher itself, just unapplied
    // here at first).
    std::string candidate = lower;
    std::vector<int> digitIdx(candidateLen, 0);
    for (int i = 0; i < candidateLen; ++i) {
        size_t pos = alphabet.find(candidate[i]);
        digitIdx[i] = (int) pos;
    }
    std::string filename = prefix + candidate + suffix;
    size_t candOffset = prefix.size();

    std::set<std::string> referenceMatches;
    uint64_t prunedCount = 0, checkedCount = 0;
    auto refStart = std::chrono::steady_clock::now();
    for (uint64_t idx = startIdx; idx < endIdxExclusive; ++idx) {
        if (idx != startIdx) {
            int p = candidateLen - 1;
            while (p >= 0) {
                if (digitIdx[p] + 1 < (int) alphabet.size()) {
                    digitIdx[p]++;
                } else {
                    digitIdx[p] = 0;
                    candidate[p] = alphabet[0];
                    filename[candOffset + p] = alphabet[0];
                    p--;
                    continue;
                }
                candidate[p] = alphabet[digitIdx[p]];
                filename[candOffset + p] = alphabet[digitIdx[p]];
                break;
            }
        }

        if (isPruned(candidate, leadingLen, pruneSymbolRuns, maxBackslashCount)) {
            prunedCount++;
            continue;
        }
        checkedCount++;
        auto h = mpqHashWithPrefixCache_CPU(filename.c_str(), cryptTable);
        if (h.first == targetHashA) {
            referenceMatches.insert(candidate);
        }
    }
    double refSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - refStart).count();
    uint64_t rangeSize = endIdxExclusive - startIdx;
    printf("CPU reference: %llu total, %llu pruned (leading-only), %llu checked, %zu match(es) [%.2fs, %.1fM/s]\n",
           (unsigned long long) rangeSize, (unsigned long long) prunedCount,
           (unsigned long long) checkedCount, referenceMatches.size(),
           refSeconds, rangeSize / refSeconds / 1e6);

    bool pruningActive = pruneSymbolRuns || maxBackslashCount != 0;
    if (pruningActive && (prunedCount == 0 || checkedCount == 0)) {
        fprintf(stderr, "TEST BUG: this scenario enables pruning but doesn't exercise both pruned "
                        "and non-pruned leading values - it can't prove what it claims.\n");
        return false;
    }
    if (!pruningActive && prunedCount != 0) {
        fprintf(stderr, "TEST BUG: this scenario has pruning disabled but the reference still "
                        "pruned something - isLeadingPruned or this scenario's settings are wrong.\n");
        return false;
    }
    if (checkedCount == 0) {
        fprintf(stderr, "TEST BUG: nothing survived to be checked at all.\n");
        return false;
    }
    if (mustBeFound && referenceMatches.find(*mustBeFound) == referenceMatches.end()) {
        fprintf(stderr, "TEST BUG: reference set doesn't contain the deliberately chosen "
                        "target candidate - this test is broken, not the code under test.\n");
        return false;
    }

    // --- The real runSearch(), same range and settings. ---
    SearchRequest req;
    req.alphabet = alphabet;
    req.maxBackslashCount = maxBackslashCount;
    req.prefix = prefix;
    req.suffix = suffix;
    req.startCandidate = lower;
    req.lowerBound = lower;
    req.upperBound = upper;
    req.targetHashA = targetHashA;
    req.targetHashB = targetHashB;
    req.pruneSymbolRuns = pruneSymbolRuns;
    req.continuous = false;

    std::set<std::string> foundMatches;
    auto onPartialMatch = [&](const std::string& filename) {
        foundMatches.insert(remove_prefix_and_suffix(filename, prefix, suffix));
    };

    auto searchStart = std::chrono::steady_clock::now();
    SearchResult result = runSearch(req, nullptr, onPartialMatch);
    double searchSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart).count();
    if (!result.ok) {
        fprintf(stderr, "runSearch() failed: %s\n", result.error.c_str());
        return false;
    }

    printf("runSearch(): %zu match(es) reported via onPartialMatch [%.2fs]\n", foundMatches.size(), searchSeconds);

    if (foundMatches != referenceMatches) {
        fprintf(stderr, "MISMATCH between runSearch() and the CPU reference:\n");
        for (const auto& c : referenceMatches) {
            if (foundMatches.find(c) == foundMatches.end())
                fprintf(stderr, "  MISSING from runSearch() (silently dropped candidate): '%s'\n", c.c_str());
        }
        for (const auto& c : foundMatches) {
            if (referenceMatches.find(c) == referenceMatches.end())
                fprintf(stderr, "  EXTRA in runSearch() (not in reference, spurious match): '%s'\n", c.c_str());
        }
        fprintf(stderr, "FAILED\n\n");
        return false;
    }

    printf("PASSED\n\n");
    return true;
}

int main() {
    bool allPassed = true;

    // --- Scenario 1: leading/trailing split, pruning disabled. ---
    {
        const std::string full = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
        const std::string alphabet = full.substr(0, 42);
        const std::string prefix = "TEST_";
        const std::string suffix = ".DAT";
        const int candidateLen = 5; // leadingLen = 5-4 = 1

        const std::string lower = std::string(candidateLen, alphabet[0]);
        const std::string upper = std::string(candidateLen, alphabet[7]);

        uint32_t cryptTable[0x500];
        prepareCryptTable(cryptTable);
        // Target picked directly (not via runScenario's pruned-range machinery,
        // since this scenario has no pruning) - middle of the actual range,
        // computed the same way runSearch() itself derives it (see
        // runScenario for why: bootstrapping from getUpperBound/make_bound_string
        // rather than a hand-computed magic number).
        std::string err, upperBoundLimit;
        getUpperBound(upper, alphabet, upperBoundLimit, err);
        std::string end_full = make_bound_string(upperBoundLimit, candidateLen);
        uint64_t endIdxExclusive;
        stringToIndex(end_full, alphabet, endIdxExclusive, err);
        uint64_t targetGlobalIdx = endIdxExclusive / 2;
        std::string targetCandidate = indexToString(targetGlobalIdx, candidateLen, alphabet);
        std::string targetFull = prefix + targetCandidate + suffix;
        auto targetHash = mpqHashWithPrefixCache_CPU(targetFull.c_str(), cryptTable);

        allPassed &= runScenario("1: leading/trailing split (pruning disabled)", alphabet, prefix, suffix,
                                  candidateLen, lower, upper, /*pruneSymbolRuns=*/false, /*maxBackslashCount=*/0,
                                  targetHash.first, 0xDEADBEEF, &targetCandidate);
    }

    // --- Scenario 2: pruneSymbolRuns, CPU-side, leading characters only. ---
    // alphabet[0]='!' (a forbidden-run character), alphabet[1]='A' (alnum).
    // leadingLen=4 (candidateLen=8, gpuWindowChars=4): leadingIdx in [0,42)
    // all start "!!!" - guaranteed pruned (3-run); leadingIdx in [42,46) start
    // "!!A" - guaranteed NOT pruned by this rule (the run resets at 'A').
    {
        const std::string alphabet = "!A &'()+,-.0123456789BCDEFGHIJKLMNOPQRSTUV"; // 42 chars
        const std::string prefix = "TEST_";
        const std::string suffix = ".DAT";
        const int candidateLen = 8;
        const uint64_t trailingSpace = 42ULL * 42 * 42 * 42; // gpuWindowChars=4

        const std::string lower = std::string(candidateLen, alphabet[0]);
        std::string leadingPart = indexToString(46, candidateLen - kGpuWindowChars, alphabet); // spans both groups (42 pruned + a few non-pruned)
        std::string trailingPart(kGpuWindowChars, alphabet[alphabet.size() - 2]); // avoid cascading past leadingIdx=85
        std::string upper = leadingPart + trailingPart;

        uint32_t cryptTable[0x500];
        prepareCryptTable(cryptTable);
        uint64_t targetGlobalIdx = 44ULL * trailingSpace + 12345; // leadingIdx=44: in the non-pruned [42,46) group
        std::string targetCandidate = indexToString(targetGlobalIdx, candidateLen, alphabet);
        std::string targetFull = prefix + targetCandidate + suffix;
        auto targetHash = mpqHashWithPrefixCache_CPU(targetFull.c_str(), cryptTable);

        allPassed &= runScenario("2: pruneSymbolRuns (CPU-side, leading characters only)", alphabet, prefix, suffix,
                                  candidateLen, lower, upper, /*pruneSymbolRuns=*/true, /*maxBackslashCount=*/0,
                                  targetHash.first, 0xDEADBEEF, &targetCandidate);
    }

    // --- Scenario 3: maxBackslashCount, CPU-side, leading characters only. ---
    // alphabet[0]='\\', alphabet[1]='A', maxBackslashCount=2. leadingIdx in
    // [0,42): leading="\\\\\\"+X - 3-4 backslashes, pruned. leadingIdx in
    // [42,46): leading="\\\\A"+X - 2 backslashes (3 only at the single
    // leadingIdx=42 edge case) - mostly NOT pruned.
    {
        const std::string alphabet = "\\A &'()+,-.0123456789BCDEFGHIJKLMNOPQRSTUV"; // 42 chars
        const std::string prefix = "TEST_";
        const std::string suffix = ".DAT";
        const int candidateLen = 8;
        const uint64_t trailingSpace = 42ULL * 42 * 42 * 42;

        const std::string lower = std::string(candidateLen, alphabet[0]);
        std::string leadingPart = indexToString(46, candidateLen - kGpuWindowChars, alphabet); // spans both groups (42 pruned + a few non-pruned)
        std::string trailingPart(kGpuWindowChars, alphabet[alphabet.size() - 2]);
        std::string upper = leadingPart + trailingPart;

        uint32_t cryptTable[0x500];
        prepareCryptTable(cryptTable);
        uint64_t targetGlobalIdx = 44ULL * trailingSpace + 12345; // leadingIdx=44, clear of the leadingIdx=42 edge case, within the shrunk [42,46) range
        std::string targetCandidate = indexToString(targetGlobalIdx, candidateLen, alphabet);
        std::string targetFull = prefix + targetCandidate + suffix;
        auto targetHash = mpqHashWithPrefixCache_CPU(targetFull.c_str(), cryptTable);

        allPassed &= runScenario("3: maxBackslashCount (CPU-side, leading characters only)", alphabet, prefix, suffix,
                                  candidateLen, lower, upper, /*pruneSymbolRuns=*/false, /*maxBackslashCount=*/2,
                                  targetHash.first, 0xDEADBEEF, &targetCandidate);
    }

    if (!allPassed) {
        fprintf(stderr, "One or more scenarios FAILED - see above.\n");
        return 1;
    }
    printf("ALL SCENARIOS PASSED\n");
    return 0;
}
