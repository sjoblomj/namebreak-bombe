// End-to-end correctness test for the *actual* runSearch() (engine/search.cpp)
// and backends, not a reimplementation of them - linked in directly.
//
// Every case compares runSearch()'s results against an independent CPU
// reference that brute-forces the same range one candidate at a time (with
// its own copy of the leading-only pruning rule), so a bug anywhere in the
// search pipeline looks like what this whole exercise is worried about: a
// candidate silently missing from a real search (or a spurious one), no
// crash, no error. Nothing here trusts the code under test to say what the
// right answer is.
//
// It tests one backend (--backend <name>, default: the first that can run
// here - see backends/backends.h). This binary is built several times (CMakeLists.txt),
// with different -DNAMEBREAK_GPU_WINDOW_CHARS / -DNAMEBREAK_ROWS_PER_LAUNCH
// values (see backends/cuda/tuning.h and CMakeLists.txt), so the same cases
// exercise different leading/trailing splits, kernel row-decode depths and
// launch-chunking boundaries. Every case derives its geometry from the
// backend under test (SearchBackend::windowChars/batchSize); none hard-codes
// a window size.
//
// Groups:
//  1-5. The original scenarios: leading/trailing split, pruneSymbolRuns,
//     pruneUnopenedBrackets and maxBackslashCount (CPU-side, leading characters only - see
//     hasForbiddenSymbolRun_CPU's comment in candidate.h), maxBackslashCount
//     excluding the prefix, and lower_bound == upper_bound. Pruning ones run
//     twice: once with the target on the surviving side (must be found) and
//     once on the pruned side (must NOT be).
//  G. Geometry, for each of the 6 supported alphabet sizes: the kernel
//     handles a *row* (every value of the last character) per thread, so
//     ranges that start/end mid-row, sit inside one row, are row-aligned,
//     straddle a leading-value boundary or a launch boundary, or touch the
//     very first/last candidate, each take a different path through it.
//  K. Every last-character position k for every alphabet size (the kernel's
//     inner loop is fully unrolled with k as a compile-time constant, so a
//     wrong constant at one position would only ever affect candidates with
//     that exact last character).
//  P. Prefix and suffix lengths (0..63, including bytes >= 0x80), which pick
//     between the kernel's compile-time suffix instantiations and its
//     runtime fallback - plus the length limits' error paths.
//  F. The both-hashes-match path: found/filename, a match in the very last
//     candidate, hashA-only hits not counting as found, no stale "found"
//     state leaking into the next runSearch() call in the same process, and a
//     genuine hashA collision pair (two candidates, one launch) with hashB
//     picking the winner.
//  A. Abort: a pre-set abort flag stops the search before any GPU work.
//  Z. A seeded fuzzer over random alphabets (including bytes >= 0x80 and
//     non-ASCII order), prefixes, suffixes, candidate lengths and ranges.
//
// Every runSearch() call's stdout/stderr is captured, and any "WARNING" in it
// fails the case: verifyMatchesKernel re-derives each reported hit through
// the original, independent hashing path and prints one whenever that
// disagrees with the fast kernel that reported it.
//
// Calls the real runSearch(), which does fopen("matches.txt", "a") relative
// to the current directory - ctest runs each test in its own directory under
// build/testrun/ to keep it away from any real matches; run the binary
// directly from somewhere else disposable if not going through ctest.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "engine/search.h"
#include "engine/candidate.h"
#include "engine/mpq_hash.h"
#include "backends/backends.h"
#include "engine/limits.h"
#include "engine/search.h"

// The backend under test, and its window size (set in main()).
static std::unique_ptr<SearchBackend> g_backend;
static int g_window = 0;
static const int kAlphabetSizes[] = {29, 30, 40, 41, 42, 43, 47, 48, 49, 50};

static uint32_t g_cryptTable[0x500];
static int g_cases = 0;
static int g_failures = 0;

// ---------------------------------------------------------------------------
// Output capture (POSIX only - elsewhere runSearch() just prints to the console
// and the "no WARNING" check is skipped).
// ---------------------------------------------------------------------------
class OutputCapture {
public:
    void start() {
#ifndef _WIN32
        fflush(stdout);
        fflush(stderr);
        fd_ = open(".capture.tmp", O_CREAT | O_TRUNC | O_WRONLY, 0644);
        savedOut_ = dup(1);
        savedErr_ = dup(2);
        dup2(fd_, 1);
        dup2(fd_, 2);
#endif
    }
    std::string stop() {
#ifndef _WIN32
        fflush(stdout);
        fflush(stderr);
        dup2(savedOut_, 1);
        dup2(savedErr_, 2);
        close(savedOut_);
        close(savedErr_);
        close(fd_);
        std::ifstream in(".capture.tmp");
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
#else
        return "";
#endif
    }
private:
    int fd_ = -1, savedOut_ = -1, savedErr_ = -1;
};

// ---------------------------------------------------------------------------
// CPU reference helpers
// ---------------------------------------------------------------------------
static uint32_t hashWithTable(const std::string& s, int tableOffset) {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : s) {
        seed1 = g_cryptTable[tableOffset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}
static uint32_t hashA(const std::string& s) { return hashWithTable(s, 0x100); }
static uint32_t hashB(const std::string& s) { return hashWithTable(s, 0x200); }

// Mirrors runSearch()'s own computation of how many candidate characters the
// backend enumerates (search.cpp: maxSafeIndexLen / trailingLen) - needed
// because the leading-only pruning rules apply to exactly the other characters.
static int maxSafeIndexLenFor(int alphabetSize) {
    uint64_t product = 1;
    int n = 0;
    while (n < MAX_CANDIDATE_LEN && product <= UINT64_MAX / alphabetSize) {
        product *= alphabetSize;
        n++;
    }
    return n;
}
static int trailingLenFor(int candidateLen, int alphabetSize) {
    return std::min(candidateLen, std::max(g_window, candidateLen - maxSafeIndexLenFor(alphabetSize)));
}

static bool isPruned(const std::string& candidate, int leadingLen, bool pruneSymbolRuns, int maxBackslashCount, bool pruneUnopenedBrackets,
                     const std::string& prefix) {
    std::string_view leading(candidate.data(), leadingLen);
    if (pruneSymbolRuns && hasForbiddenSymbolRun_CPU(leading))
        return true;
    if (maxBackslashCount != 0 && countBackslashes_CPU(leading) > maxBackslashCount)
        return true;
    if (pruneUnopenedBrackets && hasUnopenedBracket_CPU(leading, openBracketsAfter_CPU(prefix)))
        return true;
    return false;
}

// Advances `c` to its successor in `alphabet`'s order (last character
// fastest); false if it was already the very last candidate.
static bool stepCandidate(std::string& c, const std::string& alphabet) {
    // Leaves `c` untouched when it is already the last candidate (all alphabet.back()).
    if (c.find_first_not_of(alphabet.back()) == std::string::npos)
        return false;
    for (int p = (int) c.size() - 1; p >= 0; --p) {
        size_t pos = alphabet.find(c[p]);
        if (pos + 1 < alphabet.size()) {
            c[p] = alphabet[pos + 1];
            return true;
        }
        c[p] = alphabet[0];
    }
    return false; // unreachable
}

// Number of steps actually taken (< n if the end of the candidate space was hit).
static uint64_t advanceBy(std::string& c, const std::string& alphabet, uint64_t n) {
    uint64_t done = 0;
    while (done < n && stepCandidate(c, alphabet)) ++done;
    return done;
}

static std::string alphabetOfSize(int n) {
    static const std::string full = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"; // 49 characters
    if (n <= 49) return full.substr(0, n);
    return full + "#";
}

// Order of two equal-length candidates in `alphabet`'s ordering (the order a search enumerates them in).
static bool lessInAlphabet(const std::string& a, const std::string& b, const std::string& alphabet) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        return alphabet.find(a[i]) < alphabet.find(b[i]);
    }
    return false;
}

static uint64_t ipow(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i) r *= base;
    return r;
}

// ---------------------------------------------------------------------------
// One case: run the reference and runSearch() on the same request, compare.
// ---------------------------------------------------------------------------
struct CaseSpec {
    std::string alphabet, prefix, suffix, lower, upper;
    bool pruneSymbolRuns = false;
    int maxBackslashCount = 0;
    bool pruneUnopenedBrackets = false;
    uint32_t targetHashA = 0x12345678;
    uint32_t targetHashB = 0xDEADBEEF;
    // A candidate the reference must contain in its match set (a sanity check
    // that the case exercises a real match instead of passing vacuously).
    std::string mustBeFound;
    // The candidate runSearch() must report as matching *both* hashes
    // (result.found / result.filename). Empty: it must report not found.
    std::string expectFound;
    // The case enables pruning and must exercise both pruned and surviving values.
    bool requireBothPrunedAndChecked = false;
};

static bool runCase(const std::string& label, const CaseSpec& c, bool verbose = false) {
    ++g_cases;
    const int alphabetSize = (int) c.alphabet.size();
    const int candidateLen = (int) c.lower.size();
    const int leadingLen = candidateLen - trailingLenFor(candidateLen, alphabetSize);

    // --- CPU reference: every candidate in [lower, upper], inclusive. ---
    std::set<std::string> referenceMatches;
    uint64_t total = 0, pruned = 0, checked = 0;
    {
        std::string cand = c.lower;
        const uint64_t kMaxRange = 20'000'000;
        while (true) {
            ++total;
            if (isPruned(cand, leadingLen, c.pruneSymbolRuns, c.maxBackslashCount, c.pruneUnopenedBrackets, c.prefix)) {
                ++pruned;
            } else {
                ++checked;
                if (hashA(c.prefix + cand + c.suffix) == c.targetHashA)
                    referenceMatches.insert(cand);
            }
            if (cand == c.upper) break;
            if (total >= kMaxRange || !stepCandidate(cand, c.alphabet)) {
                fprintf(stderr, "TEST BUG in '%s': upper bound '%s' not reached from lower bound '%s' within the reference's limit\n",
                        label.c_str(), c.upper.c_str(), c.lower.c_str());
                ++g_failures;
                return false;
            }
        }
    }
    if (verbose)
        printf("  reference: %llu candidates, %llu pruned, %llu checked, %zu match(es)\n",
               (unsigned long long) total, (unsigned long long) pruned, (unsigned long long) checked, referenceMatches.size());

    auto testBug = [&](const char* why) {
        fprintf(stderr, "TEST BUG in '%s': %s\n", label.c_str(), why);
        ++g_failures;
        return false;
    };
    if (c.requireBothPrunedAndChecked && (pruned == 0 || checked == 0))
        return testBug("pruning is enabled but the range doesn't exercise both pruned and surviving values");
    if (!c.requireBothPrunedAndChecked && (c.pruneSymbolRuns || c.maxBackslashCount != 0 || c.pruneUnopenedBrackets) == false && pruned != 0)
        return testBug("pruning is disabled but the reference pruned something");
    if (checked == 0)
        return testBug("nothing survived to be checked at all");
    if (!c.mustBeFound.empty() && !referenceMatches.count(c.mustBeFound))
        return testBug("the reference doesn't contain the deliberately chosen target candidate");

    // --- The real runSearch(). ---
    SearchRequest req;
    req.alphabet = c.alphabet;
    req.maxBackslashCount = c.maxBackslashCount;
    req.prefix = c.prefix;
    req.suffix = c.suffix;
    req.startCandidate = c.lower;
    req.lowerBound = c.lower;
    req.upperBound = c.upper;
    req.targetHashA = c.targetHashA;
    req.targetHashB = c.targetHashB;
    req.pruneSymbolRuns = c.pruneSymbolRuns;
    req.pruneUnopenedBrackets = c.pruneUnopenedBrackets;
    req.continuous = false;

    std::set<std::string> reported;
    std::vector<std::string> reportedInOrder;
    auto onPartialMatch = [&](const std::string& filename) {
        std::string cand = removePrefixAndSuffix(filename, c.prefix, c.suffix);
        reported.insert(cand);
        reportedInOrder.push_back(cand);
    };

    OutputCapture capture;
    capture.start();
    SearchResult result = runSearch(*g_backend, req, nullptr, onPartialMatch);
    std::string log = capture.stop();

    bool ok = true;
    auto fail = [&](const std::string& msg) {
        if (ok) fprintf(stderr, "FAILED: %s\n", label.c_str());
        ok = false;
        fprintf(stderr, "  %s\n", msg.c_str());
    };

    if (!result.ok) {
        fail("runSearch() failed: " + result.error);
    } else {
        for (const auto& m : referenceMatches) {
            // Finding a match ends the search once its launch is done, so candidates *after* the
            // found one in enumeration order may legitimately go unreported (whether they do
            // depends on where the launch boundaries happen to fall). Everything up to and
            // including it must be reported.
            bool mayBeUnreported = !c.expectFound.empty() && lessInAlphabet(c.expectFound, m, c.alphabet);
            if (!reported.count(m) && !mayBeUnreported)
                fail("MISSING from runSearch() (silently dropped candidate): '" + m + "'");
        }
        for (const auto& m : reported)
            if (!referenceMatches.count(m)) fail("EXTRA in runSearch() (not in the reference, spurious match): '" + m + "'");
        if (reportedInOrder.size() != reported.size()) fail("the same match was reported more than once");

        if (c.expectFound.empty()) {
            if (result.found) fail("runSearch() reported found=true ('" + result.filename + "') but nothing should have matched hashB");
        } else {
            std::string expectedName = c.prefix + c.expectFound + c.suffix;
            if (!result.found) fail("runSearch() reported found=false, expected '" + expectedName + "'");
            else if (result.filename != expectedName) fail("runSearch() found '" + result.filename + "', expected '" + expectedName + "'");
        }
    }
    if (log.find("WARNING") != std::string::npos) {
        size_t at = log.find("WARNING");
        fail("a WARNING was printed (the kernels disagree with the reference hashing path): " + log.substr(at, 200));
    }
    if (!ok) {
        fprintf(stderr, "  alphabet size %d, prefix '%s' (%zu), suffix '%s' (%zu), range '%s'..'%s', backend %s, window %d, batch size %llu\n",
                alphabetSize, c.prefix.c_str(), c.prefix.size(), c.suffix.c_str(), c.suffix.size(), c.lower.c_str(), c.upper.c_str(),
                g_backend->name(), g_window, (unsigned long long) g_backend->batchSize(alphabetSize));
        ++g_failures;
    }
    return ok;
}

// Convenience: a case over candidate indices [startIdx, endIdx] (candidates of
// length `len`, numbered in the alphabet's order) with hashA/hashB targeted at
// the candidate at `targetIdx`. If `alsoHashB`, the target's hashB is used too
// and it must be reported as *found*.
static bool runIndexCase(const std::string& label, const std::string& alphabet, const std::string& prefix,
                          const std::string& suffix, int len, uint64_t startIdx, uint64_t endIdx, uint64_t targetIdx,
                          bool alsoHashB = false) {
    CaseSpec c;
    c.alphabet = alphabet;
    c.prefix = prefix;
    c.suffix = suffix;
    c.lower = indexToString(startIdx, len, alphabet);
    c.upper = indexToString(endIdx, len, alphabet);
    std::string target = indexToString(targetIdx, len, alphabet);
    c.targetHashA = hashA(prefix + target + suffix);
    if (alsoHashB) {
        c.targetHashB = hashB(prefix + target + suffix);
        c.expectFound = target;
    }
    c.mustBeFound = target;
    return runCase(label, c);
}

// The target is a candidate just *outside* [startIdx, endIdx] (adjacent to it, typically in the
// very same row): nothing may be reported - i.e. the kernel must not run past the range's edges.
static bool runOutsideCase(const std::string& label, const std::string& alphabet, const std::string& prefix,
                            const std::string& suffix, int len, uint64_t startIdx, uint64_t endIdx, uint64_t outsideIdx) {
    CaseSpec c;
    c.alphabet = alphabet;
    c.prefix = prefix;
    c.suffix = suffix;
    c.lower = indexToString(startIdx, len, alphabet);
    c.upper = indexToString(endIdx, len, alphabet);
    c.targetHashA = hashA(prefix + indexToString(outsideIdx, len, alphabet) + suffix);
    return runCase(label, c);
}

// Same range, but a target that matches nothing in it: nothing may be reported.
static bool runNoMatchCase(const std::string& label, const std::string& alphabet, const std::string& prefix,
                            const std::string& suffix, int len, uint64_t startIdx, uint64_t endIdx) {
    CaseSpec c;
    c.alphabet = alphabet;
    c.prefix = prefix;
    c.suffix = suffix;
    c.lower = indexToString(startIdx, len, alphabet);
    c.upper = indexToString(endIdx, len, alphabet);
    c.targetHashA = 0x0BADF00D;
    return runCase(label, c);
}

// ---------------------------------------------------------------------------
// Groups 1-5: the original scenarios, geometry derived from the window size.
// ---------------------------------------------------------------------------
static bool scenarioSplit() {
    printf("=== 1: leading/trailing split (pruning disabled) ===\n");
    const std::string alphabet = alphabetOfSize(42);
    const int len = g_window + 1; // leadingLen = 1
    const uint64_t T = ipow(alphabet.size(), g_window);
    bool ok = true;
    // Straddling one leading-value boundary (a range covering more than that would be
    // a whole leading value - far too many candidates for the CPU reference at larger
    // windows), plus the range's own first/last candidates.
    const uint64_t start = 5 * T - 250, end = 5 * T + 250;
    for (uint64_t target : {start, 5 * T - 1, 5 * T, 5 * T + 100, end})
        ok &= runIndexCase("1: split, target idx " + std::to_string(target), alphabet, "TEST_", ".DAT", len, start, end, target);
    return ok;
}

// Shared shape of scenarios 2 and 3: two adjacent leading values, the first
// pruned and the second not, each only partially covered by the range so the
// CPU reference stays cheap regardless of the window size.
static bool pruningScenario(const char* name, const std::string& alphabet, const std::string& prefix, bool prune, int maxBackslash,
                             int leadingLen, uint64_t prunedLeadingIdx, uint64_t survivingLeadingIdx, bool pruneBrackets = false) {
    printf("=== %s ===\n", name);
    const int len = g_window + leadingLen;
    const uint64_t T = ipow(alphabet.size(), g_window);
    // The range must run from the tail of one leading value into the head of the next.
    if (survivingLeadingIdx != prunedLeadingIdx + 1) {
        fprintf(stderr, "TEST BUG: %s needs adjacent leading values\n", name);
        return false;
    }
    const uint64_t start = survivingLeadingIdx * T - 200, end = survivingLeadingIdx * T + 200;
    bool ok = true;
    auto mk = [&](uint64_t targetIdx, bool expectSurvives) {
        CaseSpec c;
        c.alphabet = alphabet;
        c.prefix = prefix;
        c.suffix = ".DAT";
        c.pruneSymbolRuns = prune;
        c.maxBackslashCount = maxBackslash;
        c.pruneUnopenedBrackets = pruneBrackets;
        c.lower = indexToString(start, len, alphabet);
        c.upper = indexToString(end, len, alphabet);
        std::string target = indexToString(targetIdx, len, alphabet);
        c.targetHashA = hashA(prefix + target + c.suffix);
        if (expectSurvives) c.mustBeFound = target;
        c.requireBothPrunedAndChecked = true;
        std::string label = std::string(name) + (expectSurvives ? ", target on the surviving side" : ", target on the pruned side (must NOT be found)");
        return runCase(label, c);
    };
    ok &= mk(survivingLeadingIdx * T + 100, true);
    ok &= mk(survivingLeadingIdx * T, true);              // very first candidate of the surviving value
    ok &= mk(survivingLeadingIdx * T - 100, false);       // pruned: the reference says no match, so runSearch() must report none
    ok &= mk(survivingLeadingIdx * T - 1, false);         // very last candidate of the pruned value
    return ok;
}

static bool scenarioPruneSymbolRuns() {
    // alphabet[0]='!' (forbidden-run character), alphabet[1]='A' (alnum). With a
    // 3-character leading part, leadingIdx 0 is "!!!" (a 3-run: pruned) and
    // leadingIdx 1 is "!!A" (the run resets at 'A': not pruned).
    return pruningScenario("2: pruneSymbolRuns (CPU-side, leading characters only)", "!A &'()+,-.0123456789BCDEFGHIJKLMNOPQRSTUV",
                            "TEST_", /*prune=*/true, 0, /*leadingLen=*/3, 0, 1);
}

static bool scenarioMaxBackslash() {
    // alphabet[0]='\\', alphabet[1]='A', maxBackslashCount=2: leadingIdx 0 is
    // "\\\\\\" (3 backslashes: pruned), leadingIdx 1 is "\\\\A" (2: not pruned).
    return pruningScenario("3: maxBackslashCount (CPU-side, leading characters only)", "\\A &'()+,-.0123456789BCDEFGHIJKLMNOPQRSTUV",
                            "TEST_", false, /*maxBackslash=*/2, /*leadingLen=*/3, 0, 1);
}

static bool scenarioPruneUnopenedBrackets() {
    // alphabet[0]='A', alphabet[1]=')', alphabet[2]=' ', 2 leading characters,
    // no brackets in the prefix: leadingIdx 1 is "A)" (closes a bracket
    // never opened: pruned), leadingIdx 2 is "A " (not pruned).
    bool ok = pruningScenario("3b: pruneUnopenedBrackets (CPU-side, leading characters only)", "A) &'(+,-.!0123456789BCDEFGHIJKLMNOPQRSTUV",
                              "TEST_", false, 0, /*leadingLen=*/2, 1, 2, /*pruneBrackets=*/true);
    // The prefix opens one bracket, which a candidate may close - but only
    // once. alphabet[0]=')', alphabet[1]='A': leadingIdx 0 is "))" (the second
    // closes one never opened: pruned), leadingIdx 1 is ")A" (closes the
    // prefix's: not pruned). If the prefix's bracket weren't counted, both
    // would be pruned.
    ok &= pruningScenario("3c: pruneUnopenedBrackets counts brackets opened in the prefix", ")A &'(+,-.!0123456789BCDEFGHIJKLMNOPQRSTUV",
                          "TEST(", false, 0, /*leadingLen=*/2, 0, 1, /*pruneBrackets=*/true);
    // Round and square brackets are counted separately: the prefix's '['
    // lets a candidate close it with ']' but not with ')'.
    // alphabet[0]=')', alphabet[1]=']': leadingIdx 0 is ")" (pruned),
    // leadingIdx 1 is "]" (not pruned).
    ok &= pruningScenario("3d: pruneUnopenedBrackets counts ( ) and [ ] separately", ")]A &'(+,-.0123456789BCDEFGHIJKLMNOPQRSTUV",
                          "TEST[", false, 0, /*leadingLen=*/1, 0, 1, /*pruneBrackets=*/true);
    return ok;
}

static bool scenarioPrefixBackslashes() {
    // The prefix has 3 backslashes - more than maxBackslashCount(1) allows -
    // while the candidates' own leading characters have at most 1. If the
    // prefix's backslashes ever leaked into the count, every candidate here
    // would be wrongly pruned. alphabet[0]='A', alphabet[1]='\\', 2 leading
    // characters: leadingIdx 42 ("\\A", 1 backslash) survives, 43 ("\\\\", 2) is pruned.
    const std::string alphabet = "A\\ &'()+,-.0123456789BCDEFGHIJKLMNOPQRSTUV";
    printf("=== 4: maxBackslashCount excludes the prefix ===\n");
    const int len = g_window + 2;
    const uint64_t T = ipow(alphabet.size(), g_window);
    const std::string prefix = "TEST\\\\\\";
    bool ok = true;
    auto mk = [&](uint64_t start, uint64_t end, uint64_t targetIdx, bool expectSurvives, const char* what) {
        CaseSpec c;
        c.alphabet = alphabet;
        c.prefix = prefix;
        c.suffix = ".DAT";
        c.maxBackslashCount = 1;
        c.lower = indexToString(start, len, alphabet);
        c.upper = indexToString(end, len, alphabet);
        std::string target = indexToString(targetIdx, len, alphabet);
        c.targetHashA = hashA(prefix + target + c.suffix);
        if (expectSurvives) c.mustBeFound = target;
        c.requireBothPrunedAndChecked = true;
        return runCase(std::string("4: ") + what, c);
    };
    ok &= mk(42 * T + T - 200, 43 * T + 200, 42 * T + T - 100, true, "surviving leading value \"\\A\" is found despite the prefix's backslashes");
    ok &= mk(42 * T + T - 200, 43 * T + 200, 43 * T + 100, false, "leading value \"\\\\\" (2 backslashes) is pruned");
    // ... and leadingIdx 0 ("AA", no backslashes at all) is found too.
    {
        CaseSpec c;
        c.alphabet = alphabet;
        c.prefix = prefix;
        c.suffix = ".DAT";
        c.maxBackslashCount = 1;
        c.lower = indexToString(5, len, alphabet);
        c.upper = indexToString(400, len, alphabet);
        std::string target = indexToString(123, len, alphabet);
        c.targetHashA = hashA(prefix + target + c.suffix);
        c.mustBeFound = target;
        c.requireBothPrunedAndChecked = false;
        // maxBackslashCount != 0, but nothing in this range gets pruned - the reference agrees, that's fine
        ok &= runCase("4: leading value \"AA\" (no backslashes at all)", c);
    }
    return ok;
}

static bool scenarioSingleCandidate() {
    printf("=== 5: lower_bound == upper_bound (single-candidate search) ===\n");
    bool ok = true;
    for (int len : {1, 2, g_window, g_window + 1, g_window + 2}) {
        const std::string alphabet = alphabetOfSize(49);
        uint64_t idx = 123456789ULL % ipow(alphabet.size(), std::min(len, 8));
        ok &= runIndexCase("5: single candidate, length " + std::to_string(len), alphabet, "TEST_", ".DAT", len, idx, idx, idx);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Group G: geometry, for every supported alphabet size.
// ---------------------------------------------------------------------------
static bool geometryForAlphabet(int as) {
    const std::string alphabet = alphabetOfSize(as);
    const uint64_t AS = as;
    const int len = g_window + 1; // one leading character
    const uint64_t T = ipow(AS, g_window);
    const uint64_t rows = T / AS;
    const std::string prefix = "TEST_", suffix = ".DAT";
    const uint64_t base = 3 * T;             // leading value 3
    const uint64_t r = rows > 40 ? 17 : 0;   // an arbitrary interior row
    const std::string tag = "G[as=" + std::to_string(as) + "] ";
    bool ok = true;
    auto run = [&](const char* what, uint64_t start, uint64_t end, std::initializer_list<uint64_t> targets) {
        for (uint64_t t : targets)
            ok &= runIndexCase(tag + what + ", target " + std::to_string(t - start) + "/" + std::to_string(end - start),
                                alphabet, prefix, suffix, len, start, end, t);
        ok &= runNoMatchCase(tag + what + ", no match", alphabet, prefix, suffix, len, start, end);
        // The candidates immediately before and after the range must not be searched.
        if (start >= 2)
            for (uint64_t o : {start - 1, start - 2})
                ok &= runOutsideCase(tag + what + ", just before the range (" + std::to_string(start - o) + ")", alphabet, prefix, suffix, len, start, end, o);
        if (end + 2 < ipow(AS, len))
            for (uint64_t o : {end + 1, end + 2})
                ok &= runOutsideCase(tag + what + ", just after the range (" + std::to_string(o - end) + ")", alphabet, prefix, suffix, len, start, end, o);
    };

    if (rows >= 12) {
        // Entirely inside one row (kBegin > 0 and kEnd < AS at once).
        const uint64_t row = base + r * AS;
        run("inside one row", row + 3, row + AS - 5, {row + 3, row + AS / 2, row + AS - 5});
        // Partial first row, full middle rows, partial last row.
        run("partial first and last rows", row + 7, row + 9 * AS + 11, {row + 7, row + 9, row + 4 * AS + 20, row + 9 * AS + 10, row + 9 * AS + 11});
        // Exactly row-aligned at both ends.
        run("row-aligned", row, row + 8 * AS + AS - 1, {row, row + AS - 1, row + AS, row + 4 * AS + 3, row + 8 * AS + AS - 1});
        // Aligned start, partial end / partial start, aligned end.
        run("aligned start, partial end", row, row + 3 * AS + 13, {row, row + 3 * AS + 13, row + 3 * AS + 14 - 1});
        run("partial start, aligned end", row + 13, row + 3 * AS + AS - 1, {row + 13, row + 3 * AS + AS - 1});
    }
    // Straddling a leading-value boundary (the trailing space wraps).
    run("leading boundary", base + T - 300, base + T + 300, {base + T - 300, base + T - 1, base + T, base + T + 1, base + T + 300});
    // The very first and very last candidates of the whole candidate space.
    run("first candidates", 0, 300, {0, 1, 300});
    const uint64_t last = ipow(AS, len) - 1;
    run("last candidates", last - 300, last, {last - 300, last - 1, last});
    // Straddling batch boundaries (only exist when a leading value spans more than one batch).
    const uint64_t batch = g_backend->batchSize(as);
    for (uint64_t k : {1ULL, 2ULL, 3ULL}) {
        uint64_t b = k * batch;
        if (b + 300 < T) {
            run("launch boundary", base + b - 150, base + b + 150, {base + b - 150, base + b - 1, base + b, base + b + 1, base + b + 150});
            // A boundary that falls mid-row relative to the range start too.
            run("launch boundary, misaligned start", base + b - 2 * AS - 7, base + b + AS + 5, {base + b - 2 * AS - 7, base + b - 1, base + b});
        }
    }
    // A whole leading value's trailing space, when it's small enough to check exhaustively.
    if (T <= 200000)
        run("entire leading value", base, base + T - 1, {base, base + T / 2, base + T - 1});
    return ok;
}

static bool scenarioGeometry() {
    bool ok = true;
    for (int as : kAlphabetSizes) {
        printf("=== G: geometry, alphabet size %d ===\n", as);
        ok &= geometryForAlphabet(as);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Group K: every last-character position, every alphabet size.
// ---------------------------------------------------------------------------
static bool scenarioEveryLastCharacter() {
    bool ok = true;
    for (int as : kAlphabetSizes) {
        printf("=== K: every last-character position, alphabet size %d ===\n", as);
        const std::string alphabet = alphabetOfSize(as);
        const uint64_t AS = as;
        const int len = g_window + 1;
        const uint64_t T = ipow(AS, g_window);
        const uint64_t rows = T / AS;
        for (uint64_t k = 0; k < AS; ++k) {
            uint64_t row = (k * 7919 + 13) % rows;
            uint64_t idx = 2 * T + row * AS + k;
            const std::string tag = "K[as=" + std::to_string(as) + "] k=" + std::to_string(k);
            ok &= runIndexCase(tag, alphabet, "TEST_", ".DAT", len, idx, idx, idx);
            // Its neighbours (the same row's k-1/k+1, or the adjacent row's edge) are outside the range.
            ok &= runOutsideCase(tag + ", neighbour below", alphabet, "TEST_", ".DAT", len, idx, idx, idx - 1);
            ok &= runOutsideCase(tag + ", neighbour above", alphabet, "TEST_", ".DAT", len, idx, idx, idx + 1);
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Group P: prefix/suffix lengths and byte values.
// ---------------------------------------------------------------------------
static std::string patternString(size_t n, const char* pool, size_t poolLen) {
    std::string s;
    for (size_t i = 0; i < n; ++i) s += pool[(i * 5 + 3) % poolLen];
    return s;
}

static bool scenarioPrefixSuffixLengths() {
    printf("=== P: prefix/suffix lengths and byte values ===\n");
    bool ok = true;
    const std::string alphabet = alphabetOfSize(49);
    const int len = g_window + 1;
    const uint64_t T = ipow(alphabet.size(), g_window);
    const uint64_t start = 2 * T + 5000, end = 2 * T + 5000 + 3 * 49 + 5, target = start + 60;
    // Includes bytes >= 0x80: the hash's per-character table index must be an
    // unsigned char on both sides or it would read a negative index.
    static const char kPool[] = {'a', 'B', '.', '_', '\\', '-', (char) 0xE9, (char) 0x80, (char) 0xFF, '7', ' ', (char) 0xA0};
    std::vector<size_t> suffixLens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 33, 60, 63};
    for (size_t sl : suffixLens) {
        std::string suffix = patternString(sl, kPool, sizeof(kPool));
        ok &= runIndexCase("P: suffix length " + std::to_string(sl), alphabet, "TEST_", suffix, len, start, end, target);
    }
    for (size_t pl : std::vector<size_t>{0, 1, 2, 20, 52}) {
        std::string prefix = patternString(pl, kPool, sizeof(kPool));
        ok &= runIndexCase("P: prefix length " + std::to_string(pl), alphabet, prefix, ".wav", len, start, end, target);
    }
    // Combined extremes: long prefix and long suffix together (kept under MAX_FILENAME_LEN).
    // (52 + 59 + MAX_CANDIDATE_LEN(16) = 127 < MAX_FILENAME_LEN; 60 would be rejected - see the error cases below)
    ok &= runIndexCase("P: prefix 52 + suffix 59", alphabet, patternString(52, kPool, sizeof(kPool)), patternString(59, kPool, sizeof(kPool)),
                        len, start, end, target);
    // With hashB too, so foundFilename is built with the (long) prefix and suffix.
    ok &= runIndexCase("P: prefix 30 + suffix 9, hashB too", alphabet, patternString(30, kPool, sizeof(kPool)), patternString(9, kPool, sizeof(kPool)),
                        len, start, end, target, /*alsoHashB=*/true);

    // Limits: too-long prefix/suffix are rejected up front with a clear error, not silently truncated.
    auto expectError = [&](const char* label, const std::string& prefix, const std::string& suffix, const std::string& alphabetIn) {
        ++g_cases;
        SearchRequest req;
        req.alphabet = alphabetIn;
        req.prefix = prefix;
        req.suffix = suffix;
        req.startCandidate = req.lowerBound = std::string(3, alphabetIn[0]);
        req.upperBound = std::string(3, alphabetIn[1]);
        req.targetHashA = 1;
        req.targetHashB = 2;
        OutputCapture capture;
        capture.start();
        SearchResult r = runSearch(*g_backend, req);
        capture.stop();
        if (r.ok || r.found) {
            fprintf(stderr, "FAILED: %s: expected runSearch() to reject this request\n", label);
            ++g_failures;
            return false;
        }
        return true;
    };
    ok &= expectError("P: suffix of 64 characters", "TEST_", std::string(64, 'x'), alphabet);
    ok &= expectError("P: prefix 52 + suffix 60 (would overflow MAX_FILENAME_LEN)", std::string(52, 'x'), std::string(60, 'y'), alphabet);
    ok &= expectError("P: prefix of 53 characters", std::string(53, 'x'), ".DAT", alphabet);
    std::vector<int> supportedSizes = g_backend->supportedAlphabetSizes();
    if (!supportedSizes.empty() && std::find(supportedSizes.begin(), supportedSizes.end(), 39) == supportedSizes.end())
        ok &= expectError("P: alphabet of an unsupported size (39)", "TEST_", ".DAT", alphabetOfSize(42).substr(0, 39));
    {
        // An empty start candidate means "from the beginning": the shortest
        // candidates, one character long - here a bounded search, so only those.
        ++g_cases;
        const std::string target(1, alphabet[12]);
        SearchRequest req;
        req.alphabet = alphabet;
        req.prefix = "TEST_";
        req.suffix = ".DAT";
        req.lowerBound = std::string(1, alphabet[5]);
        req.upperBound = std::string(1, alphabet[20]);
        req.targetHashA = hashA(req.prefix + target + req.suffix);
        req.targetHashB = hashB(req.prefix + target + req.suffix);
        OutputCapture capture;
        capture.start();
        SearchResult r = runSearch(*g_backend, req); // startCandidate empty
        capture.stop();
        if (!r.ok || !r.found || r.filename != req.prefix + target + req.suffix) {
            fprintf(stderr, "FAILED: P: an empty start candidate must search from the shortest candidates (ok=%d found=%d '%s')\n", r.ok,
                    r.found, r.filename.c_str());
            ++g_failures;
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Group F: the both-hashes-match path.
// ---------------------------------------------------------------------------
static bool scenarioFound() {
    printf("=== F: both-hashes-match path ===\n");
    bool ok = true;
    const std::string alphabet = alphabetOfSize(49);
    const int len = g_window + 1;
    const uint64_t T = ipow(alphabet.size(), g_window);
    const std::string prefix = "REZ\\", suffix = ".WAV";
    const uint64_t start = 4 * T + 1000;

    // found, in the middle of the range / at its very last candidate / at its very first.
    ok &= runIndexCase("F: found mid-range", alphabet, prefix, suffix, len, start, start + 5000, start + 2500, true);
    ok &= runIndexCase("F: found in the very last candidate of the range", alphabet, prefix, suffix, len, start, start + 5000, start + 5000, true);
    ok &= runIndexCase("F: found in the very first candidate of the range", alphabet, prefix, suffix, len, start, start + 5000, start, true);
    ok &= runIndexCase("F: found across a leading boundary", alphabet, prefix, suffix, len, 5 * T - 100, 5 * T + 100, 5 * T + 37, true);

    // hashA matches but hashB doesn't: reported as a partial match, but not "found".
    {
        CaseSpec c;
        c.alphabet = alphabet; c.prefix = prefix; c.suffix = suffix;
        c.lower = indexToString(start, len, alphabet);
        c.upper = indexToString(start + 5000, len, alphabet);
        std::string target = indexToString(start + 777, len, alphabet);
        c.targetHashA = hashA(prefix + target + suffix);
        c.targetHashB = hashB(prefix + target + suffix) ^ 1;
        c.mustBeFound = target;
        ok &= runCase("F: hashA-only match is not 'found'", c);
    }

    // No stale state between calls: a search that found something, then one that can't.
    {
        ok &= runIndexCase("F: (setup) a search that finds its target", alphabet, prefix, suffix, len, start, start + 100, start + 50, true);
        ok &= runNoMatchCase("F: the *next* runSearch() in the same process reports nothing (no stale found flag)", alphabet, prefix, suffix, len, start, start + 100);
        ok &= runIndexCase("F: ... and one after that finds its own target again", alphabet, prefix, suffix, len, start, start + 100, start + 51, true);
    }

    // A genuine hashA collision: two different candidates with the same hashA
    // (birthday search over a contiguous block of the candidate space - with
    // ~2^32 hash values, a few hundred thousand candidates contain several
    // such pairs). Both must be reported; hashB then decides which one is "found".
    {
        std::unordered_map<uint32_t, uint64_t> seen;
        uint64_t collA = 0, collB = 0;
        uint32_t collHash = 0;
        bool have = false;
        // Long enough that the scanned block never wraps around the candidate space
        // (with a small window, `len` alone could be too short for 3M distinct candidates).
        const int collLen = std::max(len, 6);
        const uint64_t scanStart = 6 * T;
        for (uint64_t i = 0; i < 3'000'000 && !have; ++i) {
            uint32_t h = hashA(prefix + indexToString(scanStart + i, collLen, alphabet) + suffix);
            auto it = seen.find(h);
            if (it != seen.end()) {
                collA = it->second; collB = scanStart + i; collHash = h; have = true;
            } else {
                seen.emplace(h, scanStart + i);
            }
        }
        if (!have) {
            fprintf(stderr, "TEST BUG: no hashA collision found in the scanned block\n");
            ++g_failures;
            return false;
        }
        std::string c1 = indexToString(collA, collLen, alphabet), c2 = indexToString(collB, collLen, alphabet);
        for (int which = 0; which < 2; ++which) {
            const std::string& winner = which == 0 ? c1 : c2;
            const std::string& loser = which == 0 ? c2 : c1;
            if (hashB(prefix + winner + suffix) == hashB(prefix + loser + suffix)) continue; // (astronomically unlikely)
            CaseSpec c;
            c.alphabet = alphabet; c.prefix = prefix; c.suffix = suffix;
            c.lower = c1; c.upper = c2;
            c.targetHashA = collHash;
            c.targetHashB = hashB(prefix + winner + suffix);
            c.mustBeFound = winner;
            c.expectFound = winner;
            ok &= runCase(std::string("F: hashA collision pair, hashB selects ") + (which == 0 ? "the first" : "the second"), c);
        }
        // Both hashB values wrong: two hashA hits, no "found".
        CaseSpec c;
        c.alphabet = alphabet; c.prefix = prefix; c.suffix = suffix;
        c.lower = c1; c.upper = c2;
        c.targetHashA = collHash;
        c.mustBeFound = c1;
        ok &= runCase("F: hashA collision pair, neither matches hashB", c);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Group A: abort.
// ---------------------------------------------------------------------------
static bool scenarioAbort() {
    printf("=== A: abort ===\n");
    ++g_cases;
    const std::string alphabet = alphabetOfSize(49);
    const int len = g_window + 1;
    SearchRequest req;
    req.alphabet = alphabet;
    req.prefix = "TEST_";
    req.suffix = ".DAT";
    req.startCandidate = req.lowerBound = indexToString(1000, len, alphabet);
    req.upperBound = indexToString(1000 + 50000, len, alphabet);
    std::string target = indexToString(1500, len, alphabet);
    req.targetHashA = hashA(req.prefix + target + req.suffix);
    req.targetHashB = hashB(req.prefix + target + req.suffix);
    std::atomic<bool> abort{true};
    int reports = 0;
    OutputCapture capture;
    capture.start();
    SearchResult r = runSearch(*g_backend, req, &abort, [&](const std::string&) { ++reports; });
    capture.stop();
    if (!r.ok || !r.aborted || r.found || reports != 0) {
        fprintf(stderr, "FAILED: A: a pre-set abort flag must stop the search before any batch runs (ok=%d aborted=%d found=%d reports=%d)\n",
                r.ok, r.aborted, r.found, reports);
        ++g_failures;
        return false;
    }
    // ... and the very next search, with no abort flag, is completely unaffected.
    return runIndexCase("A: search after an aborted one", alphabet, "TEST_", ".DAT", len, 1000, 51000, 1500, true);
}

// ---------------------------------------------------------------------------
// Group Z: fuzzer.
// ---------------------------------------------------------------------------
static bool fuzz(int iterations, uint64_t seed) {
    printf("=== Z: fuzz, %d iterations, seed %llu ===\n", iterations, (unsigned long long) seed);
    std::mt19937_64 rng(seed);
    auto uni = [&](uint64_t n) { return (uint64_t) (rng() % n); };
    bool ok = true;
    int failuresBefore = g_failures;

    for (int it = 0; it < iterations; ++it) {
        CaseSpec c;
        const int as = kAlphabetSizes[uni(std::size(kAlphabetSizes))];
        const bool highBytes = uni(2) == 0;
        std::vector<char> pool;
        for (int b = 0x20; b <= 0x7E; ++b) pool.push_back((char) b);
        if (highBytes)
            for (int b = 0x80; b <= 0xFF; ++b) pool.push_back((char) b);
        std::shuffle(pool.begin(), pool.end(), rng);
        c.alphabet.assign(pool.begin(), pool.begin() + as);

        static const int prefixLens[] = {0, 1, 2, 5, 12, 30, 52};
        static const int suffixLens[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 25, 60};
        int pl = prefixLens[uni(7)];
        int sl = suffixLens[uni(13)];
        // runSearch reserves room after the prefix for as many leading
        // characters as fit a uint64_t index - 11 for sizes 41-50, 12 for 40, 13 for 29-30.
        int maxSafeIndexLen = 0;
        for (uint64_t product = 1; maxSafeIndexLen < MAX_CANDIDATE_LEN && product <= UINT64_MAX / as; product *= as) maxSafeIndexLen++;
        pl = std::min(pl, kMaxPrefixSize - 1 - maxSafeIndexLen);
        if (pl + sl + MAX_CANDIDATE_LEN >= MAX_FILENAME_LEN) sl = MAX_FILENAME_LEN - MAX_CANDIDATE_LEN - 1 - pl;
        auto randomBytes = [&](int n) {
            std::string s;
            for (int i = 0; i < n; ++i) s += (char) (1 + uni(255));
            return s;
        };
        c.prefix = randomBytes(pl);
        c.suffix = randomBytes(sl);

        int len = uni(10) < 7 ? 1 + (int) uni(g_window + 3) : 6 + (int) uni(11); // mostly short, sometimes up to 16
        std::string lower(len, c.alphabet[0]);
        for (int i = 0; i < len; ++i) lower[i] = c.alphabet[uni(as)];
        lower[0] = c.alphabet[uni(as - 1)]; // leave head-room above the first character
        static const uint64_t sizes[] = {1, 2, 3, 41, 42, 43, 49, 50, 98, 2000, 15000};
        uint64_t n = 1 + uni(sizes[uni(11)]);
        std::string upper = lower;
        uint64_t steps = advanceBy(upper, c.alphabet, n - 1);
        c.lower = lower;
        c.upper = upper;

        if (!highBytes && uni(3) == 0) c.pruneSymbolRuns = true;
        if (!highBytes && uni(4) == 0) c.maxBackslashCount = 1 + (int) uni(3);
        if (uni(3) == 0) c.pruneUnopenedBrackets = true;

        // Plant a target at a random position in the range (or none).
        if (uni(10) < 8) {
            std::string planted = lower;
            advanceBy(planted, c.alphabet, uni(steps + 1));
            c.targetHashA = hashA(c.prefix + planted + c.suffix);
            int leadingLen = len - trailingLenFor(len, as);
            bool survives = !isPruned(planted, leadingLen, c.pruneSymbolRuns, c.maxBackslashCount, c.pruneUnopenedBrackets, c.prefix);
            if (survives) c.mustBeFound = planted;
            if (uni(2) == 0) {
                c.targetHashB = hashB(c.prefix + planted + c.suffix);
                if (survives) c.expectFound = planted;
            }
        }
        char label[200];
        snprintf(label, sizeof(label), "Z[seed %llu, iteration %d] as=%d len=%d n=%llu prefix=%d suffix=%d prune=%d bs=%d brackets=%d high=%d",
                 (unsigned long long) seed, it, as, len, (unsigned long long) n, pl, sl, c.pruneSymbolRuns, c.maxBackslashCount,
                 c.pruneUnopenedBrackets, highBytes);
        // An entirely-pruned range is a legitimate outcome here; skip the reference's "nothing survived" TEST BUG check for those.
        {
            int leadingLen = len - trailingLenFor(len, as);
            std::string cand = lower;
            bool anySurvives = false;
            for (uint64_t i = 0; i <= steps && !anySurvives; ++i) {
                if (!isPruned(cand, leadingLen, c.pruneSymbolRuns, c.maxBackslashCount, c.pruneUnopenedBrackets, c.prefix)) anySurvives = true;
                stepCandidate(cand, c.alphabet);
            }
            if (!anySurvives) continue;
        }
        ok &= runCase(label, c);
    }
    printf("    fuzz: %s\n", g_failures == failuresBefore ? "all iterations passed" : "FAILURES above");
    return ok;
}

int main(int argc, char** argv) {
    // Default: the first backend that can run here (see backends/backends.h).
    std::string backendName;
    if (argc == 3 && std::string(argv[1]) == "--backend") {
        backendName = argv[2];
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [--backend <name>]\n", argv[0]);
        return 1;
    }
    std::string error;
    bool selfTestFailed = false;
    g_backend = createBackend(backendName, error, &selfTestFailed);
    if (!g_backend) {
        if (selfTestFailed) {
            fprintf(stderr, "FAILED: %s\n", error.c_str());
            return 1;
        }
        // 77: "skipped" to ctest (see SKIP_RETURN_CODE in CMakeLists.txt) - a
        // GPU backend on a machine without that kind of GPU - unless the build
        // requires its GPUs (NAMEBREAK_REQUIRE_GPU), where it's a failure.
        fprintf(stderr, "%s - exit code 77: skipped, or failed where NAMEBREAK_REQUIRE_GPU is ON\n", error.c_str());
        return 77;
    }
    g_window = g_backend->windowChars();
    std::vector<int> supportedSizes = g_backend->supportedAlphabetSizes();
    for (int as : kAlphabetSizes) {
        if (!supportedSizes.empty() && std::find(supportedSizes.begin(), supportedSizes.end(), as) == supportedSizes.end()) {
            fprintf(stderr, "TEST BUG: the %s backend doesn't support alphabet size %d\n", g_backend->name(), as);
            return 1;
        }
    }

    prepareCryptTable(g_cryptTable);
    printf("backend=%s, window=%d, batch size=%llu (49-character alphabet), MAX_MATCHES=%d\n", g_backend->name(), g_window,
           (unsigned long long) g_backend->batchSize(49), MAX_MATCHES);

    auto start = std::chrono::steady_clock::now();
    bool allPassed = true;
    allPassed &= scenarioSplit();
    allPassed &= scenarioPruneSymbolRuns();
    allPassed &= scenarioMaxBackslash();
    allPassed &= scenarioPruneUnopenedBrackets();
    allPassed &= scenarioPrefixBackslashes();
    allPassed &= scenarioSingleCandidate();
    allPassed &= scenarioGeometry();
    allPassed &= scenarioEveryLastCharacter();
    allPassed &= scenarioPrefixSuffixLengths();
    allPassed &= scenarioFound();
    allPassed &= scenarioAbort();
    allPassed &= fuzz(300, 20260921);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    printf("\n%d case(s) run in %.1fs, %d failure(s)\n", g_cases, seconds, g_failures);
    if (!allPassed || g_failures) {
        fprintf(stderr, "One or more cases FAILED - see above.\n");
        return 1;
    }
    printf("ALL SCENARIOS PASSED\n");
    return 0;
}
