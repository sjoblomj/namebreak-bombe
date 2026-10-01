// Known-answer test: every filename of a real StarCraft listfile
// (tests/data/sc-listfile.txt) must be found by the real runSearch() on the
// backend under test, given its two hashes.
//
// A whole filename is far longer than any candidate (MAX_CANDIDATE_LEN), so
// each name is turned into a small search whose answer it is: it is split
// into prefix + candidate + suffix at a place picked by an RNG seeded from its
// line number, and a range of candidates of that length around the real one
// is searched for the name's hashA and hashB. The answer sits at a seeded
// place in the range - its first or last candidate, the only one, or
// anywhere - and the ranges are between one candidate and a few leading
// values long. Every name must come back as found, with exactly its own
// filename, and every other hashA hit reported must really be one.
//
// Every tenth name (lines 1, 11, 21, ...) is also searched twice more: with
// the answer just outside the range (nothing may be found) and with a wrong
// hashB (the name must be reported as a hashA hit, but not as found).
//
// Unlike the other tests, the hashes come from the test's own copy of the MPQ
// hash, crypt table included, not from prepareCryptTable - so a bug there,
// which every other test shares with the code under test, shows here as
// names that are never found. The names are uppercased first: MPQ hashes a
// filename in upper case, and runSearch() hashes what it is given.
//
// Pruning is off: this is about whether the search finds what it should, not
// about whether the pruning rules would ever skip a real name (one of these
// names, STAREDIT\WAV\STEAL!  THE!  BEACON!!!!.WAV, has a run of symbols
// prune_symbol_runs skips).
//
// Arguments: --backend <name> (default: the first that can run here),
// --listfile <path>, --every <n> (only every n-th name, for the slow
// `reference` backend) and --only <line> (just that line, to rerun a
// failure). Calls the real runSearch(), which appends to matches.txt in the
// current directory - ctest runs this in its own directory under
// build/testrun/, and the file is removed after every case.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "backends/backends.h"
#include "engine/candidate.h"
#include "engine/limits.h"
#include "engine/search.h"

// The production alphabet, with the backslash, which the names have
// between directories: every character of an uppercased name is in it.
static const std::string kAlphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_\\";

static std::unique_ptr<SearchBackend> g_backend;
static int g_cases = 0;
static int g_failures = 0;

// ---------------------------------------------------------------------------
// Output capture (POSIX only - elsewhere runSearch() just prints to the
// console and the "no WARNING" check is skipped).
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
// The test's own MPQ hash, crypt table and all.
// ---------------------------------------------------------------------------
static uint32_t g_cryptTable[0x500];

static void buildCryptTable() {
    uint32_t seed = 0x00100001;
    for (int index1 = 0; index1 < 0x100; ++index1) {
        for (int i = 0, index2 = index1; i < 5; ++i, index2 += 0x100) {
            seed = (seed * 125 + 3) % 0x2AAAAB;
            uint32_t high = (seed & 0xFFFF) << 16;
            seed = (seed * 125 + 3) % 0x2AAAAB;
            g_cryptTable[index2] = high | (seed & 0xFFFF);
        }
    }
}

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

// ---------------------------------------------------------------------------
// Turning a name into a search.
// ---------------------------------------------------------------------------

// The longest candidate tail whose index fits in 64 bits (as in search.cpp).
static int maxSafeIndexLen() {
    uint64_t product = 1;
    int n = 0;
    while (n < MAX_CANDIDATE_LEN && product <= UINT64_MAX / kAlphabet.size()) {
        product *= kAlphabet.size();
        n++;
    }
    return n;
}

static uint64_t ipow(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i) r *= base;
    return r;
}

struct Split {
    std::string prefix, candidate, suffix;
};

// Where `name` is split, picked by `rng`. A third of the time the candidate
// ends where the extension starts, as in a real search (prefix: the
// directory and the start of the name, suffix: the extension); otherwise
// anywhere. The prefix is kept short enough for the search's buffers once the
// leading characters are added to it (see runSearch).
static Split splitName(const std::string& name, std::mt19937_64& rng) {
    const int n = (int) name.size();
    const int maxPrefix = kMaxPrefixSize - 1 - maxSafeIndexLen();
    auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    int start = -1, end = -1;
    size_t dot = name.rfind('.');
    if (rng() % 3 == 0 && dot != std::string::npos && dot > 0) {
        end = (int) dot;
        int lo = std::max(0, end - MAX_CANDIDATE_LEN), hi = std::min(end - 1, maxPrefix);
        if (lo <= hi) start = pick(lo, hi);
    }
    if (start < 0) {
        start = pick(0, std::min(n - 1, maxPrefix));
        end = start + pick(1, std::min(MAX_CANDIDATE_LEN, n - start));
    }
    return {name.substr(0, start), name.substr(start, end - start), name.substr(end)};
}

// A range of candidates of `candidate`'s length: only its last `tailLen`
// characters vary, between tail indices `lo` and `hi`.
struct Range {
    std::string lower, upper;
};
static Range makeRange(const std::string& candidate, int tailLen, uint64_t lo, uint64_t hi) {
    std::string head = candidate.substr(0, candidate.size() - tailLen);
    return {head + indexToString(lo, tailLen, kAlphabet), head + indexToString(hi, tailLen, kAlphabet)};
}

// ---------------------------------------------------------------------------
// One search, checked.
// ---------------------------------------------------------------------------

// Searches [range.lower, range.upper] for the given hashes. If `expectFound`,
// `name` must be found; otherwise nothing may be. If `expectHit`, `name` must
// be among the hashA hits reported; otherwise it must not be.
static bool runCase(int line, const char* what, const std::string& name, const Split& split, const Range& range, uint32_t targetHashA,
                    uint32_t targetHashB, bool expectFound, bool expectHit) {
    ++g_cases;
    SearchRequest req;
    req.alphabet = kAlphabet;
    req.prefix = split.prefix;
    req.suffix = split.suffix;
    req.startCandidate = range.lower;
    req.lowerBound = range.lower;
    req.upperBound = range.upper;
    req.targetHashA = targetHashA;
    req.targetHashB = targetHashB;

    std::vector<std::string> hits;
    OutputCapture capture;
    capture.start();
    SearchResult result = runSearch(*g_backend, req, nullptr, [&](const std::string& filename) { hits.push_back(filename); });
    std::string log = capture.stop();
    std::remove(req.outputFilePath.c_str());

    bool ok = true;
    auto fail = [&](const std::string& msg) {
        if (ok) fprintf(stderr, "FAILED: line %d (%s): '%s'\n", line, what, name.c_str());
        ok = false;
        fprintf(stderr, "  %s\n", msg.c_str());
    };

    if (!result.ok) {
        fail("runSearch() failed: " + result.error);
    } else {
        if (expectFound && !result.found)
            fail("not found");
        else if (expectFound && result.filename != name)
            fail("found '" + result.filename + "' instead");
        else if (!expectFound && result.found)
            fail("found '" + result.filename + "', but it must not be");

        bool nameHit = false;
        for (const std::string& hit : hits) {
            if (hit == name) nameHit = true;
            if (hashA(hit) != targetHashA) fail("reported hashA hit '" + hit + "' doesn't match hashA");
        }
        if (expectHit && !nameHit) fail("not among the hashA hits reported");
        if (!expectHit && nameHit) fail("reported as a hashA hit, but it is outside the range");
        std::vector<std::string> sorted = hits;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) fail("the same hit was reported more than once");
    }
    if (size_t at = log.find("WARNING"); at != std::string::npos)
        fail("a WARNING was printed (the backend disagrees with its own check of a hit): " + log.substr(at, 200));

    if (!ok) {
        fprintf(stderr, "  prefix '%s', suffix '%s', range '%s'..'%s', backend %s\n", split.prefix.c_str(), split.suffix.c_str(),
                range.lower.c_str(), range.upper.c_str(), g_backend->name());
        ++g_failures;
    }
    return ok;
}

// Every search for the name on `line`.
static bool testName(int line, const std::string& name) {
    for (char c : name) {
        if (kAlphabet.find(c) == std::string::npos) {
            fprintf(stderr, "TEST BUG: line %d, '%s', has a character outside the alphabet: '%c'\n", line, name.c_str(), c);
            ++g_failures;
            return false;
        }
    }

    std::mt19937_64 rng(0x5C11571Full ^ (uint64_t) line * 0x9E3779B97F4A7C15ull);
    const Split split = splitName(name, rng);
    const uint32_t targetHashA = hashA(name), targetHashB = hashB(name);

    // Only the last tailLen characters of the candidate vary within its range.
    const int tailLen = std::min((int) split.candidate.size(), maxSafeIndexLen());
    const uint64_t space = ipow(kAlphabet.size(), tailLen);
    uint64_t t = 0;
    std::string error;
    if (!stringToIndex(split.candidate.substr(split.candidate.size() - tailLen), kAlphabet, t, error)) {
        fprintf(stderr, "TEST BUG: line %d: %s\n", line, error.c_str());
        ++g_failures;
        return false;
    }

    // How many candidates before and after the answer: up to a few leading
    // values' worth, log-uniformly, so most ranges are short and some cross
    // leading values - or the answer is the range's first, last or only one.
    auto randomLength = [&]() {
        double maxLog = std::log(3.0 * (double) ipow(kAlphabet.size(), g_backend->windowChars()));
        return (uint64_t) std::exp(std::uniform_real_distribution<double>(0, maxLog)(rng));
    };
    uint64_t before = randomLength(), after = randomLength();
    switch (rng() % 8) {
        case 0: before = after = 0; break;
        case 1: before = 0; break;
        case 2: after = 0; break;
        default: break;
    }
    before = std::min(before, t);
    after = std::min(after, space - 1 - t);

    bool ok = runCase(line, "found", name, split, makeRange(split.candidate, tailLen, t - before, t + after), targetHashA, targetHashB,
                      true, true);

    if (line % 10 == 1) {
        // The answer just outside the range, on whichever side there is room.
        if (t + 1 < space)
            ok &= runCase(line, "just outside", name, split, makeRange(split.candidate, tailLen, t + 1, t + 1 + std::min(after, space - 2 - t)),
                          targetHashA, targetHashB, false, false);
        else
            ok &= runCase(line, "just outside", name, split, makeRange(split.candidate, tailLen, t - 1 - std::min(before, t - 1), t - 1),
                          targetHashA, targetHashB, false, false);
        ok &= runCase(line, "wrong hashB", name, split, makeRange(split.candidate, tailLen, t - before, t + after), targetHashA,
                      targetHashB ^ 1, false, true);
    }
    return ok;
}

int main(int argc, char** argv) {
    std::string backendName, listfile = "sc-listfile.txt";
    int every = 1, only = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 < argc && arg == "--backend") backendName = argv[++i];
        else if (i + 1 < argc && arg == "--listfile") listfile = argv[++i];
        else if (i + 1 < argc && arg == "--every") every = std::max(1, std::atoi(argv[++i]));
        else if (i + 1 < argc && arg == "--only") only = std::atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--backend <name>] [--listfile <path>] [--every <n>] [--only <line>]\n", argv[0]);
            return 1;
        }
    }

    std::ifstream in(listfile, std::ios::binary);
    if (!in) {
        fprintf(stderr, "Can't read the listfile '%s'\n", listfile.c_str());
        return 1;
    }
    std::vector<std::string> names;
    for (std::string name; std::getline(in, name);) {
        if (!name.empty() && name.back() == '\r') name.pop_back();
        for (char& c : name)
            if (c >= 'a' && c <= 'z') c = (char) (c - 'a' + 'A');
        names.push_back(name);
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
    buildCryptTable();
    printf("backend=%s, window=%d, %zu names in %s\n", g_backend->name(), g_backend->windowChars(), names.size(), listfile.c_str());

    auto start = std::chrono::steady_clock::now();
    int tested = 0, failedNames = 0;
    for (int line = 1; line <= (int) names.size(); ++line) {
        if (only ? line != only : (line - 1) % every != 0 || names[line - 1].empty()) continue;
        ++tested;
        if (!testName(line, names[line - 1])) ++failedNames;
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    printf("%d name(s) in %d search(es), %.1fs: %d name(s) failed\n", tested, g_cases, seconds, failedNames);
    if (tested == 0) {
        fprintf(stderr, "TEST BUG: no names were tested\n");
        return 1;
    }
    if (g_failures) {
        fprintf(stderr, "One or more names FAILED - see above.\n");
        return 1;
    }
    printf("ALL NAMES FOUND\n");
    return 0;
}
