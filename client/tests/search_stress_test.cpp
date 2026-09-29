// Dense-hit differential test of runSearch() (engine/search.cpp) and a backend.
//
// With a real 32-bit target, a hashA hit happens about once in 4 billion
// candidates, so the other tests only ever see the handful of matches their
// cases plant - and a backend that drops, say, every hit whose last character
// is past the 32nd, or that sits in a launch's last row, would only fail the
// ones that happen to plant exactly that. This test is built with
// -DNAMEBREAK_HASHA_MATCH_BITS=12 (see engine/hash_match.h and
// CMakeLists.txt): a candidate is a "hit" when the low 12 bits of its hashA
// match, about one in 4096 - hundreds of thousands of them per run, in every
// position of every row, at the edges of ranges, rows, launches and leading
// values, through every path that finds and reports hits (the lookup filter's
// included - it filters on the low bits, so a weaker match changes nothing
// about how it works, only how often something gets through it).
//
// Every case runs the real runSearch() on a random range - random alphabet
// (size, characters, order, bytes >= 0x80), prefix, suffix and candidate
// length - and compares the complete set of hits it reports with an
// independent brute force that hashes every candidate of the range from
// scratch: any hit missing, any extra, any reported twice, or any WARNING
// from a backend's own cross-checks, fails it. On top of the random hits,
// every case plants one at an edge - the range's first or last candidate,
// either side of a leading-value or batch boundary, a row's first or last
// character, or just outside the range, where it must not be reported -
// since a random hit lands on any one such candidate only once in 4096
// cases, and edges are where a kernel is most likely to be wrong (mutation
// tests that cut every launch's first or last row short by one, or that
// ignored where a range starts and ends, got past this test until it did).
//
// Calls the real runSearch(), which appends every hit to a matches file in
// the current directory - ctest runs this in its own directory under
// build/testrun/, and the file is removed after every case.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "backends/backends.h"
#include "engine/limits.h"
#include "engine/mpq_hash.h"
#include "engine/search.h"

static_assert(NAMEBREAK_HASHA_MATCH_BITS <= 16, "this test must be built with a weak hashA match (-DNAMEBREAK_HASHA_MATCH_BITS=12, see CMakeLists.txt)");
// The test's own copy of what a hit is, rather than the code under test's.
static const uint32_t kMatchMask = 0xFFFFFFFFu >> (32 - NAMEBREAK_HASHA_MATCH_BITS);
static const char* kMatchesFile = "stress_matches.txt";

static std::unique_ptr<SearchBackend> g_backend;
static uint32_t g_cryptTable[0x500];

// Output capture (POSIX only - elsewhere runSearch() just prints to the
// console and the "no WARNING" check is skipped).
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

static void hashStep(uint32_t& seed1, uint32_t& seed2, unsigned char ch, int offset) {
    seed1 = g_cryptTable[offset + ch] ^ (seed1 + seed2);
    seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
}

static uint32_t hashFromScratch(const std::string& s, int offset) {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : s)
        hashStep(seed1, seed2, ch, offset);
    return seed1;
}

// The candidate at `index` in the enumeration order (last character fastest).
static std::string candidateAt(uint64_t index, int len, const std::string& alphabet) {
    std::string s(len, alphabet[0]);
    for (int i = len - 1; i >= 0; --i) {
        s[i] = alphabet[index % alphabet.size()];
        index /= alphabet.size();
    }
    return s;
}

struct Case {
    std::string alphabet, prefix, suffix;
    int len;
    uint64_t first, last; // candidate indices, inclusive
    uint32_t targetA, targetB;
    // The candidate the case's targetA was taken from, so it's sure to be a
    // hit: one at an edge where a backend's handling changes (see
    // plantAtEdge) - or just outside the range, where it must not be reported.
    std::string planted;
    bool plantedInside = true;
};

// The brute force: every candidate in [first, last], hashed from scratch
// after the prefix, on every core.
static std::vector<std::string> bruteForce(const Case& c) {
    uint32_t prefix1 = 0x7FED7FED, prefix2 = 0xEEEEEEEE;
    for (unsigned char ch : c.prefix)
        hashStep(prefix1, prefix2, ch, 0x100);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    const uint64_t total = c.last - c.first + 1;
    std::vector<std::vector<std::string>> found(threads);
    auto work = [&](unsigned t) {
        const uint64_t from = c.first + total * t / threads, to = c.first + total * (t + 1) / threads;
        if (from >= to)
            return;
        std::string cand = candidateAt(from, c.len, c.alphabet);
        std::vector<size_t> digit(c.len);
        for (int i = 0; i < c.len; ++i)
            digit[i] = c.alphabet.find(cand[i]);
        for (uint64_t n = from; n < to; ++n) {
            uint32_t seed1 = prefix1, seed2 = prefix2;
            for (unsigned char ch : cand)
                hashStep(seed1, seed2, ch, 0x100);
            for (unsigned char ch : c.suffix)
                hashStep(seed1, seed2, ch, 0x100);
            if ((seed1 & kMatchMask) == (c.targetA & kMatchMask))
                found[t].push_back(c.prefix + cand + c.suffix);
            for (int i = c.len - 1; i >= 0; --i) {
                if (++digit[i] < c.alphabet.size()) {
                    cand[i] = c.alphabet[digit[i]];
                    break;
                }
                digit[i] = 0;
                cand[i] = c.alphabet[0];
            }
        }
    };
    std::vector<std::thread> workers;
    for (unsigned t = 1; t < threads; ++t)
        workers.emplace_back(work, t);
    work(0);
    for (std::thread& w : workers)
        w.join();
    std::vector<std::string> all;
    for (auto& f : found)
        all.insert(all.end(), f.begin(), f.end());
    std::sort(all.begin(), all.end());
    return all;
}

static std::string describe(const Case& c) {
    char buf[512];
    snprintf(buf, sizeof(buf), "alphabet size %zu, prefix length %zu, suffix length %zu, candidate length %d, indices %llu..%llu, target 0x%08X",
             c.alphabet.size(), c.prefix.size(), c.suffix.size(), c.len, (unsigned long long) c.first, (unsigned long long) c.last, c.targetA);
    return buf;
}

// runSearch()'s split of a candidate into leading and trailing characters
// (search.cpp: maxSafeIndexLen / trailingLen): the backend gets one leading
// value's trailing space at a time, cut at multiples of its batch size.
static int trailingLenFor(int len, int alphabetSize, int window) {
    uint64_t product = 1;
    int maxSafeIndexLen = 0;
    while (maxSafeIndexLen < MAX_CANDIDATE_LEN && product <= UINT64_MAX / alphabetSize) {
        product *= alphabetSize;
        maxSafeIndexLen++;
    }
    return std::min(len, std::max(window, len - maxSafeIndexLen));
}

// Picks where to plant a hit, and sets c.targetA to that candidate's hashA.
// Random hits land on any particular candidate once in 4096 cases; these are
// the candidates a backend handles differently, so every case gets one: the
// range's first and last candidates, either side of a boundary between two
// leading values and between two batches, the first and last candidate of a
// row - or, in a quarter of the cases, the candidate just before or just after
// the range, which must *not* be reported.
static void plantAtEdge(Case& c, int window, uint64_t batchSize, uint64_t space, std::mt19937_64& rng) {
    if (rng() % 4 == 0 && (c.first > 0 || c.last + 1 < space)) {
        const bool before = c.first > 0 && (c.last + 1 == space || rng() % 2 == 0);
        c.planted = c.prefix + candidateAt(before ? c.first - 1 : c.last + 1, c.len, c.alphabet) + c.suffix;
        c.plantedInside = false;
        c.targetA = hashFromScratch(c.planted, 0x100);
        return;
    }
    const uint64_t as = c.alphabet.size();
    uint64_t leadingBlock = 1;
    for (int i = 0; i < trailingLenFor(c.len, (int) as, window); ++i)
        leadingBlock *= as;
    std::vector<uint64_t> edges = {c.first, c.last};
    auto addPair = [&](uint64_t boundary) { // the candidates either side of `boundary`, if both are in range
        if (boundary > c.first && boundary <= c.last) {
            edges.push_back(boundary - 1);
            edges.push_back(boundary);
        }
    };
    addPair((c.first / leadingBlock + 1) * leadingBlock);
    const uint64_t leadingStart = c.first / leadingBlock * leadingBlock;
    const uint64_t nextBatch = ((c.first - leadingStart) / batchSize + 1) * batchSize;
    if (nextBatch < leadingBlock)
        addPair(leadingStart + nextBatch);
    const uint64_t somewhere = c.first + rng() % (c.last - c.first + 1);
    const uint64_t rowStart = somewhere - somewhere % as;
    if (rowStart >= c.first)
        edges.push_back(rowStart);
    if (rowStart + as - 1 <= c.last)
        edges.push_back(rowStart + as - 1);
    const uint64_t planted = edges[rng() % edges.size()];
    c.planted = c.prefix + candidateAt(planted, c.len, c.alphabet) + c.suffix;
    c.targetA = hashFromScratch(c.planted, 0x100);
}

// Runs one case; returns the number of hits it verified, or -1 if it failed.
static long runCase(Case c, std::mt19937_64& rng) {
    std::vector<std::string> expected = bruteForce(c);
    if (std::binary_search(expected.begin(), expected.end(), c.planted) != c.plantedInside) {
        fprintf(stderr, "TEST BUG: the brute force %s the hit planted %s the range, '%s' (%s)\n", c.plantedInside ? "doesn't have" : "has",
                c.plantedInside ? "inside" : "outside", c.planted.c_str(), describe(c).c_str());
        return -1;
    }
    // No hit may match hashB too: finding one would end the search early.
    for (bool clash = true; clash;) {
        clash = false;
        for (const std::string& hit : expected)
            clash |= hashFromScratch(hit, 0x200) == c.targetB;
        if (clash)
            c.targetB = (uint32_t) rng();
    }

    SearchRequest req;
    req.alphabet = c.alphabet;
    req.prefix = c.prefix;
    req.suffix = c.suffix;
    req.lowerBound = candidateAt(c.first, c.len, c.alphabet);
    req.upperBound = candidateAt(c.last, c.len, c.alphabet);
    req.startCandidate = req.lowerBound;
    req.targetHashA = c.targetA;
    req.targetHashB = c.targetB;
    req.continuous = false;
    req.outputFilePath = kMatchesFile;

    std::vector<std::string> reported;
    OutputCapture capture;
    capture.start();
    SearchResult result = runSearch(*g_backend, req, nullptr, [&](const std::string& f) { reported.push_back(f); });
    const std::string log = capture.stop();
    std::remove(kMatchesFile);

    bool ok = true;
    auto fail = [&](const std::string& what) {
        if (ok)
            fprintf(stderr, "FAILED: %s\n", describe(c).c_str());
        fprintf(stderr, "  %s\n", what.c_str());
        ok = false;
    };
    if (!result.ok) {
        fail("runSearch() failed: " + result.error);
        return -1;
    }
    if (result.found)
        fail("runSearch() reported a both-hashes match ('" + result.filename + "'), but no candidate matches hashB");
    std::sort(reported.begin(), reported.end());
    if (std::adjacent_find(reported.begin(), reported.end()) != reported.end())
        fail("a hit was reported more than once");
    reported.erase(std::unique(reported.begin(), reported.end()), reported.end());
    std::vector<std::string> missing, extra;
    std::set_difference(expected.begin(), expected.end(), reported.begin(), reported.end(), std::back_inserter(missing));
    std::set_difference(reported.begin(), reported.end(), expected.begin(), expected.end(), std::back_inserter(extra));
    if (!missing.empty())
        fail(std::to_string(missing.size()) + " of " + std::to_string(expected.size()) + " hits MISSING (silently dropped), e.g. '" + missing[0] + "'");
    if (!extra.empty())
        fail(std::to_string(extra.size()) + " EXTRA hits (not in the brute force), e.g. '" + extra[0] + "'");
    if (log.find("WARNING") != std::string::npos)
        fail("a WARNING was printed: " + log.substr(log.find("WARNING"), 200));
    return ok ? (long) expected.size() : -1;
}

static std::string randomBytes(std::mt19937_64& rng, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i)
        s += (char) (1 + rng() % 255); // any byte but NUL
    return s;
}

static uint64_t ipow(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i)
        r *= base;
    return r;
}

int main(int argc, char** argv) {
    std::string backendName;
    uint64_t budget = 300'000'000; // candidates brute-forced, over all cases
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backendName = argv[++i];
        } else if (arg == "--budget" && i + 1 < argc) {
            budget = std::stoull(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--backend <name>] [--budget <candidates>]\n", argv[0]);
            return 1;
        }
    }
    std::string error;
    bool selfTestFailed = false;
    g_backend = createBackend(backendName, error, &selfTestFailed);
    if (!g_backend) {
        if (selfTestFailed) {
            fprintf(stderr, "FAILED: %s\n", error.c_str());
            return 1;
        }
        fprintf(stderr, "%s - skipping\n", error.c_str()); // 77: "skipped" to ctest
        return 77;
    }
    prepareCryptTable(g_cryptTable);
    const int window = g_backend->windowChars();
    const std::vector<int> sizes = g_backend->supportedAlphabetSizes();
    // The builds that cut batches down to a few rows (see CMakeLists.txt)
    // would take minutes over the whole budget: about 100,000 batches' worth
    // at most - and cases small enough that there are still dozens of them.
    budget = std::min<uint64_t>(budget, g_backend->batchSize(MAX_ALPHABET_SIZE) * 100'000);
    const uint64_t maxCaseCount = std::max<uint64_t>(1000, std::min<uint64_t>(4'000'000, budget / 50));
    printf("backend=%s, window=%d, hashA match bits=%d, MAX_MATCHES=%d, budget=%llu candidates\n", g_backend->name(), window,
           NAMEBREAK_HASHA_MATCH_BITS, MAX_MATCHES, (unsigned long long) budget);

    std::mt19937_64 rng(20260929);
    const auto start = std::chrono::steady_clock::now();
    uint64_t candidates = 0;
    long hits = 0;
    int cases = 0, failures = 0;
    while (candidates < budget) {
        Case c;
        const int as = sizes.empty() ? 2 + (int) (rng() % (MAX_ALPHABET_SIZE - 1)) : sizes[rng() % sizes.size()];
        std::vector<char> pool;
        for (int b = 1; b < 256; ++b)
            pool.push_back((char) b);
        std::shuffle(pool.begin(), pool.end(), rng);
        c.alphabet.assign(pool.begin(), pool.begin() + as);
        c.prefix = randomBytes(rng, rng() % 13);
        c.suffix = randomBytes(rng, rng() % 4 == 0 ? 9 + rng() % 32 : rng() % 9);
        c.len = 1 + (int) (rng() % (window + 3));
        const uint64_t space = ipow(as, c.len);
        c.targetA = (uint32_t) rng();
        c.targetB = (uint32_t) rng();

        // Mostly ranges of up to a few million candidates anywhere; some tiny;
        // some straddling the boundary between two leading values (runSearch
        // hands the backend one leading value's trailing space at a time) or
        // touching either end of the whole space.
        const uint64_t maxCount = std::min<uint64_t>(space, maxCaseCount);
        uint64_t count = 1 + rng() % (rng() % 5 == 0 ? std::min<uint64_t>(maxCount, 200) : maxCount);
        const int shape = (int) (rng() % 8);
        const uint64_t leadingBlock = ipow(as, std::min(c.len, window));
        if (shape == 0 && space > leadingBlock) {
            // A multiple of the leading block inside the space: 1 .. space/leadingBlock - 1 blocks in.
            const uint64_t boundary = leadingBlock * (1 + rng() % (space / leadingBlock - 1));
            const uint64_t before = 1 + rng() % std::min<uint64_t>(boundary, 20'000);
            const uint64_t after = rng() % std::min<uint64_t>(space - boundary + 1, 20'000);
            c.first = boundary - before;
            c.last = std::min(space - 1, boundary + after);
        } else if (shape == 1) {
            c.first = 0;
            c.last = count - 1;
        } else if (shape == 2) {
            c.first = space - count;
            c.last = space - 1;
        } else {
            c.first = rng() % (space - count + 1);
            c.last = c.first + count - 1;
        }

        plantAtEdge(c, window, g_backend->batchSize(as), space, rng);
        const long verified = runCase(c, rng);
        ++cases;
        candidates += c.last - c.first + 1;
        if (verified < 0)
            ++failures;
        else
            hits += verified;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("%d case(s), %llu candidates, %ld hits verified, %.1fs, %d failure(s)\n", cases, (unsigned long long) candidates, hits, seconds,
           failures);
    if (failures)
        return 1;
    printf("ALL PASSED\n");
    return 0;
}
