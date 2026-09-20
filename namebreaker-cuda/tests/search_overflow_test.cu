// Correctness test for what runSearch() does when one launch has more hashA
// hits than MAX_MATCHES can record - a situation that can't happen in
// practice with MAX_MATCHES = 1024 (a single 32-bit hashA collision is
// already a ~1-in-4-billion event), so this build shrinks it to 1
// (-DMAX_MATCHES=1, see the Makefile) and uses a real hashA collision pair
// (two different candidates with the same hashA, found by a birthday search
// on the CPU) to overflow it with just two hits.
//
// What must hold - the point being that a both-hashes match must never go
// unchecked just because of how many other hashA hits share its launch:
//  * runSearch() falls back to searching the launch's range in halves, so
//    *every* hit is checked against hashB: the hashB-matching candidate is
//    found whether it is the earlier or the later of the pair,
//  * hits are still reported in enumeration order, exactly once each, and
//    finding a match still ends the search (so a hit after the found one is
//    never reported),
//  * the fallback is announced (a "note:"), nothing is silently dropped, and
//    the kernels never disagree with the reference hashing path,
//  * and it doesn't poison the next runSearch() call (matchCount is reset).
//
// Calls the real runSearch(), which does fopen("matches.txt", "a") relative
// to the current directory - `make test` runs this from build/testrun/.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "engine/search.h"
#include "engine/candidate.h"
#include "engine/mpq_hash.h"
#include "engine/constants.h"

static_assert(MAX_MATCHES == 1, "this test must be built with -DMAX_MATCHES=1 (see the Makefile)");

static uint32_t g_cryptTable[0x500];
static uint32_t hashWithTable(const std::string& s, int off) {
    uint32_t s1 = 0x7FED7FED, s2 = 0xEEEEEEEE;
    for (unsigned char ch : s) { s1 = g_cryptTable[off + ch] ^ (s1 + s2); s2 = ch + s1 + s2 + (s2 << 5) + 3; }
    return s1;
}
static uint32_t hashA(const std::string& s) { return hashWithTable(s, 0x100); }
static uint32_t hashB(const std::string& s) { return hashWithTable(s, 0x200); }

static std::string runCaptured(const SearchRequest& req, SearchResult& result, std::vector<std::string>& reported) {
#ifndef _WIN32
    fflush(stdout); fflush(stderr);
    int fd = open(".capture.tmp", O_CREAT | O_TRUNC | O_WRONLY, 0644);
    int savedOut = dup(1), savedErr = dup(2);
    dup2(fd, 1); dup2(fd, 2);
#endif
    result = runSearch(req, nullptr, [&](const std::string& f) { reported.push_back(removePrefixAndSuffix(f, req.prefix, req.suffix)); });
#ifndef _WIN32
    fflush(stdout); fflush(stderr);
    dup2(savedOut, 1); dup2(savedErr, 2);
    close(savedOut); close(savedErr); close(fd);
    std::ifstream in(".capture.tmp");
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
#else
    return "";
#endif
}

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAILED: %s\n", msg); ++g_failures; } else printf("  ok: %s\n", msg); } while (0)

int main() {
    prepareCryptTable(g_cryptTable);
    printf("MAX_MATCHES=%d\n", MAX_MATCHES);

    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
    const std::string prefix = "REZ\\", suffix = ".WAV";
    const int len = 7;

    // Birthday search for two candidates with the same hashA.
    std::unordered_map<uint32_t, uint64_t> seen;
    uint64_t idxA = 0, idxB = 0;
    uint32_t collHash = 0;
    bool have = false;
    const uint64_t scanStart = 123456789;
    for (uint64_t i = 0; i < 4'000'000 && !have; ++i) {
        uint32_t h = hashA(prefix + indexToString(scanStart + i, len, alphabet) + suffix);
        auto it = seen.find(h);
        if (it != seen.end()) { idxA = it->second; idxB = scanStart + i; collHash = h; have = true; }
        else seen.emplace(h, scanStart + i);
    }
    if (!have) { fprintf(stderr, "TEST BUG: no hashA collision found\n"); return 1; }
    const std::string c1 = indexToString(idxA, len, alphabet), c2 = indexToString(idxB, len, alphabet);
    printf("collision pair: '%s' / '%s' (hashA 0x%08X)\n", c1.c_str(), c2.c_str(), collHash);
    if (hashB(prefix + c1 + suffix) == hashB(prefix + c2 + suffix)) { fprintf(stderr, "TEST BUG: hashB collides too\n"); return 1; }

    auto makeRequest = [&](uint32_t targetA, uint32_t targetB, uint64_t lo, uint64_t hi) {
        SearchRequest req;
        req.alphabet = alphabet; req.prefix = prefix; req.suffix = suffix;
        req.startCandidate = req.lowerBound = indexToString(lo, len, alphabet);
        req.upperBound = indexToString(hi, len, alphabet);
        req.targetHashA = targetA; req.targetHashB = targetB;
        return req;
    };

    // Both candidates are in one launch (2 hits, room for 1).
    const std::string note = "note: 2 hashA hits in one batch";
    auto noKernelDisagreement = [](const std::string& log) {
        return log.find("WARNING") == std::string::npos;
    };

    {
        printf("--- 2 hits, room for 1: hashB selects the LATER candidate ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(collHash, hashB(prefix + c2 + suffix), idxA, idxB), r, reported);
        CHECK(r.ok, "runSearch() succeeded");
        CHECK(r.found && r.filename == prefix + c2 + suffix, "the hashB-matching (later) candidate is found");
        CHECK(reported.size() == 2 && reported[0] == c1 && reported[1] == c2, "both hits reported, once each, in enumeration order");
        CHECK(log.find(note) != std::string::npos, "the fallback to searching in halves was announced");
        CHECK(noKernelDisagreement(log), "no kernel/reference disagreement warnings");
    }
    {
        printf("--- 2 hits, room for 1: hashB selects the EARLIER candidate ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(collHash, hashB(prefix + c1 + suffix), idxA, idxB), r, reported);
        CHECK(r.ok, "runSearch() succeeded");
        CHECK(r.found && r.filename == prefix + c1 + suffix, "the hashB-matching (earlier) candidate is found");
        CHECK(reported.size() == 1 && reported[0] == c1, "finding it ends the search: the later hit is not reported");
        CHECK(log.find(note) != std::string::npos, "the fallback was announced");
        CHECK(noKernelDisagreement(log), "no kernel/reference disagreement warnings");
    }
    {
        printf("--- 2 hits, room for 1: hashB matches neither ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(collHash, hashB(prefix + c1 + suffix) ^ hashB(prefix + c2 + suffix) ^ 0x1, idxA, idxB), r, reported);
        CHECK(r.ok && !r.found, "nothing is found");
        CHECK(reported.size() == 2 && reported[0] == c1 && reported[1] == c2, "both hits are still reported, once each, in enumeration order");
        CHECK(noKernelDisagreement(log), "no kernel/reference disagreement warnings");
    }
    {
        printf("--- the range is narrowed to just one of the pair: no overflow, no note ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(collHash, hashB(prefix + c1 + suffix), idxA, idxB - 1), r, reported);
        CHECK(r.found && r.filename == prefix + c1 + suffix && reported.size() == 1, "the single hit is found and reported");
        CHECK(log.find("note:") == std::string::npos, "with no fallback note");
    }

    // The split point itself: a range chosen so the very first midpoint lands exactly on a hit
    // (the second half then *starts* with it), or so the first half *ends* with a hit. An
    // off-by-one in how the halves are cut would drop exactly such a hit.
    {
        const uint64_t d = idxB - idxA;
        struct Layout { const char* what; uint64_t lo, hi; };
        const Layout layouts[] = {
            {"the first midpoint is exactly the later hit (the second half starts with it)", idxA, idxA + 2 * d},
            {"the first half ends exactly with the earlier hit", idxA - d, idxB + 1},
        };
        for (const Layout& L : layouts) {
            for (int which = 0; which < 2; ++which) {
                const std::string& winner = which == 0 ? c1 : c2;
                printf("--- %s; hashB selects the %s hit ---\n", L.what, which == 0 ? "earlier" : "later");
                SearchResult r; std::vector<std::string> reported;
                std::string log = runCaptured(makeRequest(collHash, hashB(prefix + winner + suffix), L.lo, L.hi), r, reported);
                CHECK(r.ok && r.found && r.filename == prefix + winner + suffix, "the hashB-matching hit is found");
                CHECK(std::find(reported.begin(), reported.end(), winner) != reported.end(), "and reported");
                if (which == 1) // the later one: the earlier hit precedes it, so it must have been reported too
                    CHECK(reported.size() == 2 && reported[0] == c1 && reported[1] == c2, "with the earlier hit reported before it");
                CHECK(noKernelDisagreement(log), "no kernel/reference disagreement warnings");
            }
        }
    }

    // The very next search is unaffected by the overflows above.
    {
        printf("--- the next search is unaffected (no stale match count) ---\n");
        uint64_t single = idxA + 5;
        std::string cand = indexToString(single, len, alphabet);
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(hashA(prefix + cand + suffix), hashB(prefix + cand + suffix), single - 3, single + 3), r, reported);
        CHECK(r.ok && r.found && r.filename == prefix + cand + suffix, "the next search finds its own target");
        CHECK(reported.size() == 1 && reported[0] == cand, "and reports exactly that one hit");
        CHECK(log.find("WARNING") == std::string::npos && log.find("note:") == std::string::npos, "with no warnings or notes");
    }

    // And a range with no hits at all reports nothing.
    {
        printf("--- no hits ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(hashA(prefix + c1 + suffix) ^ 0x1, 0xDEADBEEF, idxA, idxB), r, reported); // matches nothing in the range
        CHECK(r.ok && reported.empty() && !r.found, "a target with no hits reports nothing");
        CHECK(log.find("WARNING") == std::string::npos && log.find("note:") == std::string::npos, "and prints no warning or note");
    }

    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
