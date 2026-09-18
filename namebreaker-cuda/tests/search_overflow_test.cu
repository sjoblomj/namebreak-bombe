// Correctness test for what runSearch() does when one launch has more hashA
// hits than MAX_MATCHES can record - a situation that can't happen in
// practice with MAX_MATCHES = 1024 (a single 32-bit hashA collision is
// already a ~1-in-4-billion event), so this build shrinks it to 1
// (-DMAX_MATCHES=1, see the Makefile) and uses a real hashA collision pair
// (two different candidates with the same hashA, found by a birthday search
// on the CPU) to overflow it with just two hits.
//
// What must hold:
//  * exactly MAX_MATCHES of the hits are reported (no out-of-bounds write
//    past the recorded-matches buffers, no crash, no duplicates),
//  * a loud WARNING is printed rather than the overflow being silent,
//  * the overflow doesn't poison the next runSearch() call (matchCount is
//    reset, nothing stale is reported),
//  * and the documented limitation: a dropped hit is never checked against
//    hashB, so `found` is true only if the recorded hit is the hashB match.
//
// Calls the real runSearch(), which does fopen("matches.txt", "a") relative
// to the current directory - `make test` runs this from tests/.testrun/.

#include <cuda_runtime.h>
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
#include "../search.h"
#include "../cpu-utils.h"
#include "../constants.h"

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
    result = runSearch(req, nullptr, [&](const std::string& f) { reported.push_back(remove_prefix_and_suffix(f, req.prefix, req.suffix)); });
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

    // Both candidates are in one launch, hashB selects c2.
    {
        printf("--- overflow: 2 hashA hits, room for 1, hashB selects the second ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(collHash, hashB(prefix + c2 + suffix), idxA, idxB), r, reported);
        CHECK(r.ok, "runSearch() succeeded");
        CHECK(reported.size() == (size_t) MAX_MATCHES, "exactly MAX_MATCHES hit(s) reported");
        CHECK(reported.size() == 1 && (reported[0] == c1 || reported[0] == c2), "the reported hit is one of the two colliding candidates");
        CHECK(log.find("WARNING: 2 hashA hits in one batch") != std::string::npos, "a WARNING about the overflow was printed");
        bool recordedIsWinner = reported.size() == 1 && reported[0] == c2;
        CHECK(r.found == recordedIsWinner, "found is true exactly when the recorded hit is the hashB match (a dropped hit is never checked against hashB)");
        if (r.found) CHECK(r.filename == prefix + c2 + suffix, "the found filename is the hashB match");
        CHECK(log.find("WARNING: bruteForceKernel reported") == std::string::npos && log.find("WARNING: hashA mismatch") == std::string::npos,
              "no kernel/reference disagreement warnings");
    }

    // The very next search is unaffected by the overflow above.
    {
        printf("--- the next search is unaffected (no stale match count) ---\n");
        uint64_t single = idxA + 5;
        std::string cand = indexToString(single, len, alphabet);
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(hashA(prefix + cand + suffix), hashB(prefix + cand + suffix), single - 3, single + 3), r, reported);
        CHECK(r.ok && r.found && r.filename == prefix + cand + suffix, "the next search finds its own target");
        CHECK(reported.size() == 1 && reported[0] == cand, "and reports exactly that one hit");
        CHECK(log.find("WARNING") == std::string::npos, "with no warnings");
    }

    // (The "next search" above also covers exactly MAX_MATCHES hits, which is not an overflow.)
    // And a range with no hits at all reports nothing.
    {
        printf("--- no hits ---\n");
        SearchResult r; std::vector<std::string> reported;
        std::string log = runCaptured(makeRequest(hashA(prefix + c1 + suffix) ^ 0x1, 0xDEADBEEF, idxA, idxB), r, reported); // matches nothing in the range
        CHECK(r.ok && reported.empty() && !r.found, "a target with no hits reports nothing");
        CHECK(log.find("WARNING") == std::string::npos, "and prints no warning");
    }

    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
