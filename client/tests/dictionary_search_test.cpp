// End-to-end correctness test for runDictionarySearch()
// (engine/dictionary_search.h) on one backend, against brute force: every
// candidate's filename made and hashed from scratch here, in number order,
// and compared with what the search reports - its hashA hits, its match of
// both hashes, the basenames it records, how many candidates it searched -
// over random words (some with '\' in them), separators, word counts,
// prefixes, suffixes, bounds and start numbers. Plus stopping and resuming,
// pausing, and the files it writes.
//
// Built against several search libraries (see CMakeLists.txt): one with only
// a few of hashA's bits compared (NAMEBREAK_HASHA_MATCH_BITS), so that
// about one candidate in 256 is a hit and the hit lists really get
// compared, and one that hands the backend only a few candidates per call
// (NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL) on several threads.
//
//   dictionary_search_test --backend <name>
//
// Writes its files under ./dictionary_search_test/ - ctest runs this from its
// own directory under build/testrun/.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "backends/backends.h"
#include "engine/dictionary_search.h"
#include "engine/hash_match.h"
#include "engine/mpq_hash.h"

static int g_failures = 0;
static int g_checks = 0;

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "dictionary_search_test";
static uint32_t g_table[0x500];

// The MPQ hash of `s` from scratch, with the crypt table at `offset` - this
// test's own, sharing nothing with the engine but the crypt table.
static uint32_t hashOf(const std::string& s, int offset) {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : s) {
        seed1 = g_table[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}

static std::string basename(const std::string& filename) {
    const size_t slash = filename.rfind('\\');
    return slash == std::string::npos ? filename : filename.substr(slash + 1);
}

// Every candidate's text in number order, by nested loops (see
// dictionary.h's numbering).
static std::vector<std::string> allCandidates(const DictionaryPattern& p) {
    std::vector<std::string> out;
    for (int k = p.minWords; k <= p.maxWords; ++k) {
        std::function<void(int, const std::string&)> gen = [&](int level, const std::string& text) {
            for (const std::string& word : p.words) {
                if (level == k - 1) {
                    out.push_back(text + word);
                    continue;
                }
                for (const std::string& separator : p.separators)
                    gen(level + 1, text + word + separator);
            }
        };
        gen(0, "");
    }
    return out;
}

static std::string randomString(std::mt19937& rng, const std::string& alphabet, int minLen, int maxLen) {
    std::string s;
    const int len = std::uniform_int_distribution<int>(minLen, maxLen)(rng);
    for (int i = 0; i < len; ++i)
        s += alphabet[std::uniform_int_distribution<size_t>(0, alphabet.size() - 1)(rng)];
    return s;
}

static std::vector<std::string> readLines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

// Forwards everything to another backend, counting its runDictionaryBatches
// calls - and setting `abortAfter`'s flag once it has made that many.
class CountingBackend : public SearchBackend {
public:
    explicit CountingBackend(SearchBackend& inner) : inner_(inner) {}
    const char* name() const override { return inner_.name(); }
    int windowChars() const override { return inner_.windowChars(); }
    int maxTrailingLen() const override { return inner_.maxTrailingLen(); }
    uint64_t batchSize(int alphabetSize) const override { return inner_.batchSize(alphabetSize); }
    void beginSearch(const SearchConstants& constants) override { inner_.beginSearch(constants); }
    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override {
        return inner_.runBatch(trailingLen, start, count, params);
    }
    void endSearch() override { inner_.endSearch(); }
    bool supportsDictionary() const override { return inner_.supportsDictionary(); }
    uint64_t dictionaryCandidatesPerCall() const override { return inner_.dictionaryCandidatesPerCall(); }
    void beginDictionarySearch(const DictionaryConstants& constants) override {
        ++begun;
        inner_.beginDictionarySearch(constants);
    }
    DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches) override {
        DictionaryOutcome outcome = inner_.runDictionaryBatches(batches);
        if (++calls == abortAfter && abortFlag)
            abortFlag->store(true);
        return outcome;
    }
    void endDictionarySearch() override {
        ++ended;
        inner_.endDictionarySearch();
    }

    std::atomic<int> calls{0};
    int begun = 0, ended = 0;
    int abortAfter = -1;
    std::atomic<bool>* abortFlag = nullptr;

private:
    SearchBackend& inner_;
};

// A backend that can't search dictionaries.
class RowsOnlyBackend : public SearchBackend {
public:
    const char* name() const override { return "rows-only"; }
    int windowChars() const override { return 1; }
    int maxTrailingLen() const override { return 1; }
    uint64_t batchSize(int) const override { return 1; }
    void beginSearch(const SearchConstants&) override {}
    BatchOutcome runBatch(int, uint64_t, uint64_t, const BatchParams&) override { return {}; }
    void endSearch() override {}
};

// A request with its files under kDir/<name>, removed first, and progress
// saved after every call to the backend.
static DictionaryRequest freshRequest(const std::string& name) {
    const std::string dir = kDir + "/" + name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    DictionaryRequest req;
    req.outputFilePath = dir + "/matches.txt";
    req.basenamesFilePath = dir + "/basenames.txt";
    req.progressFilePath = dir + "/wordnumber.txt";
    req.progressInterval = std::chrono::milliseconds(0);
    req.statusInterval = std::chrono::hours(1);
    req.wordSource = "test";
    return req;
}

// What a search of `req` must find, from brute force.
struct Expected {
    std::vector<std::string> hits;    // every hashA hit, in number order
    std::set<std::string> basenames;  // every basename that matches the key
    std::string firstFound;           // the first match of both hashes, if any
    uint64_t searched = 0;            // candidates within the bounds
    uint64_t total = 0;               // candidates in all
};

static Expected expectedOf(const DictionaryRequest& req) {
    Expected e;
    const std::vector<std::string> all = allCandidates(req.pattern);
    e.total = all.size();
    for (uint64_t n = req.startNumber; n < all.size(); ++n) {
        const std::string filename = req.prefix + all[n] + req.suffix;
        if (!req.bounds.contains(filename))
            continue;
        ++e.searched;
        if (hashAMatches(hashOf(filename, 0x100), req.targetHashA)) {
            e.hits.push_back(filename);
            if (e.firstFound.empty() && hashOf(filename, 0x200) == req.targetHashB)
                e.firstFound = filename;
        }
        if (req.checkBasename && hashOf(basename(filename), 0x300) == req.basenameKey)
            e.basenames.insert(basename(filename));
    }
    return e;
}

static DictionaryResult run(SearchBackend& backend, const DictionaryRequest& req, std::vector<std::string>* hits = nullptr,
                            std::atomic<bool>* abort = nullptr, const std::atomic<bool>* pause = nullptr) {
    return runDictionarySearch(
        backend, req, abort,
        [&](const std::string& hit) {
            if (hits)
                hits->push_back(hit);
        },
        pause);
}

// Random words, separators, prefix, suffix, bounds and start, over a few
// characters - '\' among them, so that basenames start inside words and
// separators - and short words, so that some are prefixes of others.
static DictionaryRequest randomRequest(std::mt19937& rng, int round) {
    DictionaryRequest req = freshRequest("random");
    const std::string alphabet = "AB\\_.";
    std::vector<std::string> words;
    const int wordCount = 1 + (int) (rng() % 9);
    for (int i = 0; i < wordCount; ++i)
        words.push_back(randomString(rng, alphabet, 1, 3));
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    req.pattern.words = words;
    const std::vector<std::vector<std::string>> separatorSets = {{""}, {"", "_"}, {"\\"}, {"_", "", "\\", "AB"}};
    req.pattern.separators = separatorSets[rng() % separatorSets.size()];
    req.pattern.minWords = 1 + (int) (rng() % 2);
    req.pattern.maxWords = req.pattern.minWords + (int) (rng() % (words.size() > 4 ? 2 : 3));
    req.prefix = std::vector<std::string>{"", "REZ\\", "A", "X\\Y_"}[rng() % 4];
    req.suffix = std::vector<std::string>{"", ".TXT", "\\Z", "B"}[rng() % 4];

    const std::vector<std::string> all = allCandidates(req.pattern);
    auto randomBound = [&]() {
        std::string f = req.prefix + all[rng() % all.size()] + req.suffix;
        if (rng() % 2)
            f = f.substr(0, rng() % (f.size() + 1));
        return f;
    };
    req.bounds.hasLower = rng() % 3 != 0;
    req.bounds.lower = req.bounds.hasLower ? randomBound() : "";
    req.bounds.hasUpper = rng() % 3 != 0;
    req.bounds.upper = req.bounds.hasUpper ? randomBound() : "";
    if (req.bounds.hasLower && req.bounds.hasUpper && req.bounds.upper < req.bounds.lower)
        std::swap(req.bounds.lower, req.bounds.upper);
    req.startNumber = rng() % 3 ? 0 : rng() % (all.size() + 1);

    // The targets: candidates the search reaches, if there are any - or, for
    // hashB in two rounds of three, none of them (practically).
    std::vector<std::string> reached;
    for (uint64_t n = req.startNumber; n < all.size(); ++n) {
        if (req.bounds.contains(req.prefix + all[n] + req.suffix))
            reached.push_back(req.prefix + all[n] + req.suffix);
    }
    if (reached.empty())
        reached.push_back(req.prefix + all[rng() % all.size()] + req.suffix);
    const std::string planted = reached[rng() % reached.size()];
    req.targetHashA = round % 3 == 0 || rng() % 2 ? hashOf(planted, 0x100) : (uint32_t) rng();
    req.targetHashB = round % 3 == 0 ? hashOf(planted, 0x200) : (uint32_t) rng();
    req.checkBasename = rng() % 4 != 0;
    req.basenameKey = hashOf(basename(reached[rng() % reached.size()]), 0x300);
    return req;
}

static void testAgainstBruteForce(SearchBackend& backend) {
    printf("--- against brute force ---\n");
    std::mt19937 rng(20261007);
    int rounds = 0, found = 0, withHits = 0, withBasenames = 0, mismatches = 0;
    for (int round = 0; round < 150; ++round) {
        const DictionaryRequest req = randomRequest(rng, round);
        const Expected e = expectedOf(req);
        std::vector<std::string> hits;
        const DictionaryResult r = run(backend, req, &hits);
        ++rounds;
        bool ok = r.ok && !r.aborted && r.found == !e.firstFound.empty();
        if (ok && r.found) {
            // Stopped at the call with the first match: it's that one.
            ok = r.filename == e.firstFound;
            const std::vector<std::string> matches = readLines(req.outputFilePath);
            const std::vector<std::string> foundTxt = readLines(std::filesystem::path(req.outputFilePath).parent_path().string() + "/found.txt");
            ok = ok && matches == std::vector<std::string>{e.firstFound} && foundTxt == std::vector<std::string>{e.firstFound};
            ++found;
        } else if (ok) {
            // Every candidate searched: every hit reported, and nothing else.
            std::vector<std::string> got = hits, want = e.hits;
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            ok = got == want && r.candidatesSearched == e.searched && r.nextNumber == e.total;
            if (!e.hits.empty()) {
                ++withHits;
                const std::vector<std::string> matches = readLines(req.outputFilePath);
                ok = ok && matches.size() == 1 && std::find(want.begin(), want.end(), matches[0]) != want.end();
            }
            // The basenames: each once, the file holding exactly them.
            const std::vector<std::string> lines = readLines(req.basenamesFilePath);
            const std::set<std::string> fileSet(lines.begin(), lines.end());
            ok = ok && fileSet.size() == lines.size() && fileSet == e.basenames && r.basenameHits == e.basenames.size();
            withBasenames += e.basenames.empty() ? 0 : 1;
            if (!req.checkBasename)
                ok = ok && !std::filesystem::exists(req.basenamesFilePath);
            // The progress file: done, from this search.
            DictionaryProgress progress;
            bool exists = false;
            std::string error;
            ok = ok && readDictionaryProgress(req.progressFilePath, progress, exists, error) && exists && progress.next == e.total &&
                 progress.fingerprint == dictionaryFingerprint(req);
        }
        if (!ok) {
            ++mismatches;
            if (mismatches <= 5) {
                fprintf(stderr, "  round %d: words", round);
                for (const std::string& w : req.pattern.words)
                    fprintf(stderr, " '%s'", w.c_str());
                fprintf(stderr, ", %zu separators, %d-%d words, prefix '%s', suffix '%s', lower %s'%s', upper %s'%s', start %llu\n",
                        req.pattern.separators.size(), req.pattern.minWords, req.pattern.maxWords, req.prefix.c_str(),
                        req.suffix.c_str(), req.bounds.hasLower ? "" : "(none) ", req.bounds.lower.c_str(), req.bounds.hasUpper ? "" : "(none) ",
                        req.bounds.upper.c_str(), (unsigned long long) req.startNumber);
                fprintf(stderr, "    ok %d found %d/%d ('%s' vs '%s'), hits %zu/%zu, searched %llu/%llu, next %llu/%llu, basenames %llu/%zu %s\n",
                        r.ok, r.found, !e.firstFound.empty(), r.filename.c_str(), e.firstFound.c_str(), hits.size(), e.hits.size(),
                        (unsigned long long) r.candidatesSearched, (unsigned long long) e.searched, (unsigned long long) r.nextNumber,
                        (unsigned long long) e.total, (unsigned long long) r.basenameHits, e.basenames.size(), r.error.c_str());
            }
        }
    }
    check(mismatches == 0, std::to_string(rounds) + " random searches: as brute force (" + std::to_string(mismatches) + " differ)");
    check(found >= 30, "... " + std::to_string(found) + " of them finding a planted match of both hashes");
    check(withBasenames >= 30, "... " + std::to_string(withBasenames) + " of them recording basenames");
    if (kHashAMatchMask != 0xFFFFFFFFu)
        check(withHits >= 40, "... " + std::to_string(withHits) + " of them with hashA hits to compare");
}

// The Portuguese StarDat's credits file's sibling, with its real hashes.
static DictionaryRequest creditsRequest(const std::string& name) {
    DictionaryRequest req = freshRequest(name);
    req.pattern.words = {"CRDT", "LST", "MAP", "ZERG"};
    req.pattern.separators = {"", "_"};
    req.pattern.minWords = 1;
    req.pattern.maxWords = 2;
    req.prefix = "REZ\\";
    req.suffix = ".TXT";
    req.targetHashA = 0x339EFE07;
    req.targetHashB = 0xE6B2B01C;
    req.checkBasename = true;
    req.basenameKey = 0x4565C467;
    return req;
}

static void testRealHashes(SearchBackend& backend) {
    printf("--- real hashes ---\n");
    {
        const DictionaryRequest req = creditsRequest("credits");
        const DictionaryResult r = run(backend, req);
        check(r.ok && r.found && r.filename == "REZ\\CRDT_LST.TXT", "finds REZ\\CRDT_LST.TXT by its hashA and hashB");
        check(readLines(req.basenamesFilePath) == std::vector<std::string>{"CRDT_LST.TXT"} && r.basenameHits == 1,
              "and records its basename CRDT_LST.TXT, by its encryption key");
        check(readLines(req.outputFilePath) == std::vector<std::string>{"REZ\\CRDT_LST.TXT"}, "the matches file holds the match");
        check(readLines(kDir + "/credits/found.txt") == std::vector<std::string>{"REZ\\CRDT_LST.TXT"}, "and found.txt too");
    }
    {
        // Wrong prefix: no match of both hashes, but the basename still.
        DictionaryRequest req = creditsRequest("credits-elsewhere");
        req.prefix = "SOMEWHERE\\ELSE\\";
        const DictionaryResult r = run(backend, req);
        check(r.ok && !r.found && readLines(req.basenamesFilePath) == std::vector<std::string>{"CRDT_LST.TXT"},
              "with the wrong directory: no match, but the basename is recorded");
    }
    {
        // A '\' separator: the basename is after it, the same in every
        // directory the first word makes - recorded once.
        DictionaryRequest req = creditsRequest("credits-dirs");
        req.pattern.separators = {"", "_", "\\"};
        req.pattern.maxWords = 3;
        req.prefix = "";
        req.targetHashA = 0;
        const DictionaryResult r = run(backend, req);
        check(r.ok && readLines(req.basenamesFilePath) == std::vector<std::string>{"CRDT_LST.TXT"} && r.basenameHits == 1,
              "a basename made in many directories (a '\\' separator): recorded once");
    }
    {
        DictionaryRequest req = creditsRequest("credits-no-basenames");
        req.checkBasename = false;
        const DictionaryResult r = run(backend, req);
        check(r.ok && r.found && !std::filesystem::exists(req.basenamesFilePath), "not checking basenames: no basenames file");
    }
}

static void testBasenamesFile(SearchBackend& backend) {
    printf("--- the basenames file ---\n");
    DictionaryRequest req = creditsRequest("basenames-kept");
    req.targetHashB = 0; // never found, so every candidate is searched
    {
        std::ofstream out(req.basenamesFilePath, std::ios::binary);
        out << "EARLIER.TXT\r\nCRDT_LST.TXT\n";
    }
    DictionaryResult r = run(backend, req);
    check(r.ok && readLines(req.basenamesFilePath) == std::vector<std::string>{"EARLIER.TXT", "CRDT_LST.TXT"} && r.basenameHits == 0,
          "a basename already in the file isn't written again, and what was there is kept");
    std::filesystem::remove(req.basenamesFilePath);
    run(backend, req);
    r = run(backend, req);
    check(r.ok && readLines(req.basenamesFilePath) == std::vector<std::string>{"CRDT_LST.TXT"} && r.basenameHits == 0,
          "searched twice: written once");
}

static void testEdges(SearchBackend& backend) {
    printf("--- edge cases ---\n");
    {
        DictionaryRequest req = creditsRequest("past-the-end");
        req.startNumber = 1000;
        const DictionaryResult r = run(backend, req);
        check(r.ok && !r.found && r.candidatesSearched == 0 && r.nextNumber == 4 + 4 * 2 * 4, "a start past the last number: nothing searched, done");
    }
    {
        DictionaryRequest req = creditsRequest("at-the-end");
        req.startNumber = 4 + 4 * 2 * 4 - 1; // ZERG_ZERG, the last
        req.targetHashB = 0;
        const DictionaryResult r = run(backend, req);
        check(r.ok && !r.found && r.candidatesSearched == 1 && r.nextNumber == 36, "a start at the last number: that one searched");
    }
    {
        DictionaryRequest req = creditsRequest("all-outside");
        req.bounds.hasLower = true;
        req.bounds.lower = "SOUND\\";
        CountingBackend counting(backend);
        const DictionaryResult r = run(counting, req);
        check(r.ok && !r.found && r.candidatesSearched == 0 && r.nextNumber == 36 && counting.calls == 0,
              "every candidate outside the bounds: none searched, the backend never called, done");
        check(counting.begun == 1 && counting.ended == 1, "... but the search still begun and ended on it once");
    }
    {
        DictionaryRequest req = creditsRequest("just-one");
        req.bounds = FilenameBounds{true, "REZ\\CRDT_LST.TXT", true, "REZ\\CRDT_LST.TXT"};
        const DictionaryResult r = run(backend, req);
        check(r.ok && r.found && r.candidatesSearched == 1, "bounds around the match alone: only it searched, and found");
    }
    {
        DictionaryRequest req = creditsRequest("invalid");
        req.pattern.words = {"ZERG", "CRDT"};
        const DictionaryResult r = run(backend, req);
        check(!r.ok && r.error.find("sorted") != std::string::npos, "unsorted words: not ok, saying why");
    }
    {
        RowsOnlyBackend rowsOnly;
        const DictionaryResult r = run(rowsOnly, creditsRequest("unsupported"));
        check(!r.ok && r.error.find("can't search dictionaries") != std::string::npos, "a backend that can't: not ok, saying why");
    }
    {
        DictionaryRequest req = creditsRequest("basenames-unwritable");
        req.basenamesFilePath = kDir + "/basenames-unwritable/basenames.txt";
        std::filesystem::create_directories(req.basenamesFilePath); // a directory, not a file
        check(!run(backend, req).ok, "a basenames file that can't be opened: not ok");
        req.basenamesFilePath = req.outputFilePath + "/basenames.txt"; // under a file
        check(!run(backend, req).ok, "a basenames file whose directory can't be made: not ok");
        req.checkBasename = false;
        check(run(backend, req).ok, "... unless basenames aren't checked");
    }
    {
        DictionaryRequest req = creditsRequest("progress-unwritable");
        req.progressFilePath = kDir + "/progress-unwritable/no/such/dir/wordnumber.txt";
        const DictionaryResult r = run(backend, req);
        check(r.ok && r.found, "a progress file that can't be written: only a warning, the search goes on");
    }
    {
        DictionaryRequest req = creditsRequest("status");
        req.statusInterval = std::chrono::milliseconds(0);
        req.targetHashB = 0;
        check(run(backend, req).ok, "a status line after every call: fine");
    }
    {
        DictionaryRequest req = creditsRequest("unwritable");
        req.outputFilePath = kDir + "/unwritable/matches.txt/x.txt";
        std::ofstream(kDir + "/unwritable/matches.txt") << "a file, not a directory";
        const DictionaryResult r = run(backend, req);
        check(!r.ok, "a matches file that can't be made: not ok");
    }
}

static void testStopAndResume(SearchBackend& backend) {
    printf("--- stopping and resuming ---\n");
    // About 400,000 candidates within the bounds: several calls on the
    // backends built to take few per call (see CMakeLists.txt), one on the
    // others - which then only check that it isn't stopped part way.
    DictionaryRequest req = freshRequest("resume");
    for (int i = 0; i < 40; ++i) {
        char buf[8];
        snprintf(buf, sizeof buf, "W%02d", i);
        req.pattern.words.push_back(buf);
    }
    req.pattern.separators = {"", "_", "-"};
    req.pattern.maxWords = 3;
    req.prefix = "REZ\\";
    req.suffix = ".TXT";
    req.bounds = FilenameBounds{true, "REZ\\W05", true, "REZ\\W30_W10"};
    req.targetHashA = 0x12345678;
    req.targetHashB = 0;
    req.checkBasename = true;
    req.basenameKey = hashOf("W10-W20_W07.TXT", 0x300);
    const Expected e = expectedOf(req);

    uint64_t perCall = std::max<uint64_t>(1, backend.dictionaryCandidatesPerCall());
#ifdef NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL
    perCall = std::min<uint64_t>(perCall, NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL);
#endif
    // Stop after the first call - or, with many candidates per call, a few
    // in: shrinking what one call gets isn't the backend's to say, so the
    // test only asks for a stop the search can reach.
    std::atomic<bool> abort{false};
    CountingBackend counting(backend);
    counting.abortFlag = &abort;
    counting.abortAfter = 1;
    std::vector<std::string> hits1, hits2;
    const DictionaryResult r1 = run(counting, req, &hits1, &abort);
    if (e.searched > perCall) {
        check(r1.ok && r1.aborted && !r1.found && r1.nextNumber > 0 && r1.nextNumber < e.total && r1.candidatesSearched > 0 &&
                  r1.candidatesSearched < e.searched,
              "aborted after the first call: stopped part way");
    } else {
        check(r1.ok && !r1.aborted, "everything fits in one call: not aborted");
    }
    DictionaryProgress progress;
    bool exists = false;
    std::string error;
    check(readDictionaryProgress(req.progressFilePath, progress, exists, error) && exists && progress.next == r1.nextNumber &&
              progress.fingerprint == dictionaryFingerprint(req),
          "the progress file says how far it got");

    DictionaryRequest resumed = req;
    resumed.startNumber = progress.next;
    const DictionaryResult r2 = run(backend, resumed, &hits2);
    check(r2.ok && !r2.aborted && r2.nextNumber == e.total, "resumed from there: to the end");
    check(r1.candidatesSearched + r2.candidatesSearched == e.searched,
          "the two together searched every candidate within the bounds exactly once (" + std::to_string(r1.candidatesSearched) + " + " +
              std::to_string(r2.candidatesSearched) + " of " + std::to_string(e.searched) + ")");
    std::vector<std::string> both = hits1;
    both.insert(both.end(), hits2.begin(), hits2.end());
    std::sort(both.begin(), both.end());
    std::vector<std::string> want = e.hits;
    std::sort(want.begin(), want.end());
    check(both == want, "and their hashA hits together are every one (" + std::to_string(want.size()) + ")");
    check(readLines(req.basenamesFilePath) == std::vector<std::string>{"W10-W20_W07.TXT"}, "the basename found in one of them, recorded once");

    {
        std::atomic<bool> abortNow{true};
        DictionaryRequest fromMiddle = req;
        fromMiddle.startNumber = 1000;
        const DictionaryResult r = run(backend, fromMiddle, nullptr, &abortNow);
        check(r.ok && r.aborted && r.candidatesSearched == 0 && r.nextNumber == 1000, "aborted before it starts: nothing searched");
        check(readDictionaryProgress(req.progressFilePath, progress, exists, error) && progress.next == 1000,
              "... and the progress file says so");
    }

    printf("--- pausing ---\n");
    {
        DictionaryRequest paused = req;
        paused.progressInterval = std::chrono::hours(1); // only on the pause, and at the end
        paused.startNumber = 77;
        std::filesystem::remove(paused.progressFilePath);
        std::atomic<bool> pause{true};
        CountingBackend counting2(backend);
        DictionaryResult r;
        std::thread searcher([&]() { r = run(counting2, paused, nullptr, nullptr, &pause); });
        bool saved = false;
        for (int i = 0; i < 200 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            saved = std::filesystem::exists(paused.progressFilePath);
        }
        const int callsWhilePaused = counting2.calls;
        bool savedRight = saved && readDictionaryProgress(paused.progressFilePath, progress, exists, error) && progress.next == 77;
        pause = false;
        searcher.join();
        check(savedRight, "paused: the progress is saved at once");
        check(callsWhilePaused == 0, "... and nothing searched while paused");
        check(r.ok && !r.aborted && r.nextNumber == e.total && counting2.calls > 0, "unpaused: it carries on to the end");
    }
    {
        std::atomic<bool> pause{true}, abortFlag{false};
        DictionaryRequest paused = req;
        DictionaryResult r;
        std::thread searcher([&]() { r = run(backend, paused, nullptr, &abortFlag, &pause); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        abortFlag = true;
        searcher.join();
        check(r.ok && r.aborted && r.candidatesSearched == 0, "aborted while paused: stops, without searching");
    }
}

int main(int argc, char* argv[]) {
    std::string backendName;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--backend" && i + 1 < argc)
            backendName = argv[++i];
    }
    if (backendName.empty()) {
        fprintf(stderr, "usage: %s --backend <name>\n", argv[0]);
        return 1;
    }
    std::string error;
    bool selfTestFailed = false;
    std::unique_ptr<SearchBackend> backend = createBackend(backendName, error, &selfTestFailed);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return selfTestFailed ? 1 : 77;
    }
    if (!backend->supportsDictionary()) {
        printf("the %s backend can't search dictionaries - skipped\n", backendName.c_str());
        return 77;
    }
    prepareCryptTable(g_table);
    printf("backend %s, hashA bits compared %d\n", backendName.c_str(), NAMEBREAK_HASHA_MATCH_BITS);

    testAgainstBruteForce(*backend);
    testRealHashes(*backend);
    testBasenamesFile(*backend);
    testEdges(*backend);
    testStopAndResume(*backend);

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
