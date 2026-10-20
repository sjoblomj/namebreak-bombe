// End-to-end correctness test for runJunctionSearch()
// (engine/junction_search.h) on one backend, against brute force: every
// directory of the space made and hashed from scratch here, with this file's
// own hash function, and compared with what the search reports - the
// directories that match the junction, the match of both hashes, how many
// tails it looked up - over random alphabets, prefixes, suffixes, directory
// lengths, pruning rules and, for each of them, every split between the
// table and the backward walk. Plus ranges, stopping and resuming, pausing,
// the refusals, and the files it writes.
//
// The split is what matters most here: a directory of n characters is found
// as the table's first p and the walk's last n - p, and every p from 0 (the
// whole directory rewound from the junction) to n (all of it in the table)
// has to find exactly the same directories. A table that keeps the wrong
// state, or a tail whose index doesn't line up with it, would show up as one
// split finding something the others don't.
//
// Built against several search libraries (see CMakeLists.txt): one with only
// a few of the junction state's bits compared (NAMEBREAK_JUNCTION_MATCH_BITS),
// so that directories match the junction by chance all the time and the
// reported lists really get compared, and one that hands the backend only a
// few tails per call (NAMEBREAK_JUNCTION_TAILS_PER_CALL) on several threads.
//
//   junction_search_test --backend <name>
//
// Writes its files under ./junction_search_test/ - ctest runs this from its
// own directory under build/testrun/.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "backends/backends.h"
#include "engine/hash_match.h"
#include "engine/junction_search.h"
#include "engine/mpq_hash.h"

static int g_failures = 0;
static int g_checks = 0;
static uint64_t g_hitsCompared = 0;

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "junction_search_test";
static uint32_t g_table[0x500];

// The MPQ hash state of `s` from the start, with the crypt table at `offset`
// - this test's own, sharing nothing with the engine but the crypt table.
static std::pair<uint32_t, uint32_t> stateOf(const std::string& s, int offset) {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : s) {
        seed1 = g_table[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return {seed1, seed2};
}

// Every string of `length` characters of `alphabet`, in alphabet order.
static std::vector<std::string> allStrings(const std::string& alphabet, int length) {
    std::vector<std::string> out = {""};
    for (int i = 0; i < length; ++i) {
        std::vector<std::string> longer;
        for (const std::string& s : out) {
            for (char c : alphabet)
                longer.push_back(s + c);
        }
        out = std::move(longer);
    }
    return out;
}

// --- What a search must report -----------------------------------------------

struct Expected {
    std::vector<std::string> dirs; // sorted
    uint64_t probed = 0;
    // Whether the file itself is in the range - then the search stops at it,
    // and the directories it reports are only those up to that point.
    bool found = false;
    std::string filename;
};

// Whether the rules reject `tail` on its own, from the right - what the walk
// skips.
static bool tailRejected(const PruneRules& rules, const std::string& suffix, const std::string& tail) {
    TailPruneState state = tailPruneStart(suffix);
    for (size_t i = tail.size(); i-- > 0;) {
        if (!tailPruneStep(rules, state, tail[i]))
            return true;
    }
    return false;
}

// Brute force: every directory of the request's lengths, hashed from
// scratch, that matches the junction and passes the rules in full - and that
// the range [startNumber, endNumber) covers. The directories shorter than
// the table belong to the range that holds the walk's first node.
static Expected expectedOf(const JunctionRequest& req, int tableLength) {
    JunctionSpace space;
    std::string error;
    Expected out;
    if (!JunctionSpace::create(req.alphabet, req.minLength, req.maxLength, tableLength, space, error)) {
        fprintf(stderr, "the test's own space: %s\n", error.c_str());
        return out;
    }
    const uint64_t end = std::min(req.endNumber, space.size());
    const uint64_t start = std::min(req.startNumber, end);

    for (int length = req.minLength; length <= req.maxLength; ++length) {
        for (const std::string& dir : allStrings(req.alphabet, length)) {
            const auto state = stateOf(req.prefix + dir, kHashAOffset);
            if (junctionKey(state.first, state.second) != junctionKey(req.junctionA.first, req.junctionA.second))
                continue;
            if (!dirPasses(req.rules, req.prefix, req.suffix, dir))
                continue;
            // Where the search reports it from: the walk's node, or - for a
            // directory shorter than the table - the table's own build.
            if (length < tableLength) {
                if (start != 0)
                    continue;
            } else {
                uint64_t index = 0;
                if (!space.indexOfTail(dir.substr(tableLength), index, error))
                    continue;
                const uint64_t number = space.numberOf(length - tableLength, index);
                if (number < start || number >= end)
                    continue;
            }
            out.dirs.push_back(dir);
            const std::string filename = req.prefix + dir + req.suffix;
            const auto hashA = stateOf(filename, kHashAOffset);
            const auto hashB = stateOf(filename, kHashBOffset);
            if (!out.found && hashAMatches(hashA.first, req.targetHashA) && hashB.first == req.targetHashB) {
                out.found = true;
                out.filename = filename;
            }
        }
    }
    std::sort(out.dirs.begin(), out.dirs.end());

    for (uint64_t number = start; number < end; ++number) {
        if (!tailRejected(req.rules, req.suffix, space.tailOf(number)))
            ++out.probed;
    }
    return out;
}

// --- Running one --------------------------------------------------------------

struct Run {
    JunctionResult result;
    std::vector<std::string> dirs; // what the hook reported, sorted
};

// Runs `req` with `tableLength` and collects what it reports.
static Run run(SearchBackend& backend, JunctionRequest req, int tableLength, std::atomic<bool>* abort = nullptr,
               const std::atomic<bool>* pause = nullptr) {
    req.tableLength = tableLength;
    req.outputFilePath = kDir + "/matches.txt";
    req.progressFilePath = kDir + "/junction.txt";
    Run out;
    JunctionSearchHooks hooks;
    hooks.onPartialMatch = [&](const std::string& filename) {
        // Reported with the prefix and suffix around it, as every mode does.
        if (filename.size() >= req.prefix.size() + req.suffix.size() && filename.compare(0, req.prefix.size(), req.prefix) == 0 &&
            filename.compare(filename.size() - req.suffix.size(), req.suffix.size(), req.suffix) == 0) {
            out.dirs.push_back(filename.substr(req.prefix.size(), filename.size() - req.prefix.size() - req.suffix.size()));
        } else {
            out.dirs.push_back("(not this search's name: " + filename + ")");
        }
    };
    out.result = runJunctionSearch(backend, req, abort, hooks, pause);
    std::sort(out.dirs.begin(), out.dirs.end());
    return out;
}

// A request for a file whose directory is `dir`: the junction states and the
// two hashes worked out here, from scratch, as a recovery like
// tools/mitm/junction.c would give them.
static JunctionRequest requestFor(const std::string& alphabet, const std::string& prefix, const std::string& dir,
                                  const std::string& suffix, int minLength, int maxLength) {
    JunctionRequest req;
    req.alphabet = alphabet;
    req.prefix = prefix;
    req.suffix = suffix;
    req.minLength = minLength;
    req.maxLength = maxLength;
    const std::pair<uint32_t, uint32_t> a = stateOf(prefix + dir, kHashAOffset);
    const std::pair<uint32_t, uint32_t> b = stateOf(prefix + dir, kHashBOffset);
    req.junctionA = {a.first, a.second};
    req.junctionB = {b.first, b.second};
    req.targetHashA = stateOf(prefix + dir + suffix, kHashAOffset).first;
    req.targetHashB = stateOf(prefix + dir + suffix, kHashBOffset).first;
    // Nothing is printed every few seconds in a test that takes less than
    // that; the progress file is written when it ends anyway.
    req.statusInterval = std::chrono::milliseconds(60000);
    req.progressInterval = std::chrono::milliseconds(60000);
    return req;
}

// Both halves must find the same directories, whatever the split: this runs
// `req` at every table length from 0 to maxLength and compares each with
// brute force.
static void checkEverySplit(SearchBackend& backend, const JunctionRequest& req, const std::string& what) {
    for (int tableLength = 0; tableLength <= req.maxLength; ++tableLength) {
        const Expected wanted = expectedOf(req, tableLength);
        const Run got = run(backend, req, tableLength);
        const std::string where = what + ", table " + std::to_string(tableLength);
        if (!got.result.ok) {
            check(false, where + ": " + got.result.error);
            continue;
        }
        g_hitsCompared += wanted.dirs.size();
        if (wanted.found) {
            // The search stops at the file, so it reports some of the
            // directories, ending with that one - and nothing it doesn't.
            std::vector<std::string> extra;
            std::set_difference(got.dirs.begin(), got.dirs.end(), wanted.dirs.begin(), wanted.dirs.end(), std::back_inserter(extra));
            check(got.result.found && got.result.filename == wanted.filename && extra.empty(),
                  where + ": finds '" + wanted.filename + "'" + (got.result.found ? "" : " (it didn't: '" + got.result.filename + "')"));
        } else {
            check(got.dirs == wanted.dirs && !got.result.found,
                  where + ": reports the " + std::to_string(wanted.dirs.size()) + " directories that match the junction" +
                      (got.dirs == wanted.dirs ? "" : " (it reported " + std::to_string(got.dirs.size()) + ")"));
            check(got.result.tailsProbed == wanted.probed, where + ": looks up " + std::to_string(wanted.probed) + " tails" +
                                                               (got.result.tailsProbed == wanted.probed
                                                                    ? ""
                                                                    : " (it looked up " + std::to_string(got.result.tailsProbed) + ")"));
            check(got.result.nextNumber == std::min(req.endNumber, got.result.nextNumber) && got.result.hits == got.dirs.size(),
                  where + ": and counts them");
        }
    }
}

// --- The tests ---------------------------------------------------------------

static void testEverySplit(SearchBackend& backend) {
    printf("--- every split between the table and the walk ---\n");
    // A directory within the search's lengths: found, whatever the split.
    checkEverySplit(backend, requestFor("AB-\\", "", "BA-B", "\\X.Y", 1, 4), "'BA-B' of 'AB-\\'");
    checkEverySplit(backend, requestFor("AB-\\", "SETUPDAT\\", "A\\B", "\\X.Y", 1, 4), "'A\\B' under a prefix");
    // One shorter than the longest the search goes to, so that the lengths
    // above it are searched for nothing - and one shorter than the shortest.
    checkEverySplit(backend, requestFor("AB-", "", "AB", "\\X.Y", 1, 4), "'AB', with longer ones searched too");
    checkEverySplit(backend, requestFor("AB-", "", "A", "\\X.Y", 2, 4), "'A', shorter than the shortest searched");
    // And one longer than the search goes to: nothing to find, so every
    // directory that matches the junction by chance is reported and compared.
    checkEverySplit(backend, requestFor("AB-\\", "", "BAB-AB", "\\X.Y", 1, 4), "a directory longer than the search goes to");
    checkEverySplit(backend, requestFor("AB-", "X", "ABABAB", "A.B", 1, 5), "another, with a suffix that has no '\\'");
}

static void testRules(SearchBackend& backend) {
    printf("--- the pruning rules ---\n");
    PruneRules all;
    all.symbolRuns = true;
    all.unopenedBrackets = true;
    all.adjacentBackslashes = true;
    all.maxBackslashCount = 1;
    PruneRules min;
    min.minBackslashCount = 1;

    JunctionRequest req = requestFor("A-\\()", "", "A\\A-", "\\X.Y", 1, 4);
    req.rules = all;
    checkEverySplit(backend, req, "a directory the rules allow");
    // One the rules reject: not reported at all, however the split falls.
    JunctionRequest rejected = requestFor("A-\\()", "", "A---", "\\X.Y", 1, 4);
    rejected.rules = all;
    for (int tableLength = 0; tableLength <= rejected.maxLength; ++tableLength) {
        const Run got = run(backend, rejected, tableLength);
        check(got.result.ok && !got.result.found &&
                  std::find(got.dirs.begin(), got.dirs.end(), "A---") == got.dirs.end(),
              "a directory with three symbols in a row is never reported, table " + std::to_string(tableLength));
    }
    // A '\' at the end, which the suffix's '\' makes two in a row.
    JunctionRequest trailing = requestFor("A-\\()", "", "A-A\\", "\\X.Y", 1, 4);
    trailing.rules = all;
    for (int tableLength = 0; tableLength <= trailing.maxLength; ++tableLength) {
        const Run got = run(backend, trailing, tableLength);
        check(got.result.ok && !got.result.found, "a directory ending with '\\' is never reported, table " + std::to_string(tableLength));
    }
    // min_backslash_count, which only a whole directory settles.
    JunctionRequest few = requestFor("A-\\", "", "AA-", "\\X.Y", 1, 3);
    few.rules = min;
    for (int tableLength = 0; tableLength <= few.maxLength; ++tableLength) {
        const Run got = run(backend, few, tableLength);
        check(got.result.ok && !got.result.found, "a directory without a '\\' is never reported with min_backslash_count, table " +
                                                      std::to_string(tableLength));
    }
    JunctionRequest enough = requestFor("A-\\", "", "A\\-", "\\X.Y", 1, 3);
    enough.rules = min;
    checkEverySplit(backend, enough, "one with a '\\'");
}

static void testRandom(SearchBackend& backend) {
    printf("--- random spaces ---\n");
    const std::string pool = "AB-\\()_. ";
    std::mt19937_64 rng(20261020);
    for (int round = 0; round < 12; ++round) {
        // An alphabet of a few of the characters, the directory's own among
        // them.
        std::string alphabet;
        std::vector<int> order(pool.size());
        for (size_t i = 0; i < pool.size(); ++i)
            order[i] = (int) i;
        std::shuffle(order.begin(), order.end(), rng);
        const int size = 2 + (int) (rng() % 4);
        for (int i = 0; i < size; ++i)
            alphabet += pool[order[i]];

        const int maxLength = 2 + (int) (rng() % 3);
        const int minLength = 1 + (int) (rng() % maxLength);
        // The directory: sometimes within the lengths searched, sometimes
        // longer, so that some rounds have nothing to find and every chance
        // match is compared instead.
        const int length = 1 + (int) (rng() % (maxLength + 2));
        std::string dir;
        for (int i = 0; i < length; ++i)
            dir += alphabet[rng() % alphabet.size()];
        const std::string prefix = (rng() % 3) == 0 ? "" : ((rng() % 2) == 0 ? "SETUPDAT\\" : "X");
        const std::string suffix = (rng() % 3) == 0 ? "A.B" : ((rng() % 2) == 0 ? "\\X.Y" : "\\A_LONGER_BASENAME.TXT");

        JunctionRequest req = requestFor(alphabet, prefix, dir, suffix, minLength, maxLength);
        if ((rng() % 2) == 0) {
            req.rules.symbolRuns = (rng() % 2) == 0;
            req.rules.unopenedBrackets = (rng() % 2) == 0;
            req.rules.adjacentBackslashes = (rng() % 2) == 0;
            req.rules.maxBackslashCount = (int) (rng() % 3);
            req.rules.minBackslashCount = (int) (rng() % 2);
            if (req.rules.minBackslashCount > req.rules.maxBackslashCount && req.rules.maxBackslashCount != 0)
                req.rules.minBackslashCount = req.rules.maxBackslashCount;
        }
        checkEverySplit(backend, req, "round " + std::to_string(round) + ": '" + dir + "' of '" + alphabet + "', " +
                                          std::to_string(minLength) + "-" + std::to_string(maxLength) + " characters");
    }
}

static void testRanges(SearchBackend& backend) {
    printf("--- ranges ---\n");
    // The same search in pieces must report what the whole one does. Its
    // directory is longer than the lengths searched, so that nothing stops
    // any of them part way through and every range can be held to reporting
    // exactly its own share.
    const JunctionRequest req = requestFor("AB-\\", "", "BAB-AB", "\\X.Y", 1, 4);
    const int tableLength = 2;
    JunctionSpace space;
    std::string error;
    check(JunctionSpace::create(req.alphabet, req.minLength, req.maxLength, tableLength, space, error), "the space");

    const Run whole = run(backend, req, tableLength);
    for (uint64_t piece : {(uint64_t) 1, (uint64_t) 7, space.size() / 3 + 1}) {
        std::vector<std::string> together;
        uint64_t probed = 0, found = 0;
        bool ok = true, covered = true;
        uint64_t previousNext = 0;
        for (uint64_t from = 0; from < space.size(); from += piece) {
            JunctionRequest part = req;
            part.startNumber = from;
            part.endNumber = std::min(space.size(), from + piece);
            const Expected wanted = expectedOf(part, tableLength);
            const Run got = run(backend, part, tableLength);
            ok = ok && got.result.ok;
            // A range reports exactly its own directories - none of them is
            // the file, so none of them stops early.
            covered = covered && got.dirs == wanted.dirs && got.result.tailsProbed == wanted.probed &&
                      got.result.nextNumber == part.endNumber && got.result.nextNumber > previousNext;
            previousNext = got.result.nextNumber;
            together.insert(together.end(), got.dirs.begin(), got.dirs.end());
            probed += got.result.tailsProbed;
            found += got.result.found;
        }
        std::sort(together.begin(), together.end());
        const std::string what = "in ranges of " + std::to_string(piece);
        check(ok, what + ": every range ran");
        check(covered, what + ": each reports its own directories, and says how far it got");
        check(together == whole.dirs, what + ": together they report what one run of the lot does");
        check(probed == whole.result.tailsProbed, what + ": and look up the same tails");
        check(found == (whole.result.found ? 1u : 0u), what + ": with the file found by exactly as many of them");
    }

    // A range past the end, and an empty one: nothing to do, no complaint.
    JunctionRequest past = req;
    past.startNumber = space.size();
    past.endNumber = space.size() + 100;
    const Run none = run(backend, past, tableLength);
    check(none.result.ok && !none.result.found && none.dirs.empty() && none.result.tailsProbed == 0, "a range past the last node is empty");
    JunctionRequest empty = req;
    empty.startNumber = 5;
    empty.endNumber = 5;
    check(run(backend, empty, tableLength).result.ok, "and so is an empty one");
}

static void testFindsItAndWritesIt(SearchBackend& backend) {
    printf("--- the file, and the files ---\n");
    // A real directory of eight characters, with the hashes of the name it
    // makes worked out independently (tools/mpq_hash.py). Only the seven
    // characters it is made of are searched, so the whole thing is small -
    // but it is a real recovery: five characters in the table, three rewound
    // from the junction.
    JunctionRequest req;
    req.alphabet = "SETUPDA";
    req.suffix = "\\BWUNIN.PIF";
    req.minLength = 1;
    req.maxLength = 8;
    req.junctionA = {stateOf("SETUPDAT", kHashAOffset).first, stateOf("SETUPDAT", kHashAOffset).second};
    req.junctionB = {stateOf("SETUPDAT", kHashBOffset).first, stateOf("SETUPDAT", kHashBOffset).second};
    req.targetHashA = 0x5BAB63FE;
    req.targetHashB = 0x8DF2CB08;
    req.statusInterval = std::chrono::milliseconds(60000);
    req.progressInterval = std::chrono::milliseconds(60000);
    std::string error;
    check(junctionFitsHashes(req, g_table, error), "the junction of 'SETUPDAT' fits the hashes of 'SETUPDAT\\BWUNIN.PIF'");

    std::filesystem::remove_all(kDir);
    const Run got = run(backend, req, 5);
    check(got.result.ok && got.result.found && got.result.filename == "SETUPDAT\\BWUNIN.PIF",
          "an eight-character directory is recovered from its junction: " + got.result.filename);
    check(got.result.tableLength == 5 && got.result.tableEntries == 16807, "with the first five characters of it in the table");

    // The matches file ends with the name, and found.txt has it too.
    std::vector<std::string> lines;
    std::ifstream in(kDir + "/matches.txt");
    for (std::string line; std::getline(in, line);)
        lines.push_back(line);
    check(!lines.empty() && lines.back() == "SETUPDAT\\BWUNIN.PIF", "the matches file ends with it");
    std::ifstream found(kDir + "/found.txt");
    std::string foundLine;
    check(std::getline(found, foundLine) && foundLine.find("SETUPDAT\\BWUNIN.PIF") != std::string::npos, "and found.txt has it");

    // The progress file is this search's, and says it got to the end.
    JunctionProgress progress;
    bool exists = false;
    JunctionRequest asRun = req;
    asRun.tableLength = 5;
    check(readJunctionProgress(kDir + "/junction.txt", progress, exists, error) && exists &&
              progress.fingerprint == junctionFingerprint(asRun),
          "the progress file is this search's");

    // The directory shorter than the table: found while the table is built.
    checkEverySplit(backend, requestFor("SETUPDA", "", "SET", "\\BWUNIN.PIF", 1, 5), "'SET', shorter than most of the tables");
}

static void testStopAndResume(SearchBackend& backend) {
    printf("--- stopping and resuming ---\n");
    const JunctionRequest req = requestFor("AB-\\", "", "B-A\\", "\\X.Y", 1, 4);
    const int tableLength = 1;
    JunctionSpace space;
    std::string error;
    check(JunctionSpace::create(req.alphabet, req.minLength, req.maxLength, tableLength, space, error), "the space");
    const Run whole = run(backend, req, tableLength);

    // Stopped after the first call to the backend, then resumed from where
    // the progress file says it got to - as many times as it takes.
    std::vector<std::string> together;
    uint64_t from = 0;
    int rounds = 0;
    bool resumed = true, foundOnce = false;
    while (from < space.size() && rounds < 100) {
        ++rounds;
        std::atomic<bool> abort{false};
        JunctionRequest part = req;
        part.startNumber = from;
        // Stops as soon as the search has done anything at all.
        JunctionSearchHooks hooks;
        Run got;
        hooks.onPartialMatch = [&](const std::string& filename) { got.dirs.push_back(filename.substr(0, filename.size() - req.suffix.size())); };
        hooks.onProgress = [&](uint64_t) { abort.store(true); };
        part.tableLength = tableLength;
        part.outputFilePath = kDir + "/matches.txt";
        part.progressFilePath = kDir + "/junction.txt";
        got.result = runJunctionSearch(backend, part, &abort, hooks, nullptr);
        resumed = resumed && got.result.ok && got.result.nextNumber > from;
        together.insert(together.end(), got.dirs.begin(), got.dirs.end());
        foundOnce = foundOnce || got.result.found;
        if (got.result.found)
            break;
        // What a resume reads: the progress file's own number.
        JunctionProgress progress;
        bool exists = false;
        if (!readJunctionProgress(kDir + "/junction.txt", progress, exists, error) || !exists) {
            check(false, "the progress file after a stop: " + error);
            break;
        }
        resumed = resumed && progress.next == got.result.nextNumber;
        from = progress.next;
    }
    std::sort(together.begin(), together.end());
    check(rounds > 2, "the search was stopped and resumed " + std::to_string(rounds) + " times");
    check(resumed, "each piece ran, got somewhere, and saved where");
    check(foundOnce == whole.result.found, "the file turned up in one of them, as in one whole run");
    if (!whole.result.found)
        check(together == whole.dirs, "and together they reported what one whole run does");

    // Aborted outright: it says so, and doesn't claim to have finished.
    std::atomic<bool> abort{true};
    const Run aborted = run(backend, req, tableLength, &abort);
    check(aborted.result.ok && aborted.result.aborted && !aborted.result.found && aborted.result.nextNumber == 0,
          "an abort before the first call stops it where it started");
}

static void testPause(SearchBackend& backend) {
    printf("--- pausing ---\n");
    const JunctionRequest req = requestFor("AB-\\", "", "B-AB", "\\X.Y", 1, 4);
    std::atomic<bool> pause{true};
    std::atomic<bool> abort{false};
    // Held before the first call, then let go: it finishes as usual.
    std::thread letGo([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        pause.store(false);
    });
    const Run held = run(backend, req, 2, &abort, &pause);
    letGo.join();
    const Run plain = run(backend, req, 2);
    check(held.result.ok && held.result.found == plain.result.found && held.dirs == plain.dirs,
          "a search held paused and then let go reports what an unpaused one does");

    // Aborted while paused: it stops without waiting for the pause to end.
    pause.store(true);
    std::thread abortIt([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        abort.store(true);
    });
    const Run stopped = run(backend, req, 2, &abort, &pause);
    abortIt.join();
    pause.store(false);
    check(stopped.result.ok && stopped.result.aborted, "a paused search can be aborted");
}

static void testRefusals(SearchBackend& backend) {
    printf("--- what a search refuses ---\n");
    const JunctionRequest good = requestFor("AB-", "", "AB", "\\X.Y", 1, 3);
    check(run(backend, good, 1).result.ok, "a good request runs");

    JunctionRequest wrongJunction = good;
    wrongJunction.junctionA.first ^= 1;
    const Run wrong = run(backend, wrongJunction, 1);
    check(!wrong.result.ok && wrong.result.error.find("hashA") != std::string::npos,
          "a junction that doesn't fit the file's hashes is refused: " + wrong.result.error);

    JunctionRequest badAlphabet = good;
    badAlphabet.alphabet = "AAB";
    check(!run(backend, badAlphabet, 1).result.ok, "an alphabet with a character twice is refused");

    JunctionRequest tooLong = good;
    tooLong.maxLength = MAX_CANDIDATE_LEN + 1;
    check(!run(backend, tooLong, 1).result.ok, "a directory longer than MAX_CANDIDATE_LEN is refused");

    JunctionRequest longName = requestFor("AB-", std::string(MAX_FILENAME_LEN - 4, 'P'), "AB", "\\X.Y", 1, 3);
    check(!run(backend, longName, 1).result.ok, "a filename longer than MAX_FILENAME_LEN is refused");

    // A table longer than the longest directory is cut down to it, not
    // refused - and then the walk has only the empty tail to do.
    const Run clamped = run(backend, good, good.maxLength + 3);
    check(clamped.result.ok && clamped.result.tableLength == good.maxLength, "a table longer than the directory is cut down to it");

    // And the table's length left to the memory it may use.
    JunctionRequest byMemory = good;
    byMemory.tableBytes = 1024;
    Run sized = run(backend, byMemory, -1);
    check(sized.result.ok && sized.result.tableLength >= 1 && JunctionTable::bytesFor((int) good.alphabet.size(), sized.result.tableLength) <= 1024,
          "with no table length given, as much of one as the memory allows (" + std::to_string(sized.result.tableLength) + " characters)");
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
    std::unique_ptr<SearchBackend> backend = createJunctionBackend(backendName, error, &selfTestFailed);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return selfTestFailed ? 1 : 77;
    }
    prepareCryptTable(g_table);
    std::filesystem::create_directories(kDir);
    printf("backend %s, bits compared of each of the junction's two seeds: %d\n", backendName.c_str(), NAMEBREAK_JUNCTION_MATCH_BITS);

    testEverySplit(*backend);
    testRules(*backend);
    testRandom(*backend);
    testRanges(*backend);
    testFindsItAndWritesIt(*backend);
    testStopAndResume(*backend);
    testPause(*backend);
    testRefusals(*backend);

    printf("directories that match the junction, compared one by one: %llu\n", (unsigned long long) g_hitsCompared);
    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
