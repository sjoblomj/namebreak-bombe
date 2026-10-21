// Correctness test for the parts of a meet-in-the-middle search that don't
// need a backend (engine/meet_in_the_middle.h,
// engine/meet_in_the_middle_search.h): the hash run backwards, how the
// backward walk's tails are numbered, the walk itself, the table, the pruning
// rules split between the two halves, the meeting point's own consistency
// check, the fingerprint and the progress file - every one of them against
// brute force, or against the forward hash they have to agree with. Pure CPU.
//
// What could go wrong here is the kind of bug that loses a directory
// silently rather than crashing: an inverse step that's off by one
// multiplication, a tail whose index doesn't mean what the table's lookup
// assumes, a rule that skips a subtree it shouldn't. So nothing here is
// spot-checked - the small spaces are walked exhaustively, every tail and
// every directory, and compared with what a plain loop says they should be.
//
// Writes its files under ./meet_in_the_middle_test/ - ctest runs this from
// its own directory under build/testrun/.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "engine/meet_in_the_middle.h"
#include "engine/meet_in_the_middle_search.h"
#include "engine/mpq_hash.h"

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "meet_in_the_middle_test";

static uint32_t g_cryptTable[0x500];

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

// --- The hash, backwards -----------------------------------------------------

static void testStepBack() {
    printf("--- hashStepBack ---\n");
    std::mt19937_64 rng(1234);
    int mismatches = 0;
    for (int offset : {kHashAOffset, kHashBOffset, kFileKeyOffset}) {
        const uint32_t* table = g_cryptTable + offset;
        for (int round = 0; round < 40; ++round) {
            const HashState before{(uint32_t) rng(), (uint32_t) rng()};
            // Every character a byte can be, not just the printable ones -
            // a name may hold any of them.
            for (int ch = 0; ch < 256; ++ch) {
                const HashState after = hashStep(before, (unsigned char) ch, table);
                if (hashStepBack(after, (unsigned char) ch, table) != before)
                    ++mismatches;
            }
        }
    }
    check(mismatches == 0, "one step back undoes one step forward, for every character and hash");
    check(33u * kInv33 == 1u, "kInv33 really is 33's inverse modulo 2^32");

    // And over whole strings, from the state a prefix leaves.
    int stringMismatches = 0;
    for (const std::string& text : {std::string(""), std::string("A"), std::string("SETUPDAT\\GEN"), std::string("\xC9T\xE9"),
                                    std::string("a very long piece of name, 40-odd characters")}) {
        for (int offset : {kHashAOffset, kHashBOffset}) {
            const HashState from{0x7FED7FED, 0xEEEEEEEE};
            const HashState to = continueHash(from, text, offset, g_cryptTable);
            if (rewindHash(to, text, offset, g_cryptTable) != from)
                ++stringMismatches;
        }
    }
    check(stringMismatches == 0, "rewindHash undoes continueHash over a whole string");
}

// --- The space ---------------------------------------------------------------

static void testSpace() {
    printf("--- MeetInTheMiddleSpace ---\n");
    std::string error;
    MeetInTheMiddleSpace space;
    check(!MeetInTheMiddleSpace::create("", 1, 4, 2, space, error) && !error.empty(), "an empty alphabet is refused");
    check(!MeetInTheMiddleSpace::create("AAB", 1, 4, 2, space, error) && error.find("more than once") != std::string::npos,
          "an alphabet with a character twice is refused");
    check(!MeetInTheMiddleSpace::create("AB", 0, 4, 2, space, error), "a shortest directory of 0 characters is refused");
    check(!MeetInTheMiddleSpace::create("AB", 5, 4, 2, space, error), "a shortest longer than the longest is refused");
    check(!MeetInTheMiddleSpace::create("AB", 1, MAX_CANDIDATE_LEN + 1, 2, space, error), "a directory longer than MAX_CANDIDATE_LEN is refused");
    check(!MeetInTheMiddleSpace::create("AB", 1, 4, 5, space, error), "a table longer than the longest directory is refused");
    check(!MeetInTheMiddleSpace::create(std::string(MAX_ALPHABET_SIZE + 1, 'x'), 1, 4, 2, space, error), "too big an alphabet is refused");
    // 43 characters to the power of 12 is past 2^64; to the power of 11 isn't.
    const std::string big = " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_";
    check(big.size() == 43, "the real alphabet has 43 characters");
    check(!MeetInTheMiddleSpace::create(big, 1, 12, 0, space, error) && error.find("64 bits") != std::string::npos,
          "a tail space too big to number is refused");
    check(MeetInTheMiddleSpace::create(big, 1, 11, 0, space, error), "one that just fits isn't");

    const std::string alphabet = "AB-";
    for (int maxLength = 1; maxLength <= 5; ++maxLength) {
        for (int tableLength = 0; tableLength <= maxLength; ++tableLength) {
            if (!MeetInTheMiddleSpace::create(alphabet, 1, maxLength, tableLength, space, error)) {
                check(false, "create " + std::to_string(maxLength) + "/" + std::to_string(tableLength) + ": " + error);
                continue;
            }
            // Every tail of every length, once each, in length order - and
            // its number, length and index agreeing with each other.
            std::vector<std::string> wanted;
            for (int j = 0; j <= space.maxTailLength(); ++j) {
                for (const std::string& tail : allStrings(alphabet, j))
                    wanted.push_back(tail);
            }
            std::vector<std::string> got;
            bool roundTrips = true, ordered = true;
            for (uint64_t number = 0; number < space.size(); ++number) {
                const int length = space.tailLengthOf(number);
                const uint64_t index = space.indexOf(number);
                const std::string tail = space.tailOf(number);
                got.push_back(tail);
                uint64_t back = 0;
                std::string tailError;
                roundTrips = roundTrips && space.numberOf(length, index) == number && (int) tail.size() == length &&
                             index < space.countOf(length) && space.indexOfTail(tail, back, tailError) && back == index;
                // The directory's last character is the slowest to change:
                // within one length, the tails come in the order of the
                // reversed tail - in the alphabet's own order, which isn't
                // the characters' ASCII one ('-' comes last here).
                if (number > 0 && space.tailLengthOf(number - 1) == length) {
                    const std::string previous = space.tailOf(number - 1);
                    bool isBefore = false;
                    std::string orderError;
                    ordered = ordered &&
                              isBeforeInAlphabet(std::string(previous.rbegin(), previous.rend()), std::string(tail.rbegin(), tail.rend()),
                                                 alphabet, isBefore, orderError) &&
                              isBefore;
                }
            }
            std::sort(wanted.begin(), wanted.end());
            std::vector<std::string> sortedGot = got;
            std::sort(sortedGot.begin(), sortedGot.end());
            const std::string what = "maxLength " + std::to_string(maxLength) + ", table " + std::to_string(tableLength);
            check(sortedGot == wanted, what + ": every tail of up to " + std::to_string(space.maxTailLength()) + " characters, once each");
            check(roundTrips, what + ": a node's number, length, index and tail agree");
            check(ordered, what + ": the tails of a length come in the order of the reversed tail");
        }
    }
}

// --- The rules ---------------------------------------------------------------

// The rule sets the tests use: none, each rule on its own, and all of them.
static std::vector<PruneRules> ruleSets() {
    std::vector<PruneRules> sets(1); // none
    PruneRules r;
    r.symbolRuns = true;
    sets.push_back(r);
    r = PruneRules();
    r.unopenedBrackets = true;
    sets.push_back(r);
    r = PruneRules();
    r.adjacentBackslashes = true;
    sets.push_back(r);
    r = PruneRules();
    r.maxBackslashCount = 1;
    sets.push_back(r);
    r = PruneRules();
    r.minBackslashCount = 1;
    sets.push_back(r);
    r = PruneRules();
    r.symbolRuns = true;
    r.unopenedBrackets = true;
    r.adjacentBackslashes = true;
    r.maxBackslashCount = 2;
    r.minBackslashCount = 1;
    sets.push_back(r);
    return sets;
}

// Whether `rules` reject `tail` on its own, checked from the right as the
// walk does.
static bool tailRejected(const PruneRules& rules, const std::string& suffix, const std::string& tail) {
    TailPruneState state = tailPruneStart(suffix);
    for (size_t i = tail.size(); i-- > 0;) {
        if (!tailPruneStep(rules, state, tail[i]))
            return true;
    }
    return false;
}

// Whether `rules` reject a directory's first characters on their own,
// checked from the left as the table's build does.
static bool prefixRejected(const PruneRules& rules, const std::string& namePrefix, const std::string& dirPrefix) {
    PruneState state = dirPruneStart(namePrefix);
    for (char c : dirPrefix) {
        if (!pruneStep_CPU(rules, state, c))
            return true;
    }
    return false;
}

static void testRules() {
    printf("--- the rules, split between the two halves ---\n");
    // The property the whole mode rests on: whatever either half skips on
    // its own, the directory it was part of would have been rejected in
    // full anyway. Checked over every directory of up to four characters, at
    // every split between the table and the walk.
    const std::string alphabet = "A-\\()";
    int unsound = 0, prefixSkipped = 0, tailSkipped = 0;
    for (const PruneRules& rules : ruleSets()) {
        for (const std::string& namePrefix : {std::string(""), std::string("SETUPDAT\\"), std::string("("), std::string("X")}) {
            for (const std::string& suffix : {std::string("\\A.B"), std::string("A.B"), std::string("")}) {
                for (int length = 1; length <= 4; ++length) {
                    for (const std::string& dir : allStrings(alphabet, length)) {
                        const bool passes = dirPasses(rules, namePrefix, suffix, dir);
                        for (int split = 0; split <= length; ++split) {
                            const std::string head = dir.substr(0, split), tail = dir.substr(split);
                            if (prefixRejected(rules, namePrefix, head)) {
                                ++prefixSkipped;
                                unsound += passes;
                            }
                            if (tailRejected(rules, suffix, tail)) {
                                ++tailSkipped;
                                unsound += passes;
                            }
                        }
                    }
                }
            }
        }
    }
    check(unsound == 0, "nothing either half skips would have passed the rules in full");
    check(prefixSkipped > 0 && tailSkipped > 0, "and both halves do skip plenty (" + std::to_string(prefixSkipped) + " and " +
                                                    std::to_string(tailSkipped) + " of the splits tried)");

    // The rules themselves, on a few directories.
    PruneRules all;
    all.symbolRuns = true;
    all.unopenedBrackets = true;
    all.adjacentBackslashes = true;
    all.maxBackslashCount = 1;
    check(!dirPasses(all, "", "\\A.B", "A---B"), "three symbols in a row are rejected");
    check(dirPasses(all, "", "\\A.B", "A--B"), "two aren't");
    check(!dirPasses(all, "", "\\A.B", "AB\\"), "a '\\' at the end, next to the suffix's, is rejected");
    check(dirPasses(all, "", "A.B", "AB\\"), "but not when the suffix doesn't start with one");
    check(!dirPasses(all, "X\\", "A.B", "\\AB"), "nor is one at the start next to the prefix's");
    check(!dirPasses(all, "", "\\A.B", "A\\B\\C"), "more backslashes than max_backslash_count is rejected");
    check(!dirPasses(all, "", "\\A.B", "A)B"), "a bracket that was never opened is rejected");
    check(dirPasses(all, "(", "\\A.B", "A)B"), "unless the prefix opened it");
    PruneRules min;
    min.minBackslashCount = 1;
    check(!dirPasses(min, "", "\\A.B", "AB"), "fewer backslashes than min_backslash_count is rejected");
    check(dirPasses(min, "", "\\A.B", "A\\B"), "enough of them isn't");
}

// --- The walk ----------------------------------------------------------------

static void testWalk() {
    printf("--- MeetInTheMiddleWalk ---\n");
    const std::string alphabet = "AB-\\";
    const std::string suffix = "\\X.Y";
    const HashState meetingPoint = continueHash(kInitialHashState, "SETUPDAT\\SOMEDIR", kHashAOffset, g_cryptTable);
    int visited = 0;
    for (const PruneRules& rules : ruleSets()) {
        for (int tailLength = 0; tailLength <= 4; ++tailLength) {
            MeetInTheMiddleSpace space;
            std::string error;
            if (!MeetInTheMiddleSpace::create(alphabet, 1, tailLength + 1, 1, space, error)) {
                check(false, error);
                continue;
            }
            // What the walk must visit, and with what: every tail of this
            // length the rules allow, in index order, each with the meeting
            // point rewound over it from scratch.
            std::vector<std::pair<uint64_t, std::string>> wanted;
            for (uint64_t index = 0; index < space.countOf(tailLength); ++index) {
                const std::string tail = space.tailOf(space.numberOf(tailLength, index));
                if (!rules.any() || !tailRejected(rules, suffix, tail))
                    wanted.emplace_back(index, tail);
            }

            // In one go, and in every size of piece - a walk started part
            // way through must visit exactly the tails of its own range.
            for (uint64_t piece : {space.countOf(tailLength), (uint64_t) 1, (uint64_t) 2, (uint64_t) 3, (uint64_t) 5}) {
                std::vector<std::pair<uint64_t, std::string>> got;
                bool states = true;
                for (uint64_t from = 0; from < space.countOf(tailLength); from += piece) {
                    const uint64_t to = std::min(space.countOf(tailLength), from + piece);
                    MeetInTheMiddleWalk walk(alphabet, tailLength, meetingPoint, g_cryptTable + kHashAOffset, rules, tailPruneStart(suffix));
                    for (bool more = walk.start(from, to); more; more = walk.next()) {
                        got.emplace_back(walk.index(), walk.tail());
                        states = states && walk.state() == rewindHash(meetingPoint, walk.tail(), kHashAOffset, g_cryptTable);
                        ++visited;
                    }
                }
                const std::string what = "tails of " + std::to_string(tailLength) + " in pieces of " + std::to_string(piece);
                check(got == wanted, what + ": exactly the tails the rules allow, in index order");
                check(states, what + ": each with the meeting point rewound over it");
            }
            // A range that starts past the end has nothing in it.
            MeetInTheMiddleWalk walk(alphabet, tailLength, meetingPoint, g_cryptTable + kHashAOffset, rules, tailPruneStart(suffix));
            check(!walk.start(space.countOf(tailLength), space.countOf(tailLength) + 10), "a range past the last tail is empty");
            check(!walk.start(3, 3), "and so is an empty one");
        }
    }
    check(visited > 10000, "the walk was taken through plenty of tails (" + std::to_string(visited) + ")");
}

// --- The table ---------------------------------------------------------------

static void testTable() {
    printf("--- MeetInTheMiddleTable ---\n");
    const std::string alphabet = "AB-\\";
    const std::string namePrefix = "SETUPDAT\\";
    const std::string suffix = "\\X.Y";
    for (const PruneRules& rules : ruleSets()) {
        for (int tableLength = 0; tableLength <= 3; ++tableLength) {
            // The meeting point of a directory that's partly in the table and
            // partly in the walk, so that the shorter directories have
            // something to miss.
            const int maxLength = 4;
            MeetInTheMiddleSpace space;
            std::string error;
            if (!MeetInTheMiddleSpace::create(alphabet, 1, maxLength, tableLength, space, error)) {
                check(false, error);
                continue;
            }
            const HashState startState = continueHash(kInitialHashState, namePrefix, kHashAOffset, g_cryptTable);
            // A meeting point that some *short* directory reaches, to test that
            // the build reports those: the state after "AB".
            const HashState meetingPoint = continueHash(startState, "AB", kHashAOffset, g_cryptTable);

            std::vector<std::string> shortMatches;
            MeetInTheMiddleTable table;
            if (!MeetInTheMiddleTable::build(space, startState, dirPruneStart(namePrefix), rules, g_cryptTable, meetingPoint, shortMatches, table,
                                      error)) {
                check(false, error);
                continue;
            }

            // What it must hold: every directory of the table's length the
            // rules allow on their own, by the state it reaches.
            std::vector<std::string> wantedPrefixes;
            std::set<uint64_t> wantedKeys;
            for (const std::string& dir : allStrings(alphabet, tableLength)) {
                if (rules.any() && prefixRejected(rules, namePrefix, dir))
                    continue;
                wantedPrefixes.push_back(dir);
                wantedKeys.insert(meetingPointKey(continueHash(startState, dir, kHashAOffset, g_cryptTable)));
            }
            const std::string what = "table of " + std::to_string(tableLength);
            check(table.size() == wantedPrefixes.size(),
                  what + ": holds every directory of its length the rules allow (" + std::to_string(table.size()) + ")");

            // Every key it must hold, it holds - and says so through the
            // filter too; keys it mustn't, it doesn't.
            bool holdsAll = true, filtersAll = true;
            for (uint64_t key : wantedKeys)
                holdsAll = holdsAll && table.holds(key) && table.mayHold(key);
            std::mt19937_64 rng(99);
            int falseHolds = 0;
            for (int i = 0; i < 2000; ++i) {
                const uint64_t key = meetingPointKey((uint32_t) rng(), (uint32_t) rng());
                if (!wantedKeys.count(key) && table.holds(key))
                    ++falseHolds;
                // A key the table holds must get past the filter; one it
                // doesn't may or may not.
                if (table.holds(key) && !table.mayHold(key))
                    filtersAll = false;
            }
            check(holdsAll, what + ": holds every key it should");
            check(filtersAll && falseHolds == 0, what + ": and no others, filter and keys agreeing");

            // Looking a key back up gives exactly the directories that reach it.
            bool resolved = true;
            std::vector<uint64_t> keys;
            std::vector<std::vector<std::string>> wantedFor;
            for (const std::string& dir : wantedPrefixes) {
                const uint64_t key = meetingPointKey(continueHash(startState, dir, kHashAOffset, g_cryptTable));
                std::vector<std::string> mine;
                for (const std::string& other : wantedPrefixes) {
                    if (meetingPointKey(continueHash(startState, other, kHashAOffset, g_cryptTable)) == key)
                        mine.push_back(other);
                }
                keys.push_back(key);
                wantedFor.push_back(mine);
                std::vector<std::string> got;
                table.prefixesFor(key, got);
                resolved = resolved && got == mine;
            }
            check(resolved, what + ": every key looked back up gives the directories that reach it");
            std::vector<std::vector<std::string>> gotFor;
            table.prefixesFor(keys, gotFor);
            check(gotFor == wantedFor, what + ": and a batch of keys at once gives the same");

            // The shorter directories the build checks on the way: exactly
            // those shorter than the table that reach the meeting point.
            std::vector<std::string> wantedShort;
            for (int length = 1; length < tableLength; ++length) {
                for (const std::string& dir : allStrings(alphabet, length)) {
                    if (rules.any() && prefixRejected(rules, namePrefix, dir))
                        continue;
                    if (meetingPointKey(continueHash(startState, dir, kHashAOffset, g_cryptTable)) == meetingPointKey(meetingPoint))
                        wantedShort.push_back(dir);
                }
            }
            std::sort(shortMatches.begin(), shortMatches.end());
            std::sort(wantedShort.begin(), wantedShort.end());
            check(shortMatches == wantedShort, what + ": the directories shorter than it that reach the meeting point are reported (" +
                                                   std::to_string(wantedShort.size()) + ")");
            if (tableLength > 2 && !rules.any())
                check(!wantedShort.empty(), what + ": and there is one to report ('AB')");
        }
    }
}

// --- The table's size --------------------------------------------------------

static void testTableSize() {
    printf("--- how big a table fits ---\n");
    const int alphabetSize = 43;
    check(meetInTheMiddleTableCount(alphabetSize, 0) == 1 && meetInTheMiddleTableCount(alphabetSize, 4) == 3418801 &&
              meetInTheMiddleTableCount(alphabetSize, 16) == UINT64_MAX,
          "how many directories a table holds");
    check(MeetInTheMiddleTable::bytesFor(alphabetSize, 0) > 0, "even a table of nothing takes a little");
    bool grows = true;
    for (int length = 1; length <= 8; ++length)
        grows = grows && MeetInTheMiddleTable::bytesFor(alphabetSize, length) > MeetInTheMiddleTable::bytesFor(alphabetSize, length - 1);
    check(grows, "a character more needs more memory");
    check(MeetInTheMiddleTable::bytesFor(alphabetSize, 5) / (1024 * 1024) > 1000 &&
              MeetInTheMiddleTable::bytesFor(alphabetSize, 5) / (1024 * 1024) < 2000,
          "five characters of the real alphabet is between 1 and 2 GB");
    check(MeetInTheMiddleTable::bytesFor(alphabetSize, 16) == UINT64_MAX, "a table too big to count is UINT64_MAX");

    bool withinBudget = true, largest = true;
    for (uint64_t bytes : {(uint64_t) 0, (uint64_t) 1024, (uint64_t) 1 << 20, (uint64_t) 1 << 30, (uint64_t) 8 << 30}) {
        const int length = meetInTheMiddleTableLengthForBytes(alphabetSize, 12, bytes);
        withinBudget = withinBudget && (length == 0 || MeetInTheMiddleTable::bytesFor(alphabetSize, length) <= bytes);
        largest = largest && (length == 12 || MeetInTheMiddleTable::bytesFor(alphabetSize, length + 1) > bytes);
    }
    check(withinBudget, "the table length chosen for a budget fits in it");
    check(largest, "and is the largest that does");
    check(meetInTheMiddleTableLengthForBytes(alphabetSize, 3, (uint64_t) 8 << 30) == 3, "never longer than the longest directory");
}

// --- The meeting point's own consistency ------------------------------------------

static void testFitsHashes() {
    printf("--- meetingPointFitsHashes ---\n");
    MeetInTheMiddleRequest req;
    req.alphabet = "AB";
    req.prefix = "SETUPDAT\\";
    req.suffix = "\\BWUNIN.PIF";
    const std::string dir = "GEN";
    req.meetingPointA = continueHash(kInitialHashState, req.prefix + dir, kHashAOffset, g_cryptTable);
    req.meetingPointB = continueHash(kInitialHashState, req.prefix + dir, kHashBOffset, g_cryptTable);
    const std::string filename = req.prefix + dir + req.suffix;
    req.targetHashA = continueHash(kInitialHashState, filename, kHashAOffset, g_cryptTable).first;
    req.targetHashB = continueHash(kInitialHashState, filename, kHashBOffset, g_cryptTable).first;

    std::string error;
    check(meetingPointFitsHashes(req, g_cryptTable, error), "a meeting point and the hashes of a real name fit");
    MeetInTheMiddleRequest wrongA = req;
    wrongA.meetingPointA.second ^= 1;
    check(!meetingPointFitsHashes(wrongA, g_cryptTable, error) && error.find("hashA") != std::string::npos,
          "a meeting point that isn't hashA's is caught");
    MeetInTheMiddleRequest wrongB = req;
    wrongB.meetingPointB.first ^= 0x100;
    check(!meetingPointFitsHashes(wrongB, g_cryptTable, error) && error.find("hashB") != std::string::npos, "and so is hashB's");
    MeetInTheMiddleRequest wrongSuffix = req;
    wrongSuffix.suffix = "\\BWUNIN.EXE";
    check(!meetingPointFitsHashes(wrongSuffix, g_cryptTable, error), "a meeting point at the wrong place in the name is caught");

    // A known answer: the two hashes of "SETUPDAT\BWUNIN.PIF", worked out by
    // another implementation of the MPQ hash altogether, must fit the state
    // after the directory "SETUPDAT" - the same shape of thing a meeting point
    // search is given, with the hashes in it from something other than this
    // code.
    MeetInTheMiddleRequest known;
    known.suffix = "\\BWUNIN.PIF";
    known.meetingPointA = continueHash(kInitialHashState, "SETUPDAT", kHashAOffset, g_cryptTable);
    known.meetingPointB = continueHash(kInitialHashState, "SETUPDAT", kHashBOffset, g_cryptTable);
    known.targetHashA = 0x5BAB63FE;
    known.targetHashB = 0x8DF2CB08;
    check(meetingPointFitsHashes(known, g_cryptTable, error),
          "the state after 'SETUPDAT' fits the known hashes of 'SETUPDAT\\BWUNIN.PIF'");
}

// --- The fingerprint and the progress file -----------------------------------

static MeetInTheMiddleRequest sampleRequest() {
    MeetInTheMiddleRequest req;
    req.alphabet = "AB-";
    req.prefix = "SETUPDAT\\";
    req.suffix = "\\X.Y";
    req.meetingPointA = {0x11111111, 0x22222222};
    req.meetingPointB = {0x33333333, 0x44444444};
    req.targetHashA = 0x55555555;
    req.targetHashB = 0x66666666;
    req.minLength = 1;
    req.maxLength = 5;
    req.tableLength = 2;
    return req;
}

static void testFingerprint() {
    printf("--- meetInTheMiddleFingerprint ---\n");
    const MeetInTheMiddleRequest base = sampleRequest();
    const std::string fingerprint = meetInTheMiddleFingerprint(base);
    check(fingerprint.size() == 16 && fingerprint.find_first_not_of("0123456789abcdef") == std::string::npos, "16 hex digits");

    // What must change it.
    std::vector<std::pair<std::string, MeetInTheMiddleRequest>> changed;
    auto with = [&](const std::string& what, MeetInTheMiddleRequest req) { changed.emplace_back(what, req); };
    MeetInTheMiddleRequest r = base;
    r.alphabet = "AB\\";
    with("another alphabet", r);
    r = base;
    r.prefix = "SETUP\\";
    with("another prefix", r);
    r = base;
    r.suffix = "\\X.Z";
    with("another suffix", r);
    r = base;
    r.meetingPointA.first ^= 1;
    with("another meeting point for hashA", r);
    r = base;
    r.meetingPointB.second ^= 1;
    with("another meeting point for hashB", r);
    r = base;
    r.targetHashA ^= 1;
    with("another hashA", r);
    r = base;
    r.targetHashB ^= 1;
    with("another hashB", r);
    r = base;
    r.minLength = 2;
    with("another shortest directory", r);
    r = base;
    r.maxLength = 6;
    with("another longest directory", r);
    r = base;
    r.tableLength = 3;
    with("another table length", r);
    r = base;
    r.rules.symbolRuns = true;
    with("another rule", r);
    r = base;
    r.rules.maxBackslashCount = 2;
    with("another backslash count", r);
    for (const auto& one : changed)
        check(meetInTheMiddleFingerprint(one.second) != fingerprint, one.first + " makes another fingerprint");

    // What mustn't.
    r = base;
    r.startNumber = 1234;
    r.endNumber = 5678;
    r.outputFilePath = "elsewhere.txt";
    r.progressFilePath = "elsewhere-too.txt";
    r.tableBytes = 12345;
    r.statusInterval = std::chrono::milliseconds(1);
    check(meetInTheMiddleFingerprint(r) == fingerprint, "where it starts and stops, its files and its memory don't");
}

static void testProgressFile() {
    printf("--- the progress file ---\n");
    std::filesystem::create_directories(kDir);
    const std::string path = kDir + "/meet-in-the-middle.txt";
    std::filesystem::remove(path);

    MeetInTheMiddleProgress progress;
    bool exists = true;
    std::string error;
    check(readMeetInTheMiddleProgress(path, progress, exists, error) && !exists, "no file yet: no progress, no error");

    const std::string fingerprint = meetInTheMiddleFingerprint(sampleRequest());
    check(writeMeetInTheMiddleProgress(path, {fingerprint, 4242}, 99999, "??AB-", error), "written");
    MeetInTheMiddleProgress read;
    check(readMeetInTheMiddleProgress(path, read, exists, error) && exists && read.fingerprint == fingerprint && read.next == 4242,
          "and read back");

    std::ofstream(path, std::ios::trunc) << "fingerprint = " << fingerprint << "\n";
    check(!readMeetInTheMiddleProgress(path, read, exists, error) && error.find("'next'") != std::string::npos, "a file without `next` is refused");
    std::ofstream(path, std::ios::trunc) << "next = 1\n";
    check(!readMeetInTheMiddleProgress(path, read, exists, error) && error.find("'fingerprint'") != std::string::npos,
          "and one without a fingerprint");
    std::ofstream(path, std::ios::trunc) << "fingerprint = nothex\nnext = 1\n";
    check(!readMeetInTheMiddleProgress(path, read, exists, error), "and one whose fingerprint isn't hex");
    std::ofstream(path, std::ios::trunc) << "fingerprint = " << fingerprint << "\nnext = soon\n";
    check(!readMeetInTheMiddleProgress(path, read, exists, error), "and one whose `next` isn't a number");
    std::ofstream(path, std::ios::trunc) << "no equals sign here\n";
    check(!readMeetInTheMiddleProgress(path, read, exists, error), "and one that isn't key = value at all");
}

int main() {
    prepareCryptTable(g_cryptTable);

    testStepBack();
    testSpace();
    testRules();
    testWalk();
    testTable();
    testTableSize();
    testFitsHashes();
    testFingerprint();
    testProgressFile();

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
