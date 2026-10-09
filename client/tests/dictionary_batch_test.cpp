// Correctness test for what the backends that search dictionaries share
// (backends/common/dictionary_batch.h): the word table a GPU kernel reads,
// the suffix's keys, the suffix filters and their check, a launch's batches
// and segments, and turning what a backend found into a DictionaryOutcome -
// each against a plain recomputation here. Pure CPU, so that a machine
// without a GPU tests all of it too.

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backends/common/dictionary_batch.h"
#include "backends/common/lowbits_filter.h"
#include "engine/dictionary_search.h"
#include "engine/hash_match.h"
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

static uint32_t g_table[0x500];

// The MPQ hash of `s` from scratch, with the crypt table at `offset`.
static uint32_t hashOf(const std::string& s, int offset) {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : s) {
        seed1 = g_table[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}

// Word `entry`'s characters, read back from the table as a kernel does.
static std::string wordOf(const DictionaryWordTable& table, const DictionaryWordEntry& entry) {
    std::string word;
    for (uint32_t c = 0; c < entry.length; ++c)
        word += (char) (table.chars[entry.offset + c / 4] >> (8 * (c % 4)) & 0xFF);
    return word;
}

static void testWordTable() {
    printf("--- the word table ---\n");
    const std::vector<std::string> words = {"ABCDE", "B", "C\\D", "\\E", "F\\", "GHIJ", "\xC9T\xE9", "KL", "MNOPQRSTU", "VW"};
    DictionaryWordTable table;
    check(makeDictionaryWordTable(words, table), "made");
    check(table.entries.size() == words.size() && table.positions.size() == words.size(), "an entry and a position for every word");

    bool readBack = true, positions = true, starts = true;
    for (uint32_t w = 0; w < words.size(); ++w) {
        const DictionaryWordEntry& entry = table.entries[table.positions[w]];
        positions = positions && entry.index == w;
        readBack = readBack && wordOf(table, entry) == words[w];
        const size_t slash = words[w].rfind('\\');
        starts = starts && entry.basenameStart == (slash == std::string::npos ? 0 : slash + 1);
    }
    check(positions, "positions[w] is word w's entry");
    check(readBack, "every word's characters read back as they were, those past ASCII too");
    check(starts, "a word's basename starts after its last '\\', or at 0 if it has none");
    std::vector<uint32_t> lengths;
    for (const DictionaryWordEntry& entry : table.entries)
        lengths.push_back(entry.length);
    check(std::is_sorted(lengths.begin(), lengths.end()), "the entries in the order of the words' lengths");
    bool stable = true;
    for (size_t i = 1; i < table.entries.size(); ++i)
        stable = stable && (table.entries[i - 1].length < table.entries[i].length || table.entries[i - 1].index < table.entries[i].index);
    check(stable, "... and of the list among words of the same length");
    size_t uint32s = 0;
    for (const std::string& word : words)
        uint32s += (word.size() + 3) / 4;
    check(table.chars.size() == uint32s, "every word from a uint32 of its own: " + std::to_string(table.chars.size()) + " uint32s");
    bool padded = true;
    for (const DictionaryWordEntry& entry : table.entries) {
        if (entry.length % 4 != 0)
            padded = padded && (table.chars[entry.offset + entry.length / 4] >> (8 * (entry.length % 4))) == 0;
    }
    check(padded, "the rest of a word's last uint32 is zero");

    DictionaryWordTable empty;
    check(makeDictionaryWordTable({}, empty) && empty.chars.empty() && empty.entries.empty(), "no words: an empty table");
}

static void testSuffixKeys() {
    printf("--- the suffix's keys ---\n");
    const std::string suffix = ".W\xE9V";
    const std::vector<uint32_t> keys = dictionarySuffixKeys(suffix, g_table);
    bool ok = keys.size() == 3 * suffix.size();
    for (size_t i = 0; ok && i < suffix.size(); ++i) {
        const unsigned char ch = (unsigned char) suffix[i];
        ok = keys[3 * i] == g_table[0x100 + ch] && keys[3 * i + 1] == g_table[0x300 + ch] && keys[3 * i + 2] == ch + 3u;
    }
    check(ok, "hashA's key, the basename hash's and the character plus 3, for each character");
    // A step with them, its sums in another order, is the usual step.
    uint32_t a1 = 0x12345678, a2 = 0x9ABCDEF0, b1 = a1, b2 = a2;
    for (size_t i = 0; i < suffix.size(); ++i) {
        a1 = keys[3 * i] ^ (a1 + a2);
        a2 = a1 + 33 * a2 + keys[3 * i + 2];
    }
    const HashState usual = continueHash({b1, b2}, suffix, kHashAOffset, g_table);
    check(a1 == usual.first && a2 == usual.second, "a step with them gives the usual step's state");
    check(dictionarySuffixKeys("", g_table).empty(), "no suffix: no keys");
}

static void testSuffixFilters() {
    printf("--- the suffix filters ---\n");
    std::mt19937 rng(42);
    int checked = 0;
    bool built = true, exact = true, neverMissed = true;
    for (const std::string& suffix : {std::string(""), std::string(".WAV"), std::string("A LONGER SUFFIX\xE9")}) {
        for (bool basename : {false, true}) {
            const int offset = basename ? kFileKeyOffset : kHashAOffset;
            const uint32_t target = (uint32_t) rng(), mask = dictionaryFilterMask(basename);
            const std::vector<uint32_t> filter = buildDictionarySuffixFilter(suffix, g_table, offset, target, mask);
            std::string error;
            built = built && filter.size() == kDictionaryFilterWords && checkDictionarySuffixFilter(filter, suffix, g_table, offset, target, mask, 7, error);
            // From random states: a state whose suffix hash has the target's
            // bits always gets through - and its bit says exactly that.
            for (int i = 0; i < 2000; ++i) {
                const uint32_t s1 = (uint32_t) rng(), s2 = (uint32_t) rng();
                const uint32_t index = lowBitsFilterIndex(s1, s2, kDictionaryFilterBits);
                const bool passes = filter[index / 32] >> (index % 32) & 1;
                const uint32_t hash = continueHash({s1, s2}, suffix, offset, g_table).first;
                exact = exact && passes == ((hash & mask) == (target & mask));
                neverMissed = neverMissed && (passes || hash != target);
                ++checked;
            }
            // One with a match planted: its state.
            const uint32_t s1 = (uint32_t) rng(), s2 = (uint32_t) rng();
            const std::vector<uint32_t> planted =
                buildDictionarySuffixFilter(suffix, g_table, offset, continueHash({s1, s2}, suffix, offset, g_table).first, mask);
            const uint32_t index = lowBitsFilterIndex(s1, s2, kDictionaryFilterBits);
            neverMissed = neverMissed && (planted[index / 32] >> (index % 32) & 1);
        }
    }
    check(built, "built for no suffix, a short one and a long one, hashA's and the basename hash's - and each passes its check");
    check(exact, "a state's bit is set exactly when its suffix hash has the target's low bits (" + std::to_string(checked) + " random states)");
    check(neverMissed, "a state whose suffix hash is the target always gets through");
    check(dictionaryFilterMask(true) == lowBitsFilterStateMask(kDictionaryFilterBits), "the basename hash's filter compares all of its low bits");
    check(dictionaryFilterMask(false) == (lowBitsFilterStateMask(kDictionaryFilterBits) & kHashAMatchMask),
          "hashA's, those of them a hit must match");

    // The check catches a wrong filter: every bit flipped in turn (a sample).
    const std::vector<uint32_t> good = buildDictionarySuffixFilter(".WAV", g_table, kHashAOffset, 0x12345678, dictionaryFilterMask(false));
    int caught = 0, flipped = 0;
    for (uint32_t bit = 0; bit < 32 * kDictionaryFilterWords; bit += 1 + bit / 7) {
        std::vector<uint32_t> bad = good;
        bad[bit / 32] ^= 1u << (bit % 32);
        std::string error;
        ++flipped;
        if (!checkDictionarySuffixFilter(bad, ".WAV", g_table, kHashAOffset, 0x12345678, dictionaryFilterMask(false), 7, error) &&
            error.find("bit " + std::to_string(bit)) != std::string::npos)
            ++caught;
    }
    check(caught == flipped, "a filter with any one bit wrong fails its check, naming the bit (" + std::to_string(caught) + " of " +
                                 std::to_string(flipped) + ")");
    std::vector<uint32_t> shorter = good;
    shorter.pop_back();
    std::string error;
    check(!checkDictionarySuffixFilter(shorter, ".WAV", g_table, kHashAOffset, 0x12345678, dictionaryFilterMask(false), 7, error),
          "a filter of the wrong size fails it");
}

static DictionaryBatch batchOf(const std::string& leading, uint32_t firstWord, uint32_t wordCount) {
    DictionaryBatch b;
    b.leading = leading;
    const HashState a = continueHash(kInitialHashState, leading, kHashAOffset, g_table);
    const HashState k = continueBasenameHash(kInitialHashState, leading, g_table);
    b.seed1 = a.first;
    b.seed2 = a.second;
    b.basenameSeed1 = k.first;
    b.basenameSeed2 = k.second;
    b.firstWord = firstWord;
    b.wordCount = wordCount;
    return b;
}

static void testLaunchPlan() {
    printf("--- a launch's segments ---\n");
    const std::vector<DictionaryBatch> batches = {batchOf("A", 0, 10), batchOf("B\\", 3, 1), batchOf("", 0, 24), batchOf("C", 7, 25)};
    std::vector<DictionaryLaunchBatch> launch;
    const uint64_t segments = planDictionaryLaunch(batches, 8, launch);
    // 10 words: 2 segments; 1: 1; 24: 3; 25: 4.
    check(segments == 10, "segments of 8 words: 2 + 1 + 3 + 4 = " + std::to_string(segments));
    bool copied = launch.size() == batches.size();
    for (size_t b = 0; copied && b < batches.size(); ++b) {
        copied = launch[b].seed1 == batches[b].seed1 && launch[b].seed2 == batches[b].seed2 &&
                 launch[b].basenameSeed1 == batches[b].basenameSeed1 && launch[b].basenameSeed2 == batches[b].basenameSeed2 &&
                 launch[b].firstWord == batches[b].firstWord && launch[b].wordCount == batches[b].wordCount;
    }
    check(copied, "each batch's states and words as they are");
    check(launch[0].firstSegment == 0 && launch[1].firstSegment == 2 && launch[2].firstSegment == 3 && launch[3].firstSegment == 6,
          "each batch's first segment: 0, 2, 3, 6");
    const uint32_t expected[] = {0, 0, 1, 2, 2, 2, 3, 3, 3, 3};
    bool found = true;
    for (uint64_t s = 0; s < segments; ++s)
        found = found && dictionaryBatchOfSegment(launch, s) == expected[s];
    check(found, "every segment's batch found by its first segment");
    check(planDictionaryLaunch(batches, 1000, launch) == 4, "segments larger than any batch: one each");
    check(planDictionaryLaunch({}, 8, launch) == 0 && launch.empty(), "no batches: nothing");
    check(dictionaryBatchOfSegment({launch.empty() ? DictionaryLaunchBatch{} : launch[0]}, 0) == 0, "one batch: it");
}

static void testVerifier() {
    printf("--- checking hits ---\n");
    const std::vector<std::string> words = {"CRDT", "LST", "MAP", "X\\LST"};
    const std::vector<DictionaryBatch> batches = {batchOf("REZ\\", 0, 4), batchOf("REZ\\CRDT_", 0, 4), batchOf("SCRIPTS\\CRDT_", 1, 1)};
    DictionaryConstants constants;
    constants.words = words;
    constants.suffix = ".TXT";
    constants.cryptTable = g_table;
    constants.targetHashA = hashOf("REZ\\CRDT_LST.TXT", 0x100);
    constants.targetHashB = hashOf("REZ\\CRDT_LST.TXT", 0x200);
    constants.checkBasename = true;
    constants.basenameKey = hashOf("CRDT_LST.TXT", 0x300);
    DictionaryHitVerifier verifier;
    verifier.begin(constants);
    check(verifier.candidatesHaveBasenames(), "checking basenames, a suffix without a '\\': a backend hashes each candidate's");

    // The backend reports basename hits alone; their hashA and hashB are
    // checked here.
    DictionaryOutcome outcome;
    verifier.addHits(batches, {}, {{1, 1}, {2, 1}, {0, 3}}, outcome);
    check(outcome.basenameHits == std::vector<std::string>{"REZ\\CRDT_LST.TXT", "SCRIPTS\\CRDT_LST.TXT"},
          "the basename hits that match kept, in any directory - one that doesn't dropped");
    check(outcome.hits == std::vector<std::string>{"REZ\\CRDT_LST.TXT"}, "... the one matching hashA a hashA hit too, the others not");
    check(outcome.found && outcome.foundFilename == "REZ\\CRDT_LST.TXT", "... and found, matching hashB too");

    // A basename hit matching hashA but not hashB: a hashA hit, not found.
    constants.targetHashB = 0;
    verifier.begin(constants);
    outcome = DictionaryOutcome();
    verifier.addHits(batches, {}, {{1, 1}}, outcome);
    check(outcome.hits.size() == 1 && !outcome.found, "a hit that doesn't match hashB: reported, not found");

    // Not checking basenames: the backend reports hashA hits, and basename
    // hits are ignored.
    constants.checkBasename = false;
    constants.targetHashB = hashOf("REZ\\CRDT_LST.TXT", 0x200);
    verifier.begin(constants);
    check(!verifier.candidatesHaveBasenames(), "not checking basenames: a backend hashes no candidate's");
    outcome = DictionaryOutcome();
    verifier.addHits(batches, {{1, 1}, {0, 2}}, {{1, 1}}, outcome);
    check(outcome.hits == std::vector<std::string>{"REZ\\CRDT_LST.TXT"}, "a hashA hit that matches kept, one that doesn't dropped");
    check(outcome.found && outcome.foundFilename == "REZ\\CRDT_LST.TXT", "... and found, matching hashB too");
    check(outcome.basenameHits.empty(), "... and any basename hits it reports are ignored");

    // A suffix with a '\': every candidate's basename is its end - checked
    // here, once a call.
    constants.checkBasename = true;
    constants.suffix = "\\CRDT_LST.TXT";
    verifier.begin(constants);
    check(!verifier.candidatesHaveBasenames(), "a suffix with a '\\': a backend hashes no candidate's basename");
    outcome = DictionaryOutcome();
    verifier.addHits(batches, {}, {}, outcome);
    check(outcome.basenameHits == std::vector<std::string>{"REZ\\CRDT\\CRDT_LST.TXT"}, "... the call's first candidate reported, as its basename matches");
    outcome = DictionaryOutcome();
    verifier.addHits({}, {}, {}, outcome);
    check(outcome.basenameHits.empty(), "... nothing for a call without batches");
    constants.suffix = "\\OTHER.TXT";
    verifier.begin(constants);
    outcome = DictionaryOutcome();
    verifier.addHits(batches, {}, {}, outcome);
    check(outcome.basenameHits.empty(), "... and nothing if it doesn't match");
}

int main() {
    prepareCryptTable(g_table);
    testWordTable();
    testSuffixKeys();
    testSuffixFilters();
    testLaunchPlan();
    testVerifier();
    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
