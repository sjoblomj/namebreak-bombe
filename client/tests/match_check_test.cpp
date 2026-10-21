// What a match of both hashes is checked against before it's taken for the
// file (engine/match_check.h): the file's encryption key, and where its
// entry is in the archive's hash table - with the name that set this off,
// REZ\SFZSSP050WSRL.WAV, which matched the Hash A and Hash B of a file in
// the Brazilian Brood War installer's archive and is nothing like its name.
// And the false_matches.txt a search appends such a name to
// (MatchWriter::falseMatch). CPU only.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "engine/match_check.h"
#include "engine/match_writer.h"
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

// The name, and what's known about the file whose hashes it matched (worked
// out from bw-ptbr-install.exe with scripts/basename_checker.py and its hash
// table): the file's key, and its entry in slot 851 of 2048, with a slot
// never used at 848 - so lookups starting at 849 to 851 reach it.
static const char* kName = "REZ\\SFZSSP050WSRL.WAV";
static const uint32_t kNameKey = 0x38D3F662;
static const uint32_t kNameSlot = 1786;
static const uint32_t kFileKey = 0x4A8FA566;

static MatchCheck keyCheck(uint32_t key) {
    MatchCheck check;
    check.checkKey = true;
    check.key = key;
    return check;
}

static MatchCheck slotCheck(uint32_t tableSize, uint32_t first, uint32_t last) {
    MatchCheck check;
    check.checkSlot = true;
    check.hashTableSize = tableSize;
    check.firstStartSlot = first;
    check.lastStartSlot = last;
    return check;
}

static void testChecks(const uint32_t* table) {
    printf("--- the checks ---\n");
    check(hashTableStartSlot(kName, 2048, table) == kNameSlot, "the name's lookup starts at slot 1786 of 2048");
    check(hashTableStartSlot(kName, 1, table) == 0, "... and at 0 of a table of one");

    check(falseMatchReason(MatchCheck(), kName, table).empty(), "nothing known: every match of both hashes passes");
    check(falseMatchReason(keyCheck(kNameKey), kName, table).empty(), "the name's own key: passes");
    check(falseMatchReason(keyCheck(kFileKey), kName, table) == "its basename's key is 0x38D3F662, not the file's 0x4A8FA566",
          "the file's key: a false match, and why");
    check(falseMatchReason(keyCheck(kNameKey), "OTHER\\DIR\\SFZSSP050WSRL.WAV", table).empty(), "the key is the basename's, whatever the directory");
    check(!falseMatchReason(keyCheck(kNameKey), "SFZSSP050WSRL.WAV\\X", table).empty(), "... what follows the last '\\'");
    check(falseMatchReason(keyCheck(kNameKey), "SFZSSP050WSRL.WAV", table).empty(), "... all of it, without one");

    check(falseMatchReason(slotCheck(2048, 849, 851), kName, table) == "its lookup starts at hash table slot 1786, not at one from 849 to 851 of 2048",
          "the file's slots: a false match, and why");
    check(falseMatchReason(slotCheck(2048, 1786, 1786), kName, table).empty(), "its own slot alone: passes");
    check(falseMatchReason(slotCheck(2048, 1700, 1786), kName, table).empty(), "... as the last of several");
    check(falseMatchReason(slotCheck(2048, 1786, 1800), kName, table).empty(), "... as the first");
    check(!falseMatchReason(slotCheck(2048, 1787, 1800), kName, table).empty(), "just after it: fails");
    check(!falseMatchReason(slotCheck(2048, 1700, 1785), kName, table).empty(), "just before it: fails");
    check(falseMatchReason(slotCheck(2048, 1780, 3), kName, table).empty(), "slots wrapping past the end, it among the last");
    check(falseMatchReason(slotCheck(2048, 2040, 1786), kName, table).empty(), "... among the first");
    check(!falseMatchReason(slotCheck(2048, 1787, 1785), kName, table).empty(), "... all but it");

    MatchCheck both = keyCheck(kFileKey);
    both.checkSlot = true;
    both.hashTableSize = 2048;
    both.firstStartSlot = 849;
    both.lastStartSlot = 851;
    check(falseMatchReason(both, kName, table) ==
              "its basename's key is 0x38D3F662, not the file's 0x4A8FA566, and its lookup starts at hash table slot 1786, not at one from 849 to 851 of 2048",
          "both: both reasons");
    both.key = kNameKey;
    check(falseMatchReason(both, kName, table).rfind("its lookup starts", 0) == 0, "the key right, the slots wrong: the slots' reason");
}

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static void testFalseMatchesFile() {
    printf("--- false_matches.txt ---\n");
    const std::string matchesPath = "match_check_matches.txt";
    const std::string falsePath = falseMatchesFilePath(matchesPath);
    std::remove(matchesPath.c_str());
    std::remove(falsePath.c_str());
    check(falsePath == "false_matches.txt", "next to the matches file");
    {
        MatchWriter writer;
        std::string error;
        check(writer.open(matchesPath, error), "the matches file opens");
        writer.falseMatch({kName, "its basename's key is wrong"});
        writer.falseMatch({"REZ\\B.WAV", "and so is this one's"});
    }
    {
        MatchWriter writer;
        std::string error;
        writer.open(matchesPath, error);
        writer.falseMatch({"REZ\\C.WAV", "a later search's"});
    }
#ifdef _WIN32
    const char* eol = "\r\n";
#else
    const char* eol = "\n";
#endif
    check(readFile(falsePath) == std::string(kName) + "\tits basename's key is wrong" + eol + "REZ\\B.WAV\tand so is this one's" + eol +
                                     "REZ\\C.WAV\ta later search's" + eol,
          "a line each - the name, a tab and why - appended to, never replaced");
    check(readFile(matchesPath).empty(), "the matches file itself left alone");
    std::remove(matchesPath.c_str());
    std::remove(falsePath.c_str());
}

int main() {
    uint32_t table[0x500];
    prepareCryptTable(table);
    testChecks(table);
    testFalseMatchesFile();
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
