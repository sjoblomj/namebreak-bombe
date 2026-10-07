// Correctness test for the word lists a dictionary search reads
// (engine/wordlist.h): how a list's lines become words, the compiled-in
// english-1 - pinned, so that it can never change by accident - and the
// checksums. Pure CPU.
//
// Writes its files under ./wordlist_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "engine/wordlist.h"

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "wordlist_test";

static std::vector<std::string> parse(const std::string& text, std::vector<std::string>* warnings = nullptr) {
    std::vector<std::string> words, ignored;
    parseWordList(text, "list.txt", words, warnings ? *warnings : ignored);
    return words;
}

static std::string joined(const std::vector<std::string>& words) {
    std::string out;
    for (const std::string& word : words)
        out += (out.empty() ? "" : "|") + word;
    return out;
}

int main() {
    printf("--- normalizeMpqName ---\n");
    check(normalizeMpqName("rez/Crdt_lst.txt") == "REZ\\CRDT_LST.TXT", "lowercase made uppercase, '/' made '\\'");
    check(normalizeMpqName("AZ az 09 ~!@[]`{}") == "AZ AZ 09 ~!@[]`{}", "nothing else changes (not even '`' and '{', next to the letters)");
    check(normalizeMpqName(std::string("\xE9\xC9", 2)) == std::string("\xE9\xC9", 2), "bytes above 0x7F stay as they are");
    check(normalizeMpqName("") == "", "empty");

    printf("--- parseWordList ---\n");
    check(joined(parse("zerg\nprotoss\nterran\n")) == "ZERG|PROTOSS|TERRAN", "one word per line, normalized, in file order");
    check(joined(parse("zerg\r\nprotoss\r\n")) == "ZERG|PROTOSS", "CRLF");
    check(joined(parse("zerg\nprotoss")) == "ZERG|PROTOSS", "no line break after the last word");
    check(joined(parse("  zerg \t\n\tprotoss  \r\n")) == "ZERG|PROTOSS", "spaces and tabs around a word dropped");
    check(joined(parse("long beach\n")) == "LONG BEACH", "a space inside a word kept");
    check(joined(parse("\n   \n\t\r\nzerg\n\n")) == "ZERG", "blank lines skipped");
    check(joined(parse("# comment\n  # indented comment\nzerg\nmy#word\n")) == "ZERG|MY#WORD", "'#' lines skipped, a '#' later kept");
    check(joined(parse("sound\\zerg\nart/x\n")) == "SOUND\\ZERG|ART\\X", "backslashes kept, slashes made backslashes");
    check(joined(parse("b\na\nb\n")) == "B|A|B", "duplicates and order kept - sortedUniqueWords sorts them out");
    check(parse("").empty(), "empty text: no words");
    {
        std::vector<std::string> warnings;
        std::vector<std::string> words = parse("ok\ncaf\xC3\xA9\nta\tb\nbell\x07\nfine\n", &warnings);
        check(joined(words) == "OK|FINE", "a word with a non-printable or non-ASCII byte skipped");
        check(warnings.size() == 3, "... with one warning each");
        check(warnings.size() == 3 && warnings[0].rfind("list.txt:2:", 0) == 0 && warnings[1].rfind("list.txt:3:", 0) == 0 &&
                  warnings[2].rfind("list.txt:4:", 0) == 0,
              "... naming the list and the line");
    }
    check(joined(parse(" ~\n")) == "~", "'~' is printable");

    printf("--- loadWordListFile ---\n");
    {
        std::filesystem::create_directories(kDir);
        {
            std::ofstream out(kDir + "/words.txt", std::ios::binary);
            out << "zerg\r\nprotoss\n";
        }
        std::vector<std::string> words{"EXISTING"}, warnings;
        std::string error;
        check(loadWordListFile(kDir + "/words.txt", words, warnings, error) && joined(words) == "EXISTING|ZERG|PROTOSS",
              "a file's words appended to those already there");
        check(!loadWordListFile(kDir + "/missing.txt", words, warnings, error) && error.find("missing.txt") != std::string::npos,
              "a missing file: false, with an error naming it");
    }

    printf("--- sortedUniqueWords ---\n");
    check(joined(sortedUniqueWords({"B", "A", "B", "A_", "A\\", "A"})) == "A|A\\|A_|B", "sorted byte by byte, duplicates dropped");
    check(sortedUniqueWords({}).empty(), "empty");

    printf("--- checksums ---\n");
    check(fnv1a64("") == 0xCBF29CE484222325ull, "FNV-1a 64 of nothing: its offset basis");
    check(fnv1a64("a") == 0xAF63DC4C8601EC8Cull, "FNV-1a 64 of 'a': the published value");
    check(fnv1a64("b", fnv1a64("a")) == fnv1a64("ab"), "continued over two pieces: as over both at once");
    check(wordListChecksum({"A", "B"}) == 0x76FCA88CB6D9040Eull, "wordListChecksum of A, B (computed independently)");
    check(wordListChecksum({"A", "B"}) != wordListChecksum({"B", "A"}), "order matters");
    check(wordListChecksum({"AB"}) != wordListChecksum({"A", "B"}), "where the words split matters");
    check(hex64(0x0123456789ABCDEFull) == "0123456789abcdef", "hex64: 16 lowercase digits");
    check(hex64(0) == "0000000000000000", "hex64 of 0: padded");

    printf("--- the compiled-in dictionary ---\n");
    check(isBuiltinWordList("english-1") && kEnglish1 == std::string("english-1"), "english-1 is compiled in");
    check(!isBuiltinWordList("english-2") && !isBuiltinWordList("") && !isBuiltinWordList("none"), "nothing else is");
    check(builtinWordList("english-2").empty(), "an unknown name: no words");
    {
        const std::vector<std::string> words = builtinWordList(kEnglish1);
        const std::vector<std::string> sorted = sortedUniqueWords(words);
        // Pinned: english-1 never changes - a different list is english-2.
        // These values were computed from data/english-1.txt independently
        // of this code.
        check(words.size() == 63875, "english-1 has 63875 words (has " + std::to_string(words.size()) + ")");
        check(sorted == words, "... already sorted and without duplicates");
        check(wordListChecksum(sorted) == 0x63B352823C6059B0ull, "... with checksum 63b352823c6059b0 (has " + hex64(wordListChecksum(sorted)) + ")");
        check(!words.empty() && words.front() == "A" && words.back() == "ZYGOTES", "... from A to ZYGOTES");
        bool lettersOnly = true;
        for (const std::string& word : words)
            lettersOnly = lettersOnly && !word.empty() && word.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == std::string::npos;
        check(lettersOnly, "... every one of them A-Z only");
    }

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
