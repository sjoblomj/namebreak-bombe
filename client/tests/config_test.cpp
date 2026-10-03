// Correctness test for where buildSearchRequest (common/config.h) starts a
// search: start_candidate, which is optional, and resume_from_last_candidate,
// which may move the start to the last line of the matches file. Pure CPU.
//
// Writes matches files under ./config_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "common/config.h"

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "config_test";

// A [search] config for prefix "REZ\", suffix ".WAV", bounds "A".."ZZZZ".
static ConfigFile makeConfig(const std::string& startCandidate, const std::string& resume) {
    ConfigFile config;
    config.matchesDir = kDir;
    config.search = {
        {"alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}, {"max_backslash_count", "0"}, {"prefix", "REZ\\"},  {"suffix", ".WAV"},
        {"lower_bound", "REZ\\A.WAV"},              {"upper_bound", "REZ\\ZZZZ.WAV"}, {"hash_a", "0x1"}, {"hash_b", "0x2"},
    };
    if (!startCandidate.empty())
        config.search["start_candidate"] = startCandidate;
    if (!resume.empty())
        config.search["resume_from_last_candidate"] = resume;
    return config;
}

static void writeMatches(const std::string& lastLine) {
    std::filesystem::create_directories(kDir);
    std::ofstream out(kDir + "/matches.txt", std::ios::trunc);
    out << "REZ\\AAA.WAV\n" << lastLine << "\n";
}

static void removeMatches() {
    std::filesystem::remove(kDir + "/matches.txt");
}

// The start candidate buildSearchRequest picks, or "<error: ...>".
static std::string startOf(const ConfigFile& config, bool continuous) {
    SearchRequest req;
    std::string error;
    if (!buildSearchRequest(config, continuous, req, error))
        return "<error: " + error + ">";
    return req.startCandidate;
}

int main() {
    printf("--- without resume_from_last_candidate ---\n");
    writeMatches("REZ\\QQQ.WAV"); // must be ignored
    check(startOf(makeConfig("REZ\\MMM.WAV", ""), true) == "MMM", "start_candidate given: starts there");
    check(startOf(makeConfig("", ""), true) == "", "no start_candidate, continuous: from the beginning (empty candidate)");
    check(startOf(makeConfig("", ""), false) == "A", "no start_candidate, bounded: from the lower bound");
    check(startOf(makeConfig("REZ\\.WAV", ""), true) == "", "start_candidate = prefix + suffix: from the beginning");
    check(startOf(makeConfig("REZ\\MMM.WAV", "false"), true) == "MMM", "resume_from_last_candidate = false: start_candidate");

    printf("--- resume_from_last_candidate, no start_candidate ---\n");
    writeMatches("REZ\\QQQ.WAV");
    check(startOf(makeConfig("", "true"), true) == "QQQ", "resumes from the last match");
    check(startOf(makeConfig("", "true"), false) == "QQQ", "... in bounded mode too");
    removeMatches();
    check(startOf(makeConfig("", "true"), true) == "", "no matches file: from the beginning");
    writeMatches("");
    check(startOf(makeConfig("", "true"), true) == "", "matches file ending in a blank line: from the beginning");

    printf("--- resume_from_last_candidate, with start_candidate ---\n");
    writeMatches("REZ\\QQQ.WAV");
    check(startOf(makeConfig("REZ\\MMM.WAV", "true"), true) == "QQQ", "last match after start_candidate: the last match");
    check(startOf(makeConfig("REZ\\TTT.WAV", "true"), true) == "TTT", "last match before start_candidate: start_candidate");
    check(startOf(makeConfig("REZ\\QQQ.WAV", "true"), true) == "QQQ", "last match equal to start_candidate: either, the same");
    writeMatches("REZ\\AAAAA.WAV");
    check(startOf(makeConfig("REZ\\ZZZZ.WAV", "true"), true) == "AAAAA", "a longer last match comes after (searched later)");
    writeMatches("REZ\\ZZ.WAV");
    check(startOf(makeConfig("REZ\\AAAA.WAV", "true"), true) == "AAAA", "a shorter last match comes before");
    writeMatches("ART\\QQQ.PCX");
    check(startOf(makeConfig("REZ\\MMM.WAV", "true"), true) == "MMM", "last match from another search (other prefix/suffix): ignored");
    writeMatches("REZ\\qqq.WAV");
    check(startOf(makeConfig("REZ\\MMM.WAV", "true"), true) == "MMM", "last match with characters outside the alphabet: ignored");

    printf("--- invalid values ---\n");
    check(startOf(makeConfig("REZ\\MMM.PCX", ""), true).rfind("<error:", 0) == 0, "start_candidate with the wrong suffix: an error");
    check(startOf(makeConfig("", "sometimes"), true).rfind("<error:", 0) == 0, "resume_from_last_candidate not a boolean: an error");

    printf("--- prune_whole_candidate ---\n");
    auto wholeOf = [](const std::string& value) {
        ConfigFile config = makeConfig("", "");
        if (!value.empty())
            config.search["prune_whole_candidate"] = value;
        SearchRequest req;
        std::string error;
        if (!buildSearchRequest(config, true, req, error))
            return std::string("<error: ") + error + ">";
        return std::string(req.pruneWholeCandidate ? "true" : "false");
    };
    check(wholeOf("") == "false", "not given: false (leading characters only)");
    check(wholeOf("true") == "true", "true: every character but the last");
    check(wholeOf("false") == "false", "false: leading characters only");
    check(wholeOf("sometimes").rfind("<error:", 0) == 0, "not a boolean: an error");

    printf("--- min_backslash_count and prune_adjacent_backslashes ---\n");
    auto backslashRulesOf = [](const std::string& minCount, const std::string& adjacent) {
        ConfigFile config = makeConfig("", "");
        if (!minCount.empty())
            config.search["min_backslash_count"] = minCount;
        if (!adjacent.empty())
            config.search["prune_adjacent_backslashes"] = adjacent;
        SearchRequest req;
        std::string error;
        if (!buildSearchRequest(config, true, req, error))
            return std::string("<error: ") + error + ">";
        return std::to_string(req.minBackslashCount) + (req.pruneAdjacentBackslashes ? " adjacent" : "");
    };
    check(backslashRulesOf("", "") == "0", "not given: no minimum, adjacent backslashes allowed");
    check(backslashRulesOf("2", "true") == "2 adjacent", "given: read");
    check(backslashRulesOf("lots", "").rfind("<error:", 0) == 0, "min_backslash_count not a number: an error");
    check(backslashRulesOf("", "sometimes").rfind("<error:", 0) == 0, "prune_adjacent_backslashes not a boolean: an error");

    printf("--- insert_from_start and insert_from_end ---\n");
    auto insertionsOf = [](const std::string& fromStart, const std::string& fromEnd) {
        ConfigFile config = makeConfig("", "");
        if (!fromStart.empty())
            config.search["insert_from_start"] = fromStart;
        if (!fromEnd.empty())
            config.search["insert_from_end"] = fromEnd;
        SearchRequest req;
        std::string error;
        if (!buildSearchRequest(config, true, req, error))
            return std::string("<error: ") + error + ">";
        return "[" + req.insertFromStart.text + "]@" + std::to_string(req.insertFromStart.position) + " [" + req.insertFromEnd.text + "]@" +
               std::to_string(req.insertFromEnd.position);
    };
    check(insertionsOf("", "") == "[]@0 []@0", "not given: nothing inserted");
    check(insertionsOf("\\, 3", "_X_,0") == "[\\]@3 [_X_]@0", "text, position");
    check(insertionsOf("A,B, 2", "") == "[A,B]@2 []@0", "the last comma separates");
    check(insertionsOf("\" A \", 1", "") == "[ A ]@1 []@0", "quoted text keeps its spaces");
    check(insertionsOf("\\", "").rfind("<error:", 0) == 0, "no position: an error");
    check(insertionsOf("\\, -1", "").rfind("<error:", 0) == 0, "a negative position: an error");
    check(insertionsOf(", 3", "").rfind("<error:", 0) == 0, "no text: an error");

    // The start candidate and the last match are filenames with the text;
    // the bounds are without it.
    {
        ConfigFile config = makeConfig("REZ\\M\\MM.WAV", "");
        config.search["insert_from_start"] = "\\, 1";
        check(startOf(config, true) == "MMM", "start_candidate with the inserted text: the candidate without it");
        config.search["start_candidate"] = "REZ\\MMM.WAV";
        check(startOf(config, true).rfind("<error:", 0) == 0, "start_candidate without the text where it goes: an error");
        config.search["start_candidate"] = "REZ\\M\\MM.WAV";
        config.search["resume_from_last_candidate"] = "true";
        writeMatches("REZ\\Q\\QQ.WAV");
        check(startOf(config, true) == "QQQ", "resumes from the last match, without the inserted text");
    }

    printf("--- insertIntoCandidate and removeInsertions ---\n");
    {
        const Insertion fromStart{"(S", 2}, fromEnd{"E)", 1};
        check(insertIntoCandidate("ABCD", fromStart, fromEnd) == "AB(SCE)D", "both inside");
        check(insertIntoCandidate("AB", fromStart, fromEnd) == "AE)B(S", "from the start at the very end, from the end before the last");
        check(insertIntoCandidate("A", fromStart, fromEnd) == "E)A", "too short for from the start");
        check(insertIntoCandidate("AB", Insertion{"X", 1}, Insertion{"Y", 1}) == "AXYB", "where they meet, from the start first");
        bool roundTrips = true;
        for (const std::string candidate : {"", "A", "AB", "ABC", "ABCD", "ABCDEFG"}) {
            std::string back;
            roundTrips &= removeInsertions(insertIntoCandidate(candidate, fromStart, fromEnd), fromStart, fromEnd, back) && back == candidate;
        }
        check(roundTrips, "removeInsertions undoes insertIntoCandidate, at every length");
        std::string back;
        check(!removeInsertions("ABCD", fromStart, fromEnd, back), "without the text where a candidate of its length has it: false");
    }

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
