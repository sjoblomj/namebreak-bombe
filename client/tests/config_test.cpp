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

static void writeFile(const std::string& path, const std::string& text) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary);
    out << text;
}

static void testMatchesName() {
    printf("--- matches_name in [search] ---\n");
    ConfigFile config = makeConfig("", "");
    SearchRequest req;
    std::string error;
    check(buildSearchRequest(config, true, req, error) && req.outputFilePath == kDir + "/matches.txt", "not given: matches.txt");
    config.search["matches_name"] = "credits";
    check(buildSearchRequest(config, true, req, error) && req.outputFilePath == kDir + "/matches-credits.txt", "given: matches-<name>.txt");
    config.search["matches_name"] = "../x\\y";
    check(buildSearchRequest(config, true, req, error) && req.outputFilePath == kDir + "/matches-.._x_y.txt",
          "a name with path separators: made safe, so it stays in matches_dir");
}

static void testConfigList() {
    printf("--- parseConfigList ---\n");
    auto list = [](const std::string& value) {
        std::vector<std::string> items;
        std::string error;
        if (!parseConfigList(value, items, error))
            return "<error: " + error + ">";
        std::string out;
        for (const std::string& item : items)
            out += "[" + item + "]";
        return out;
    };
    check(list("") == "" && list("   ") == "", "empty: no items");
    check(list("a, b ,c") == "[a][b][c]", "unquoted items, trimmed");
    check(list("\"\", \"_\", \"-\", \" \"") == "[][_][-][ ]", "quoted items kept as they are - empty and a space included");
    check(list("\"a, b\", c") == "[a, b][c]", "a quoted item may have a comma");
    check(list("  \"x\"  ,y") == "[x][y]", "spaces around a quoted item dropped");
    check(list("lists/sc words.txt, other.txt") == "[lists/sc words.txt][other.txt]", "a space inside an unquoted item kept");
    check(list("\"_\"") == "[_]", "one quoted item");
    check(list("a,,b").rfind("<error:", 0) == 0, "an empty unquoted item: an error");
    check(list("a,").rfind("<error:", 0) == 0, "a trailing comma: an error");
    check(list("\"a").rfind("<error:", 0) == 0, "an unterminated quote: an error");
    check(list("\"a\"b, c").rfind("<error:", 0) == 0, "text after a quoted item: an error");
}

static void testLoadDictionarySection() {
    printf("--- loadConfigFile with [dictionary] ---\n");
    const std::string path = kDir + "/dictionary.conf";
    writeFile(path, "mode = dictionary\n[search]\nprefix = \" A \"\n[dictionary]\nseparators = \"\", \"_\"\nprefix = \" B \"\n");
    ConfigFile config;
    std::string error;
    check(loadConfigFile(path, config, error), "a [dictionary] section is read");
    check(config.mode == "dictionary", "mode = dictionary");
    check(config.dictionary["separators"] == "\"\", \"_\"", "its values are kept as written, quotes and all");
    check(config.dictionary["prefix"] == "\" B \"" && config.search["prefix"] == " A ", "... while [search]'s are unquoted, as before");
    writeFile(path, "[dictonary]\n");
    check(!loadConfigFile(path, config, error) && error.find("[dictionary]") != std::string::npos, "a misspelt section: an error naming the real ones");
}

static const std::string kWords = kDir + "/words.txt";

// A [dictionary] config with its own word list (no english-1).
static ConfigFile makeDictionaryConfig() {
    writeFile(kWords, "zerg\ncrdt\nlst\nZerg\n");
    ConfigFile config;
    config.matchesDir = kDir;
    config.dictionary = {
        {"builtin_dictionary", "none"}, {"dictionaries", kWords}, {"max_words", "2"}, {"prefix", "rez/"},
        {"suffix", ".txt"},             {"hash_a", "0x339EFE07"}, {"hash_b", "0xE6B2B01C"},
    };
    return config;
}

// buildDictionaryRequest on `config`: the request, or "<error: ...>" in `error`.
static bool buildDict(const ConfigFile& config, DictionaryRequest& req, std::string& error, std::vector<std::string>* warnings = nullptr) {
    std::vector<std::string> ignored;
    return buildDictionaryRequest(config, req, warnings ? *warnings : ignored, error);
}

static std::string errorOf(const ConfigFile& config) {
    DictionaryRequest req;
    std::string error;
    if (buildDict(config, req, error))
        return "<no error>";
    return error;
}

static void testDictionaryRequest() {
    printf("--- buildDictionaryRequest ---\n");
    DictionaryRequest req;
    std::string error;
    {
        ConfigFile config = makeDictionaryConfig();
        check(buildDict(config, req, error), "a minimal config builds (" + error + ")");
        check(req.pattern.words == std::vector<std::string>{"CRDT", "LST", "ZERG"}, "words normalized, sorted and without duplicates");
        check(req.wordSource == kWords, "where they came from: the list");
        check(req.pattern.separators == std::vector<std::string>{""} && req.pattern.minWords == 1 && req.pattern.maxWords == 2,
              "defaults: separator \"\", min_words 1");
        check(req.prefix == "REZ\\" && req.suffix == ".TXT", "prefix and suffix normalized");
        check(!req.bounds.hasLower && !req.bounds.hasUpper, "no bounds");
        check(req.targetHashA == 0x339EFE07 && req.targetHashB == 0xE6B2B01C, "the hashes");
        check(!req.checkBasename, "no encryption_key: no basenames");
        check(req.outputFilePath == kDir + "/matches.txt" && req.basenamesFilePath == kDir + "/basenames.txt" &&
                  req.progressFilePath == kDir + "/wordnumber.txt",
              "files: matches.txt, basenames.txt, wordnumber.txt in matches_dir");
        check(req.startNumber == 0, "from the beginning");
    }
    {
        ConfigFile config = makeDictionaryConfig();
        config.dictionary.erase("builtin_dictionary");
        check(buildDict(config, req, error) && req.pattern.words.size() == 63875 + 3 && req.wordSource == "english-1 + " + kWords,
              "builtin_dictionary not given: english-1, plus the list's three words (in neither twice)");
        config.dictionary["builtin_dictionary"] = "english-1";
        check(buildDict(config, req, error) && req.pattern.words.size() == 63878, "english-1 by name: the same");
        config.dictionary.erase("dictionaries");
        check(buildDict(config, req, error) && req.pattern.words.size() == 63875 && req.wordSource == "english-1", "english-1 alone");
        config.dictionary["builtin_dictionary"] = "english-2";
        check(errorOf(config).find("unknown builtin_dictionary") != std::string::npos, "an unknown builtin_dictionary: an error");
    }
    {
        ConfigFile config = makeDictionaryConfig();
        writeFile(kDir + "/more.txt", "# protoss\nprotoss\nZERG\n");
        config.dictionary["dictionaries"] = kWords + ", \"" + kDir + "/more.txt\"";
        check(buildDict(config, req, error) && req.pattern.words == std::vector<std::string>{"CRDT", "LST", "PROTOSS", "ZERG"} &&
                  req.wordSource == kWords + " + " + kDir + "/more.txt",
              "several lists: combined");
        config.dictionary["dictionaries"] = kWords + ", " + kDir + "/missing.txt";
        check(errorOf(config).find("missing.txt") != std::string::npos, "a list that can't be read: an error naming it");
        config.dictionary["dictionaries"] = kWords + ",";
        check(errorOf(config).find("invalid dictionaries") != std::string::npos, "a malformed list of lists: an error");
        writeFile(kDir + "/empty.txt", "# nothing\n");
        config.dictionary["dictionaries"] = kDir + "/empty.txt";
        check(errorOf(config).find("no words") != std::string::npos, "no words at all: an error");
        writeFile(kDir + "/bad.txt", "ok\ncaf\xC3\xA9\n");
        config.dictionary["dictionaries"] = kDir + "/bad.txt";
        std::vector<std::string> warnings;
        check(buildDict(config, req, error, &warnings) && req.pattern.words == std::vector<std::string>{"OK"} && warnings.size() == 1 &&
                  warnings[0].find("bad.txt:2") != std::string::npos,
              "a line a list skips: a warning naming it");
    }
    {
        ConfigFile config = makeDictionaryConfig();
        config.dictionary["separators"] = "\"\", \"_\", \"/\", \" \"";
        config.dictionary["min_words"] = "2";
        config.dictionary["max_words"] = "3";
        check(buildDict(config, req, error) && req.pattern.separators == std::vector<std::string>{"", "_", "\\", " "} &&
                  req.pattern.minWords == 2 && req.pattern.maxWords == 3,
              "separators (normalized) and word counts");
        config.dictionary["separators"] = "a, A";
        check(errorOf(config).find("more than once") != std::string::npos, "separators the same once normalized: an error");
        config.dictionary["separators"] = "\"_";
        check(errorOf(config).find("invalid separators") != std::string::npos, "malformed separators: an error");
        config.dictionary["separators"] = "_";
        config.dictionary["min_words"] = "3";
        config.dictionary["max_words"] = "2";
        check(errorOf(config).find("min_words") != std::string::npos, "min_words > max_words: an error");
        config.dictionary["min_words"] = "one";
        check(errorOf(config).find("invalid min_words") != std::string::npos, "min_words not a number: an error");
        config.dictionary["min_words"] = "1";
        config.dictionary["max_words"] = "-2";
        check(errorOf(config).find("invalid max_words") != std::string::npos, "max_words not a number: an error");
        config.dictionary["max_words"] = "9";
        check(errorOf(config).find("max_words") != std::string::npos, "max_words over the limit: an error");
    }
    {
        for (const char* key : {"max_words", "prefix", "suffix", "hash_a", "hash_b"}) {
            ConfigFile config = makeDictionaryConfig();
            config.dictionary.erase(key);
            check(errorOf(config).find(std::string("missing required key '") + key + "'") != std::string::npos,
                  std::string("no ") + key + ": an error");
        }
        ConfigFile config = makeDictionaryConfig();
        config.dictionary["alphabet"] = "ABC";
        check(errorOf(config).find("unknown key 'alphabet'") != std::string::npos, "an unknown key: an error");
        config = makeDictionaryConfig();
        config.dictionary["hash_a"] = "xyz";
        check(errorOf(config).find("invalid hash_a") != std::string::npos, "a hash that isn't hex: an error");
        config = makeDictionaryConfig();
        config.dictionary["hash_b"] = "0xG";
        check(errorOf(config).find("invalid hash_b") != std::string::npos, "hash_b that isn't hex: an error");
        config = makeDictionaryConfig();
        config.dictionary["hash_b"] = "\"0x12\"";
        check(buildDict(config, req, error) && req.targetHashB == 0x12, "a quoted value: unquoted");
    }
    {
        ConfigFile config = makeDictionaryConfig();
        config.dictionary["prefix"] = "\" rez\\\\ \"";
        config.dictionary["suffix"] = "\"\"";
        check(buildDict(config, req, error) && req.prefix == " REZ\\\\ " && req.suffix == "", "a quoted prefix keeps its spaces; \"\" is empty");
        config = makeDictionaryConfig();
        config.dictionary["lower_bound"] = "rez/crdt_lst.txt";
        config.dictionary["upper_bound"] = "REZ\\glucmpgn.bin";
        check(buildDict(config, req, error) && req.bounds.hasLower && req.bounds.lower == "REZ\\CRDT_LST.TXT" && req.bounds.hasUpper &&
                  req.bounds.upper == "REZ\\GLUCMPGN.BIN",
              "bounds normalized");
        config.dictionary["lower_bound"] = "REZ\\Z";
        check(errorOf(config).find("sorts after") != std::string::npos, "lower_bound after upper_bound: an error");
        config.dictionary.erase("upper_bound");
        config.dictionary["lower_bound"] = "\"\"";
        check(buildDict(config, req, error) && req.bounds.hasLower && req.bounds.lower.empty() && !req.bounds.hasUpper,
              "a bound of \"\" is a bound (an empty one)");
    }
    {
        ConfigFile config = makeDictionaryConfig();
        config.dictionary["encryption_key"] = "0x4565C467";
        check(buildDict(config, req, error) && req.checkBasename && req.basenameKey == 0x4565C467 && req.recordBasenames &&
                  !req.recordHashAMatches,
              "an encryption_key: used, basenames recorded, not every hashA hit");
        config.dictionary["record_basenames"] = "false";
        check(buildDict(config, req, error) && req.checkBasename && !req.recordBasenames,
              "... the basenames not recorded with record_basenames = false - the key used all the same");
        config.dictionary["record_basenames"] = "true";
        check(buildDict(config, req, error) && req.checkBasename && req.recordBasenames, "record_basenames = true: recorded");
        config.dictionary["record_hasha_matches"] = "true";
        check(buildDict(config, req, error) && req.recordHashAMatches, "record_hasha_matches = true: every hashA hit");
        config.dictionary["record_hasha_matches"] = "sometimes";
        check(errorOf(config).find("invalid record_hasha_matches") != std::string::npos, "record_hasha_matches not a boolean: an error");
        config.dictionary.erase("record_hasha_matches");
        config.dictionary["record_basenames"] = "maybe";
        check(errorOf(config).find("invalid record_basenames") != std::string::npos, "record_basenames not a boolean: an error");
        config.dictionary["record_basenames"] = "true";
        config.dictionary["encryption_key"] = "0xNOPE";
        check(errorOf(config).find("invalid encryption_key") != std::string::npos, "an encryption_key that isn't hex: an error");
        config.dictionary.erase("encryption_key");
        check(errorOf(config).find("needs an encryption_key") != std::string::npos, "record_basenames = true without a key: an error");
        config.dictionary["record_basenames"] = "false";
        check(buildDict(config, req, error) && !req.checkBasename, "record_basenames = false without a key: fine");
        config.dictionary["record_hasha_matches"] = "true";
        check(buildDict(config, req, error) && !req.checkBasename, "record_hasha_matches = true without a key: fine (every hashA hit anyway)");
    }
    {
        ConfigFile config = makeDictionaryConfig();
        config.dictionary["matches_name"] = "credits";
        check(buildDict(config, req, error) && req.outputFilePath == kDir + "/matches-credits.txt" &&
                  req.basenamesFilePath == kDir + "/basenames-credits.txt" && req.progressFilePath == kDir + "/wordnumber-credits.txt",
              "matches_name: matches-, basenames- and wordnumber-<name>.txt");
        config.dictionary["resume_from_last_candidate"] = "perhaps";
        check(errorOf(config).find("invalid resume_from_last_candidate") != std::string::npos, "resume not a boolean: an error");
    }
}

static void testDictionaryResume() {
    printf("--- resume_from_last_candidate in [dictionary] ---\n");
    ConfigFile config = makeDictionaryConfig();
    config.dictionary["matches_name"] = "resume";
    DictionaryRequest req;
    std::string error;
    buildDict(config, req, error);
    const std::string progressPath = req.progressFilePath;
    std::filesystem::remove(progressPath);
    const std::string fingerprint = dictionaryFingerprint(req);

    config.dictionary["resume_from_last_candidate"] = "true";
    check(buildDict(config, req, error) && req.startNumber == 0, "no progress file: from the beginning");
    writeDictionaryProgress(progressPath, {fingerprint, 7}, 12, "LST_ZERG", error);
    check(buildDict(config, req, error) && req.startNumber == 7, "a progress file from this search: resumes from its number");
    config.dictionary["resume_from_last_candidate"] = "false";
    check(buildDict(config, req, error) && req.startNumber == 0, "resume_from_last_candidate = false: from the beginning anyway");
    config.dictionary["resume_from_last_candidate"] = "true";
    config.dictionary["separators"] = "\"\", _";
    check(errorOf(config).find("different settings") != std::string::npos, "the settings changed since: refuses to resume, saying why");
    config.dictionary["separators"] = "\"\"";
    config.dictionary["lower_bound"] = "REZ\\A";
    check(errorOf(config).find("different settings") != std::string::npos, "... a bound changed, too");
    config.dictionary.erase("lower_bound");
    writeFile(progressPath, "fingerprint = 0123456789abcdef\n");
    check(errorOf(config).find("isn't a dictionary search's progress file") != std::string::npos, "a broken progress file: an error");
    std::filesystem::remove(progressPath);
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

    testMatchesName();
    testConfigList();
    testLoadDictionarySection();
    testDictionaryRequest();
    testDictionaryResume();

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
