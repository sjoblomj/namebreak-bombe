// The coordinator protocol's JSON (net/protocol.h): the reports this client
// sends, exactly, and the claims it reads - an alphabet target's, as clients
// before protocol 1.5 read them too, a dictionary target's, with its arrays
// of strings, and a meet-in-the-middle target's. The claims are written as the
// server's serde writes them (coordinator/protocol/src/lib.rs). No network
// access.

#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "net/protocol.h"

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

static void testReports() {
    printf("--- reports ---\n");
    HeartbeatRequest hb;
    check(toJson(hb) == R"({"last_hash_a_match_filename":null})", "an empty heartbeat: as before protocol 1.5");
    hb.lastHashAMatchFilename = "REZ\\A\"B.WAV";
    check(toJson(hb) == R"({"last_hash_a_match_filename":"REZ\\A\"B.WAV"})", "an alphabet range's heartbeat: as before protocol 1.5");
    hb = HeartbeatRequest();
    hb.nextCandidateNumber = 123456789012345;
    hb.basenames = {"A.WAV", "B\"C.WAV"};
    check(toJson(hb) == R"({"last_hash_a_match_filename":null,"next_candidate_number":123456789012345,"basenames":["A.WAV","B\"C.WAV"]})",
          "a dictionary range's heartbeat: its number and basenames");
    hb.basenames.clear();
    hb.nextCandidateNumber = 0;
    check(toJson(hb) == R"({"last_hash_a_match_filename":null,"next_candidate_number":0})", "no basenames: none sent, not an empty array");

    QuitRequest quit;
    check(toJson(quit) == R"({"last_hash_a_match_filename":null})", "an empty quit");
    quit.nextCandidateNumber = 7;
    quit.basenames = {"X.WAV"};
    check(toJson(quit) == R"({"last_hash_a_match_filename":null,"next_candidate_number":7,"basenames":["X.WAV"]})", "a dictionary range's quit");
    quit = QuitRequest();
    quit.nextTailIndex = 9;
    check(toJson(quit) == R"({"last_hash_a_match_filename":null,"next_tail_index":9})", "a meet-in-the-middle range's quit");

    RegisterRequest reg;
    reg.username = "ojan";
    reg.hostname = "laptop";
    reg.backend = "cuda";
    reg.clientRelease = "dev";
    check(toJson(reg) == R"({"username":"ojan","hostname":"laptop","backend":"cuda","protocol_version":"1.7.0",)"
                         R"("client_release":"dev","meet_in_the_middle_table_bytes":null})",
          "a registration without meet_in_the_middle_table_memory: no memory for a meet-in-the-middle table");
    reg.meetInTheMiddleTableBytes = 1073741824;
    check(toJson(reg).find(R"("meet_in_the_middle_table_bytes":1073741824})") != std::string::npos, "... and with it, what it will give");

    hb = HeartbeatRequest();
    hb.nextTailIndex = 4242;
    check(toJson(hb) == R"({"last_hash_a_match_filename":null,"next_tail_index":4242})",
          "a meet-in-the-middle range's heartbeat: the tail it has got to");

    CompleteRequest complete;
    complete.found = true;
    complete.filename = "MUSIC\\A.WAV";
    complete.elapsedSeconds = 1.5;
    complete.candidatesProcessed = 42;
    const std::string before = R"({"found":true,"filename":"MUSIC\\A.WAV","elapsed_seconds":1.500000,"candidates_processed":42})";
    check(toJson(complete) == before, "a completion without basenames: as before protocol 1.5");
    complete.basenames = {"Y.WAV", "Z.WAV"};
    check(toJson(complete) == before.substr(0, before.size() - 1) + R"(,"basenames":["Y.WAV","Z.WAV"]})", "... and with them");

    // False matches, from protocol 1.7: with any kind of report, left out
    // when there are none - as every report above has none.
    complete.basenames.clear();
    complete.falseMatches = {"REZ\\SFZSSP050WSRL.WAV"};
    check(toJson(complete) == before.substr(0, before.size() - 1) + R"(,"false_matches":["REZ\\SFZSSP050WSRL.WAV"]})",
          "a completion with a false match");
    hb = HeartbeatRequest();
    hb.lastHashAMatchFilename = "REZ\\B.WAV";
    hb.falseMatches = {"REZ\\A.WAV", "REZ\\B.WAV"};
    check(toJson(hb) == R"({"last_hash_a_match_filename":"REZ\\B.WAV","false_matches":["REZ\\A.WAV","REZ\\B.WAV"]})",
          "an alphabet range's heartbeat with false matches");
    hb = HeartbeatRequest();
    hb.nextCandidateNumber = 3;
    hb.basenames = {"A.WAV"};
    hb.falseMatches = {"MUSIC\\A.WAV"};
    check(toJson(hb) == R"({"last_hash_a_match_filename":null,"next_candidate_number":3,"basenames":["A.WAV"],"false_matches":["MUSIC\\A.WAV"]})",
          "a dictionary range's heartbeat with basenames and a false match");
    quit = QuitRequest();
    quit.nextTailIndex = 2;
    quit.falseMatches = {"X\\Y.PIF"};
    check(toJson(quit) == R"({"last_hash_a_match_filename":null,"next_tail_index":2,"false_matches":["X\\Y.PIF"]})", "a quit with a false match");
}

// An alphabet target's claim, as the server writes it.
static const char* kAlphabetClaim = R"({"range_id":5,"target_id":2,"target_name":"rez","prefix":"REZ\\","suffix":".WAV",
    "hash_a_hex":"0xF60F5D90","hash_b_hex":"0xCE0A9BDB","prune_symbol_runs":true,"prune_unopened_brackets":false,
    "prune_whole_candidate":true,"max_backslash_count":0,"min_backslash_count":0,"prune_adjacent_backslashes":false,
    "insert_from_start_text":null,"insert_from_start_position":0,"insert_from_end_text":null,"insert_from_end_position":0,
    "lower_bound_filename":"REZ\\AAAA.WAV","upper_bound_filename":"REZ\\AZZZ.WAV","alphabet":"ABZ","candidate_count":100,"lease_seconds":21600})";

// A dictionary target's, with `fields` in place of its dictionary fields.
static std::string dictionaryClaim(const std::string& fields) {
    return R"({"range_id":9,"target_id":3,"target_name":"words","prefix":"MUSIC\\","suffix":".WAV","hash_a_hex":"0x11111111",
        "hash_b_hex":"0x22222222","prune_symbol_runs":false,"prune_unopened_brackets":false,"prune_whole_candidate":false,
        "max_backslash_count":0,"min_backslash_count":0,"prune_adjacent_backslashes":false,"insert_from_start_text":null,
        "insert_from_start_position":0,"insert_from_end_text":null,"insert_from_end_position":0,
        "lower_bound_filename":"MUSIC\\BRAVO.WAV","upper_bound_filename":"MUSIC\\DELTA.WAV","alphabet":"","candidate_count":3,
        "lease_seconds":60)" + fields + "}";
}

// A dictionary claim's own fields, as (key, JSON value) - with `skip` left
// out and `changed`'s values put in.
using Fields = std::vector<std::pair<std::string, std::string>>;
static const Fields kDictionaryFields = {
    {"dictionary", "true"},
    {"word_lists", R"(["english-1","sc-words"])"},
    {"word_list_checksums", R"(["63b352823c6059b0","0123456789abcdef"])"},
    {"words_checksum", R"("fedcba9876543210")"},
    {"separators", R"([ "", "_" , "\\", "\"" ])"},
    {"min_words", "1"},
    {"max_words", "2"},
    {"tails", R"(["digits:1-2", "_|"])"},
    {"tails_checksum", R"("0011223344556677")"},
    {"first_candidate_number", "1"},
    {"end_candidate_number", "4"},
    {"filename_lower_bound", R"("MUSIC\\B")"},
    {"send_basenames", "true"},
    {"encryption_key_hex", R"("0x0000ABCD")"},
};

static std::string fieldsJson(const std::string& skip = "", const Fields& changed = {}) {
    std::string out;
    for (const auto& [key, value] : kDictionaryFields) {
        if (key == skip)
            continue;
        std::string v = value;
        for (const auto& [changedKey, changedValue] : changed) {
            if (changedKey == key)
                v = changedValue;
        }
        out += ",\"" + key + "\":" + v;
    }
    return out;
}

// A meet-in-the-middle target's claim, with `fields` in place of its meet-in-the-middle fields.
static std::string meetInTheMiddleClaim(const std::string& fields) {
    return R"({"range_id":11,"target_id":4,"target_name":"setupdat","prefix":"","suffix":"\\BWUNIN.PIF","hash_a_hex":"0x5BAB63FE",
        "hash_b_hex":"0x8DF2CB08","prune_symbol_runs":true,"prune_unopened_brackets":false,"prune_whole_candidate":false,
        "max_backslash_count":0,"min_backslash_count":0,"prune_adjacent_backslashes":true,"insert_from_start_text":null,
        "insert_from_start_position":0,"insert_from_end_text":null,"insert_from_end_position":0,
        "lower_bound_filename":"????  \\BWUNIN.PIF","upper_bound_filename":"????__\\BWUNIN.PIF",
        "alphabet":" ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_","candidate_count":1849,"lease_seconds":21600)" +
           fields + "}";
}

static const Fields kMeetInTheMiddleFields = {
    {"meet_in_the_middle", "true"},
    {"meeting_point_a_seed1_hex", R"("0x92F62345")"},
    {"meeting_point_a_seed2_hex", R"("0xB9F61953")"},
    {"meeting_point_b_seed1_hex", R"("0xB95115E6")"},
    {"meeting_point_b_seed2_hex", R"("0x7609B4F4")"},
    {"directory_length", "6"},
    {"meet_in_the_middle_table_length", "4"},
    {"first_tail_index", "0"},
    {"end_tail_index", "1849"},
};

static std::string meetInTheMiddleFieldsJson(const std::string& skip = "", const Fields& changed = {}) {
    std::string out;
    for (const auto& [key, value] : kMeetInTheMiddleFields) {
        if (key == skip)
            continue;
        std::string v = value;
        for (const auto& [changedKey, changedValue] : changed) {
            if (changedKey == key)
                v = changedValue;
        }
        out += ",\"" + key + "\":" + v;
    }
    return out;
}

static void testMeetInTheMiddleClaims() {
    printf("--- a meet-in-the-middle target's claims ---\n");
    ClaimResponse c;
    check(parseClaimResponse(meetInTheMiddleClaim(meetInTheMiddleFieldsJson()), c), "a meet-in-the-middle target's claim parses");
    check(c.meetInTheMiddle && !c.dictionary && c.rangeId == 11 && c.targetName == "setupdat", "... as a meet-in-the-middle target's");
    check(c.meetingPointASeed1Hex == "0x92F62345" && c.meetingPointASeed2Hex == "0xB9F61953" && c.meetingPointBSeed1Hex == "0xB95115E6" &&
              c.meetingPointBSeed2Hex == "0x7609B4F4",
          "... its meeting point, both hashes' seeds");
    check(c.directoryLength == 6 && c.meetInTheMiddleTableLength == 4 && c.firstTailIndex == 0 && c.endTailIndex == 1849 && c.candidateCount == 1849,
          "... the directory's length, the table's, and the tails to rewind");
    check(c.alphabet.size() == 43 && c.pruneSymbolRuns && c.pruneAdjacentBackslashes && !c.pruneWholeCandidate,
          "... its alphabet and pruning rules");
    check(c.wordLists.empty() && !c.sendBasenames, "... and none of a dictionary target's fields");

    // What a meet-in-the-middle claim has to have.
    for (const char* key : {"meeting_point_a_seed1_hex", "meeting_point_a_seed2_hex", "meeting_point_b_seed1_hex",
                            "meeting_point_b_seed2_hex", "directory_length", "meet_in_the_middle_table_length", "first_tail_index",
                            "end_tail_index"}) {
        ClaimResponse missing;
        check(!parseClaimResponse(meetInTheMiddleClaim(meetInTheMiddleFieldsJson(key)), missing),
              std::string("a meet-in-the-middle claim without ") + key + " is refused");
    }
    ClaimResponse plain;
    check(parseClaimResponse(meetInTheMiddleClaim(meetInTheMiddleFieldsJson("meet_in_the_middle")), plain) && !plain.meetInTheMiddle,
          "... but without meet_in_the_middle: true, it's an alphabet claim, whatever else it has");
    auto refused = [](const Fields& changed, const std::string& what) {
        ClaimResponse bad;
        check(!parseClaimResponse(meetInTheMiddleClaim(meetInTheMiddleFieldsJson("", changed)), bad), what);
    };
    refused({{"meet_in_the_middle_table_length", "7"}}, "a table longer than the directory: refused");
    refused({{"meet_in_the_middle_table_length", "-1"}}, "a table of fewer than no characters: refused");
    refused({{"first_tail_index", "-1"}}, "a negative tail index: refused");
    refused({{"end_tail_index", "0"}, {"first_tail_index", "1"}}, "a range of tails ending before it starts: refused");
    refused({{"meeting_point_a_seed1_hex", "null"}}, "a meeting point seed left null: refused");
    {
        // A table of none is a search that rewinds the whole directory.
        ClaimResponse noTable;
        check(parseClaimResponse(meetInTheMiddleClaim(meetInTheMiddleFieldsJson("", {{"meet_in_the_middle_table_length", "0"}})), noTable) &&
                  noTable.meetInTheMiddleTableLength == 0,
              "no table at all: parses");
    }
}

// What's known about a target's file beyond its hashes, from protocol 1.7 -
// in any kind of target's claim.
static void testMatchChecks() {
    printf("--- the key and the hash table position ---\n");
    auto withFields = [](const std::string& fields) {
        std::string claim = kAlphabetClaim;
        claim.insert(claim.size() - 1, fields);
        return claim;
    };
    ClaimResponse c;
    check(parseClaimResponse(kAlphabetClaim, c) && c.encryptionKeyHex.empty() && !c.hashTableSize && !c.hashTableFirstStartSlot &&
              !c.hashTableLastStartSlot,
          "an alphabet claim without them: nothing known");
    c = ClaimResponse();
    check(parseClaimResponse(withFields(R"(,"encryption_key_hex":"0x4A8FA566","hash_table_size":2048,"hash_table_first_start_slot":849,)"
                                        R"("hash_table_last_start_slot":851)"),
                             c) &&
              c.encryptionKeyHex == "0x4A8FA566" && c.hashTableSize == std::optional<int64_t>(2048) &&
              c.hashTableFirstStartSlot == std::optional<int64_t>(849) && c.hashTableLastStartSlot == std::optional<int64_t>(851),
          "an alphabet claim with the key and the hash table position");
    c = ClaimResponse();
    check(parseClaimResponse(withFields(R"(,"hash_table_size":16,"hash_table_first_start_slot":14,"hash_table_last_start_slot":1)"), c) &&
              c.hashTableFirstStartSlot == std::optional<int64_t>(14) && c.hashTableLastStartSlot == std::optional<int64_t>(1),
          "slots wrapping past the table's end");
    c = ClaimResponse();
    check(parseClaimResponse(withFields(R"(,"hash_table_size":null,"hash_table_first_start_slot":null,"hash_table_last_start_slot":null)"), c) &&
              !c.hashTableSize,
          "all three null: none");
    c = ClaimResponse();
    check(parseClaimResponse(meetInTheMiddleClaim(meetInTheMiddleFieldsJson() + R"(,"hash_table_size":1,"hash_table_first_start_slot":0,)"
                                                                                R"("hash_table_last_start_slot":0)"),
                             c) &&
              c.meetInTheMiddle && c.hashTableSize == std::optional<int64_t>(1),
          "in a meet-in-the-middle claim too");
    for (const char* bad : {R"(,"hash_table_size":2048,"hash_table_first_start_slot":849)",
                            R"(,"hash_table_size":2048,"hash_table_last_start_slot":851)",
                            R"(,"hash_table_first_start_slot":849,"hash_table_last_start_slot":851)",
                            R"(,"hash_table_size":2000,"hash_table_first_start_slot":1,"hash_table_last_start_slot":2)",
                            R"(,"hash_table_size":0,"hash_table_first_start_slot":0,"hash_table_last_start_slot":0)",
                            R"(,"hash_table_size":2048,"hash_table_first_start_slot":2048,"hash_table_last_start_slot":2)",
                            R"(,"hash_table_size":2048,"hash_table_first_start_slot":1,"hash_table_last_start_slot":-1)",
                            R"(,"hash_table_size":4294967296,"hash_table_first_start_slot":1,"hash_table_last_start_slot":2)",
                            R"(,"hash_table_size":"2048","hash_table_first_start_slot":1,"hash_table_last_start_slot":2)",
                            R"(,"encryption_key_hex":12345)"}) {
        ClaimResponse refused;
        check(!parseClaimResponse(withFields(bad), refused), std::string("refused: ") + bad);
    }
}

static void testClaims() {
    printf("--- claims ---\n");
    ClaimResponse c;
    check(parseClaimResponse(kAlphabetClaim, c), "an alphabet target's claim parses");
    check(!c.dictionary && c.rangeId == 5 && c.alphabet == "ABZ" && c.pruneSymbolRuns && c.pruneWholeCandidate && c.wordLists.empty(),
          "... as an alphabet target's");

    c = ClaimResponse();
    check(parseClaimResponse(dictionaryClaim(fieldsJson()), c), "a dictionary target's claim parses");
    check(c.dictionary && c.wordLists == std::vector<std::string>{"english-1", "sc-words"} &&
              c.wordListChecksums == std::vector<std::string>{"63b352823c6059b0", "0123456789abcdef"} && c.wordsChecksum == "fedcba9876543210",
          "... its word lists and checksums");
    check(c.separators == std::vector<std::string>{"", "_", "\\", "\""}, "... its separators, escapes and all");
    check(c.minWords == 1 && c.maxWords == 2 && c.firstCandidateNumber == 1 && c.endCandidateNumber == 4 && c.candidateCount == 3,
          "... its word counts and numbers");
    check(c.filenameLowerBound == std::optional<std::string>("MUSIC\\B") && !c.filenameUpperBound, "... its bounds, one of them left out");
    check(c.sendBasenames && c.encryptionKeyHex == "0x0000ABCD", "... and the basenames to send");
    check(c.tails == std::vector<std::string>{"digits:1-2", "_|"} && c.tailsChecksum == "0011223344556677", "... and its tails and their checksum");

    c = ClaimResponse();
    const std::string emptyAndNull = fieldsJson("send_basenames", {{"separators", "[]"}, {"filename_lower_bound", "null"}, {"encryption_key_hex", "null"}}) +
                                     R"(,"filename_upper_bound":"Z")";
    check(parseClaimResponse(dictionaryClaim(emptyAndNull), c) && c.separators.empty() && !c.filenameLowerBound &&
              c.filenameUpperBound == std::optional<std::string>("Z") && !c.sendBasenames && c.encryptionKeyHex.empty(),
          "an empty array, a null bound, and send_basenames left out");
    c = ClaimResponse();
    check(parseClaimResponse(dictionaryClaim(fieldsJson("send_basenames")), c) && !c.sendBasenames && c.encryptionKeyHex == "0x0000ABCD",
          "a key without send_basenames: kept, for hashing the basenames, with none to send");

    // What a dictionary claim has to have.
    for (const char* key : {"word_lists", "word_list_checksums", "words_checksum", "separators", "min_words", "max_words",
                            "first_candidate_number", "end_candidate_number"}) {
        ClaimResponse missing;
        check(!parseClaimResponse(dictionaryClaim(fieldsJson(key)), missing), std::string("a dictionary claim without ") + key + " is refused");
    }
    ClaimResponse plain;
    check(parseClaimResponse(dictionaryClaim(fieldsJson("dictionary")), plain) && !plain.dictionary,
          "... but without dictionary: true, it's an alphabet claim, whatever else it has");
    auto refused = [](const Fields& changed, const std::string& what) {
        ClaimResponse bad;
        check(!parseClaimResponse(dictionaryClaim(fieldsJson("", changed)), bad), what);
    };
    refused({{"word_list_checksums", R"(["63b352823c6059b0"])"}}, "a checksum short: refused");
    refused({{"encryption_key_hex", "null"}}, "basenames to send, but no key: refused");
    refused({{"tails_checksum", "null"}}, "tails without their checksum: refused");
    refused({{"tails", "null"}}, "a tails checksum without tails: refused");
    {
        ClaimResponse noTails;
        check(parseClaimResponse(dictionaryClaim(fieldsJson("", {{"tails", "null"}, {"tails_checksum", "null"}})), noTails) && noTails.tails.empty() &&
                  noTails.tailsChecksum.empty(),
              "no tails (null, as left out): none");
    }
    refused({{"end_candidate_number", "0"}}, "a range ending before it starts: refused");
    refused({{"first_candidate_number", "-1"}}, "a negative number: refused");
    refused({{"word_lists", R"(["english-1",2])"}}, "an array of anything but strings: refused");
    refused({{"word_lists", R"(["english-1","sc-words")"}}, "an array never closed: refused");
    refused({{"word_lists", R"(["english-1" "sc-words"])"}}, "an array without commas: refused");
    refused({{"word_lists", R"({"a":"b"})"}}, "an object: refused");
    refused({{"words_checksum", "[]"}}, "an array where a string goes: refused");

    // Arrays are fine anywhere, read or not - a newer server's alphabet claim
    // may have some.
    std::string alphabetWithArray = kAlphabetClaim;
    alphabetWithArray.insert(alphabetWithArray.size() - 1, R"(,"something_new":["x","y"])");
    c = ClaimResponse();
    check(parseClaimResponse(alphabetWithArray, c) && !c.dictionary, "an alphabet claim with an array it doesn't know of: parses");
}

int main() {
    testReports();
    testClaims();
    testMeetInTheMiddleClaims();
    testMatchChecks();
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
