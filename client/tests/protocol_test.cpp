// The coordinator protocol's JSON (net/protocol.h): the reports this client
// sends, exactly, and the claims it reads - an alphabet target's, as clients
// before protocol 1.5 read them too, and a dictionary target's, with its
// arrays of strings. The claims are written as the server's serde writes
// them (coordinator/protocol/src/lib.rs). No network access.

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

    CompleteRequest complete;
    complete.found = true;
    complete.filename = "MUSIC\\A.WAV";
    complete.elapsedSeconds = 1.5;
    complete.candidatesProcessed = 42;
    const std::string before = R"({"found":true,"filename":"MUSIC\\A.WAV","elapsed_seconds":1.500000,"candidates_processed":42})";
    check(toJson(complete) == before, "a completion without basenames: as before protocol 1.5");
    complete.basenames = {"Y.WAV", "Z.WAV"};
    check(toJson(complete) == before.substr(0, before.size() - 1) + R"(,"basenames":["Y.WAV","Z.WAV"]})", "... and with them");
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
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
