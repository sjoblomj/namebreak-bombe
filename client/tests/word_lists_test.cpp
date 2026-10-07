// resolveClaimWords (net/word_lists.h): the words of a coordinator's
// dictionary claim - english-1 from the client itself, other lists from its
// cache or downloaded into it, every one checked against the checksum the
// server gives it, and the merged words against theirs. With a stand-in for
// the server's download. No network access.
//
// Writes files under ./word_lists_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "engine/wordlist.h"
#include "net/word_lists.h"

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

static const std::string kDir = "word_lists_test";

// The checksum the server gives a list with `text`.
static std::string checksumOf(const std::string& text) {
    std::vector<std::string> words, warnings;
    parseWordList(text, "test", words, warnings);
    return hex64(wordListChecksum(sortedUniqueWords(words)));
}

// The stand-in server: its lists, and how often each was downloaded.
struct FakeServer {
    std::map<std::string, std::string> lists;
    std::map<std::string, int> downloads;
    bool reachable = true;

    WordListDownloader downloader() {
        return [this](const std::string& name, std::string& text, std::string& error) {
            if (!reachable) {
                error = "request failed: no route to host";
                return false;
            }
            ++downloads[name];
            auto it = lists.find(name);
            if (it == lists.end()) {
                error = "HTTP 404";
                return false;
            }
            text = it->second;
            return true;
        };
    }
};

static ClaimResponse claimOf(const std::vector<std::string>& names, const std::vector<std::string>& checksums, const std::string& wordsChecksum) {
    ClaimResponse claim;
    claim.dictionary = true;
    claim.wordLists = names;
    claim.wordListChecksums = checksums;
    claim.wordsChecksum = wordsChecksum;
    return claim;
}

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

int main() {
    std::filesystem::remove_all(kDir);
    const std::string cache = kDir + "/word-lists";
    const std::string english1 = hex64(wordListChecksum(sortedUniqueWords(builtinWordList("english-1"))));
    FakeServer server;
    server.lists["sc"] = "zergling\r\nhydralisk\n# units\nZERGLING\naardvark\n";
    const std::string sc = checksumOf(server.lists["sc"]);

    std::vector<std::string> merged = sortedUniqueWords(builtinWordList("english-1"));
    merged.push_back("HYDRALISK");
    merged.push_back("ZERGLING");
    merged = sortedUniqueWords(merged);
    const std::string mergedChecksum = hex64(wordListChecksum(merged));
    const ClaimResponse claim = claimOf({"english-1", "sc"}, {english1, sc}, mergedChecksum);

    printf("--- downloading and keeping ---\n");
    std::vector<std::string> words, downloaded;
    std::string error;
    // Each call first, then its check - so that its description shows the
    // error the call left.
    bool ok = false;
    ok = resolveClaimWords(claim, cache, server.downloader(), words, downloaded, error);
    check(ok, "english-1 and a downloaded list: " + error);
    check(words == merged, "... merged: " + std::to_string(words.size()) + " words, sorted, AARDVARK once");
    check(server.downloads["sc"] == 1 && server.downloads.count("english-1") == 0 && downloaded == std::vector<std::string>{"sc"},
          "only the list not compiled in is downloaded");
    check(readFile(cache + "/sc.txt") == server.lists["sc"], "... and kept as the server sent it");

    ok = resolveClaimWords(claim, cache, server.downloader(), words, downloaded, error) && words == merged;
    check(ok, "again: the same words");
    check(server.downloads["sc"] == 1 && downloaded.empty(), "... from the copy kept, without downloading");

    std::ofstream(cache + "/sc.txt", std::ios::binary) << "zergling\n";
    ok = resolveClaimWords(claim, cache, server.downloader(), words, downloaded, error) && words == merged;
    check(ok, "a kept copy with other words: downloaded again");
    check(server.downloads["sc"] == 2 && readFile(cache + "/sc.txt") == server.lists["sc"], "... and the copy replaced");

    printf("--- what doesn't check out ---\n");
    std::filesystem::remove(cache + "/sc.txt");
    server.reachable = false;
    ok = !resolveClaimWords(claim, cache, server.downloader(), words, downloaded, error) && error.find("no route") != std::string::npos;
    check(ok, "the server can't be reached, nothing kept: " + error);
    server.reachable = true;

    server.lists["sc"] = "zergling\nhydralisk\nultralisk\n";
    ok = !resolveClaimWords(claim, cache, server.downloader(), words, downloaded, error) && error.find("checksum") != std::string::npos;
    check(ok, "the server sends a list without the checksum it gave: " + error);
    check(!std::filesystem::exists(cache + "/sc.txt"), "... which isn't kept");

    ok = !resolveClaimWords(claimOf({"english-1"}, {"0000000000000000"}, english1), cache, server.downloader(), words, downloaded, error) &&
              error.find("compiled into this client") != std::string::npos;
    check(ok, "a server whose english-1 isn't this client's: " + error);
    ok = !resolveClaimWords(claimOf({"english-1"}, {english1}, "0000000000000000"), cache, server.downloader(), words, downloaded, error) &&
              error.find("together") != std::string::npos;
    check(ok, "the words merged without the checksum given: " + error);
    ok = !resolveClaimWords(claimOf({"english-1", "sc"}, {english1}, mergedChecksum), cache, server.downloader(), words, downloaded, error);
    check(ok, "a checksum missing: refused");
    ok = !resolveClaimWords(claimOf({}, {}, mergedChecksum), cache, server.downloader(), words, downloaded, error);
    check(ok, "no lists: refused");
    const int before = server.downloads["../x"];
    ok = !resolveClaimWords(claimOf({"../x"}, {sc}, mergedChecksum), cache, server.downloader(), words, downloaded, error) &&
              server.downloads["../x"] == before;
    check(ok, "a name that could leave the cache: refused, without asking for it");

    printf("--- names ---\n");
    for (const char* good : {"sc", "english-2", "a.b_c-d", "0"})
        check(isValidWordListName(good), std::string("'") + good + "' is a word list's name");
    for (const std::string& bad : {std::string(""), std::string(".x"), std::string("-x"), std::string("a/b"), std::string("a\\b"),
                                   std::string("a b"), std::string(65, 'x')})
        check(!isValidWordListName(bad), "'" + bad + "' isn't");

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
