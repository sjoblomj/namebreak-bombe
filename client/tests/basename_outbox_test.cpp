// BasenameOutbox (net/basename_outbox.h): the basenames a coordinator client
// found for a dictionary target that the server hasn't got yet - kept in a
// file that's cleared as reports get through, and read back by the next
// range of the target. Including while the search adds to it on one thread
// and the heartbeats send from it on another. No network access.
//
// Writes files under ./basename_outbox_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "net/basename_outbox.h"

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

static const std::string kDir = "basename_outbox_test";

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

static void testFile() {
    printf("--- the file ---\n");
    std::filesystem::remove_all(kDir);
    const std::string path = BasenameOutbox::pathFor(kDir, "sc/music bg");
    check(path == kDir + "/unsent-basenames-sc_music_bg.txt", "one file per target, its name made safe: " + path);

    BasenameOutbox outbox;
    std::string error;
    check(outbox.open(path, error) && outbox.size() == 0, "no file yet: nothing waiting");
    check(!std::filesystem::exists(path), "... and none made by opening");

    outbox.add("A.WAV");
    outbox.add("B.WAV");
    outbox.add("A.WAV");
    check(outbox.size() == 2 && readFile(path) == "A.WAV\nB.WAV\n", "added, each once, in the file as they come");
    check(outbox.peek(1) == std::vector<std::string>{"A.WAV"} && outbox.peek(10) == std::vector<std::string>{"A.WAV", "B.WAV"},
          "peek: the oldest, as many as asked and there are");

    // A report carrying the two goes out; the search finds another meanwhile.
    const std::vector<std::string> sent = outbox.peek(10);
    outbox.add("C.WAV");
    outbox.remove(sent.size());
    check(outbox.peek(10) == std::vector<std::string>{"C.WAV"} && readFile(path) == "C.WAV\n", "the sent ones dropped, the one found meanwhile kept");

    BasenameOutbox later;
    check(later.open(path, error) && later.peek(10) == std::vector<std::string>{"C.WAV"}, "a later range of the target finds it waiting");
    later.remove(1);
    check(later.size() == 0 && !std::filesystem::exists(path), "everything sent: the file is gone");
    later.remove(1);
    check(later.size() == 0, "removing from nothing is nothing");

    std::filesystem::create_directories(kDir);
    std::ofstream(path, std::ios::binary) << "X.WAV\r\n\nY.WAV\nX.WAV\nZ.WAV";
    BasenameOutbox crlf;
    check(crlf.open(path, error) && crlf.peek(10) == std::vector<std::string>{"X.WAV", "Y.WAV", "Z.WAV"},
          "a file from elsewhere: CRLF, blank lines, duplicates and no last newline read right");

    // A file that can't be written: still kept in memory, and sent.
    std::ofstream(kDir + "/not-a-directory") << "x";
    BasenameOutbox unwritable;
    check(unwritable.open(kDir + "/not-a-directory/unsent.txt", error), "a path that can't be written: opens, as there's nothing in it");
    unwritable.add("W.WAV");
    check(unwritable.peek(10) == std::vector<std::string>{"W.WAV"}, "... and what's added is still sent");
}

static void testThreads() {
    printf("--- a search adding while heartbeats send ---\n");
    std::filesystem::remove_all(kDir);
    BasenameOutbox outbox;
    std::string error;
    outbox.open(BasenameOutbox::pathFor(kDir, "threads"), error);
    const int count = 2000;
    std::atomic<bool> done{false};
    std::thread search([&]() {
        for (int i = 0; i < count; ++i)
            outbox.add("N" + std::to_string(i) + ".WAV");
        done = true;
    });
    std::vector<std::string> delivered;
    while (!done || outbox.size() > 0) {
        const std::vector<std::string> sent = outbox.peek(37);
        delivered.insert(delivered.end(), sent.begin(), sent.end());
        outbox.remove(sent.size());
    }
    search.join();
    bool inOrder = (int) delivered.size() == count;
    for (int i = 0; inOrder && i < count; ++i)
        inOrder = delivered[i] == "N" + std::to_string(i) + ".WAV";
    check(inOrder, std::to_string(delivered.size()) + " of " + std::to_string(count) + " delivered, each once, in order");
    check(!std::filesystem::exists(outbox.path()), "... and the file gone at the end");
}

int main() {
    testFile();
    testThreads();
    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
