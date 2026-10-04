// Correctness test for readLastLines (common/matches_file.h), which reads a
// matches file's last lines from its end rather than reading it whole:
// checked against reading it whole, over files spanning many of its chunks.
// Pure CPU.
//
// Writes files under ./matches_file_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "common/matches_file.h"

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "matches_file_test";
static const std::string kPath = kDir + "/matches.txt";

static void writeFile(const std::string& contents) {
    std::filesystem::create_directories(kDir);
    std::ofstream out(kPath, std::ios::binary | std::ios::trunc);
    out << contents;
}

// The last `maxLines` lines of `contents`, split the way std::getline would,
// with a '\r' before each '\n' dropped.
static std::vector<std::string> expectedLastLines(const std::string& contents, size_t maxLines) {
    std::vector<std::string> lines;
    std::string line;
    for (size_t i = 0; i < contents.size(); ++i) {
        if (contents[i] == '\n') {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            lines.push_back(line);
            line.clear();
        } else {
            line += contents[i];
        }
    }
    if (!line.empty()) {
        if (line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    if (lines.size() > maxLines)
        lines.erase(lines.begin(), lines.begin() + (lines.size() - maxLines));
    return lines;
}

static void checkFile(const std::string& name, const std::string& contents, size_t maxLines) {
    writeFile(contents);
    std::vector<std::string> got = readLastLines(kPath, maxLines);
    std::vector<std::string> want = expectedLastLines(contents, maxLines);
    std::string what = name + ", last " + std::to_string(maxLines) + " lines";
    if (got != want)
        what += " (got " + std::to_string(got.size()) + " lines, wanted " + std::to_string(want.size()) + ")";
    check(got == want, what);
}

// Lines of matches-file-like length, some blank, and now and then one longer
// than readLastLines' chunks.
static std::string randomContents(std::mt19937& rng, size_t lineCount, const std::string& lineEnd) {
    std::uniform_int_distribution<int> length(0, 40);
    std::uniform_int_distribution<int> oneIn(0, 199);
    std::uniform_int_distribution<int> letter('A', 'Z');
    std::string contents;
    for (size_t i = 0; i < lineCount; ++i) {
        int len = oneIn(rng) == 0 ? 5000 : length(rng);
        for (int j = 0; j < len; ++j)
            contents += (char) letter(rng);
        contents += lineEnd;
    }
    return contents;
}

int main() {
    printf("readLastLines:\n");
    std::filesystem::remove(kPath);
    check(readLastLines(kPath, 100).empty(), "a file that doesn't exist yet reads as no lines");

    for (size_t maxLines : {0, 1, 2, 3, 100}) {
        checkFile("empty file", "", maxLines);
        checkFile("one line", "REZ\\AAA.WAV\n", maxLines);
        checkFile("one line, no final line break", "REZ\\AAA.WAV", maxLines);
        checkFile("three lines", "REZ\\AAA.WAV\nREZ\\AAB.WAV\nREZ\\AAC.WAV\n", maxLines);
        checkFile("three lines, no final line break", "REZ\\AAA.WAV\nREZ\\AAB.WAV\nREZ\\AAC.WAV", maxLines);
        checkFile("three lines, \\r\\n", "REZ\\AAA.WAV\r\nREZ\\AAB.WAV\r\nREZ\\AAC.WAV\r\n", maxLines);
        checkFile("blank lines", "\nREZ\\AAA.WAV\n\n\nREZ\\AAC.WAV\n\n", maxLines);
        checkFile("blank lines, \\r\\n", "\r\nREZ\\AAA.WAV\r\n\r\n\r\nREZ\\AAC.WAV\r\n\r\n", maxLines);
        checkFile("just a line break", "\n", maxLines);
        checkFile("a line longer than a chunk", std::string(10000, 'A') + "\n", maxLines);
    }

    std::mt19937 rng(12345);
    for (const std::string& lineEnd : {std::string("\n"), std::string("\r\n")}) {
        std::string ending = lineEnd == "\n" ? "\\n" : "\\r\\n";
        for (size_t lineCount : {50, 1000, 20000}) {
            std::string contents = randomContents(rng, lineCount, lineEnd);
            for (size_t maxLines : {1, 2, 99, 100, 101, 1000, 30000}) {
                std::string name = std::to_string(lineCount) + " random lines (" + std::to_string(contents.size()) + " bytes, " + ending + ")";
                checkFile(name, contents, maxLines);
                checkFile(name + ", no final line break", contents.substr(0, contents.size() - lineEnd.size()), maxLines);
            }
        }
    }

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("\nAll checks passed.\n");
    return 0;
}
