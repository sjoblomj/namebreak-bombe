// Correctness test for readLastLines (common/matches_file.h), which reads a
// matches file's last lines from its end rather than reading it whole:
// checked against reading it whole, over files spanning many of its chunks.
// And for MatchWriter (engine/match_writer.h), which keeps a matches file at
// one line - the match of both hashes, once there is one. Pure CPU.
//
// Writes files under ./matches_file_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "common/matches_file.h"
#include "engine/match_writer.h"

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

// All of `path`, or "" if it doesn't exist.
static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return contents;
}

// What a line written in text mode reads as in binary.
static std::string asWritten(const std::string& lines) {
#ifdef _WIN32
    std::string out;
    for (char c : lines)
        out += c == '\n' ? std::string("\r\n") : std::string(1, c);
    return out;
#else
    return lines;
#endif
}

static void testMatchWriter() {
    printf("MatchWriter:\n");
    const std::string dir = kDir + "/writer/sub";
    const std::string path = dir + "/matches-x.txt";
    const std::string found = foundFilePath(path);
    std::filesystem::remove_all(kDir);

    std::string error;
    {
        MatchWriter writer(std::chrono::milliseconds(0));
        check(writer.open(path, error), "open() creates the missing directories");
        check(std::filesystem::exists(path) && readFile(path).empty(), "... and an empty matches file");
        // As paths: on Windows, found.txt is joined on with a backslash.
        check(std::filesystem::path(found) == std::filesystem::path(dir + "/found.txt"), "found.txt is beside the matches file");
        writer.hit("REZ\\AAA.WAV");
        check(readFile(path).empty(), "a hit isn't written before writeIfDue()");
        writer.writeIfDue();
        check(readFile(path) == asWritten("REZ\\AAA.WAV\n"), "writeIfDue() writes it");
        writer.hit("REZ\\AAB.WAV");
        writer.hit("REZ\\AAC.WAV");
        writer.writeIfDue();
        check(readFile(path) == asWritten("REZ\\AAC.WAV\n"), "the latest hit replaces it, with no interval");
        check(!std::filesystem::exists(path + ".tmp"), "nothing is left beside it");
        check(!std::filesystem::exists(found), "found.txt isn't written until something is found");
    }
    {
        {
            // An older version's file, every hit appended.
            std::ofstream old(path, std::ios::trunc);
            old << std::string(100000, 'A') << "\nREZ\\OLD.WAV\n";
        }
        MatchWriter writer(std::chrono::hours(1));
        check(writer.open(path, error) && readFile(path).size() > 100000, "open() leaves what the file holds");
        writer.hit("REZ\\AAA.WAV");
        writer.writeIfDue();
        check(readFile(path) == asWritten("REZ\\AAA.WAV\n"), "the first hit is written at once, replacing an old long file");
        writer.hit("REZ\\AAB.WAV");
        writer.writeIfDue();
        check(readFile(path) == asWritten("REZ\\AAA.WAV\n"), "the next isn't, within the interval");
        writer.flush();
        check(readFile(path) == asWritten("REZ\\AAB.WAV\n"), "flush() writes it");
        writer.flush();
        check(readFile(path) == asWritten("REZ\\AAB.WAV\n"), "flush() with nothing new leaves it");

        writer.hit("REZ\\AAC.WAV");
        check(writer.found("REZ\\FND.WAV"), "found() succeeds");
        check(readFile(path) == asWritten("REZ\\FND.WAV\n"), "found() writes the match at once, over a hit not written yet");
        check(readFile(found) == asWritten("REZ\\FND.WAV\n"), "... and to found.txt");
        writer.flush();
        writer.hit("REZ\\AAD.WAV");
        writer.writeIfDue();
        writer.flush();
        check(readFile(path) == asWritten("REZ\\FND.WAV\n"), "no hit replaces it, before or after it");
    }
    {
        MatchWriter writer(std::chrono::milliseconds(0));
        check(writer.open(path, error), "open() again");
        check(readFile(path) == asWritten("REZ\\FND.WAV\n"), "... still leaves it");
        writer.hit("REZ\\AAE.WAV");
        writer.writeIfDue();
        check(readFile(path) == asWritten("REZ\\AAE.WAV\n"), "another search's hit replaces it");
        writer.found("REZ\\FN2.WAV");
        check(readFile(found) == asWritten("REZ\\FND.WAV\nREZ\\FN2.WAV\n"), "but found.txt keeps every match, the latest last");
    }
    {
        MatchWriter writer;
        error.clear();
        check(!writer.open(dir, error) && !error.empty(), "open() fails, saying why, for a path it can't write");
    }
}

static void testFilePaths() {
    printf("namedFilePath and matchesFilePath:\n");
    check(namedFilePath("matches", "basenames", "") == "matches/basenames.txt", "no name: <base>.txt");
    check(namedFilePath("matches", "wordnumber", "credits") == "matches/wordnumber-credits.txt", "a name: <base>-<name>.txt");
    check(namedFilePath("m/", "basenames", "x") == "m/basenames-x.txt" && namedFilePath("m\\", "basenames", "x") == "m\\basenames-x.txt",
          "a directory ending in a separator: no second one");
    check(namedFilePath("", "basenames", "x") == "basenames-x.txt", "no directory: the current one");
    check(namedFilePath("m", "basenames", "../a\\b c") == "m/basenames-.._a_b_c.txt", "a name with path separators: made safe");
    check(matchesFilePath("m", "") == "m/matches.txt" && matchesFilePath("m", "REZ\\X") == "m/matches-REZ_X.txt",
          "matchesFilePath: namedFilePath with base matches");
    // replaceFileContents, which the progress file is written with.
    std::filesystem::create_directories(kDir);
    std::string error;
    check(replaceFileContents(kPath, "one\ntwo\n", error), "replaceFileContents: written");
    std::ifstream in(kPath);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(text == "one\ntwo\n" && !std::filesystem::exists(kPath + ".tmp"), "... replacing what was there, no temporary file left");
    check(!replaceFileContents(kDir + "/no/such/dir/file.txt", "x", error) && !error.empty(), "... false, saying why, where it can't be");
}

int main() {
    testMatchWriter();
    testFilePaths();

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
