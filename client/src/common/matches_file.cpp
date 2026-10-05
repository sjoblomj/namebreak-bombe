#include "common/matches_file.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>

#include "engine/candidate.h"

namespace {

// Turns a target name (an admin-chosen label, not guaranteed filesystem-safe -
// e.g. it may contain the same backslashes a target MPQ filename would) into
// a bare filename component: every character other than
// alnum/'.'/'-'/'_' becomes '_', so it can't smuggle in a path separator
// (or ".." for one) and land outside the matches directory.
std::string sanitizeForFilename(const std::string& name) {
    std::string out = name;
    for (char& c : out) {
        if (!isalnum((unsigned char) c) && c != '.' && c != '-' && c != '_')
            c = '_';
    }
    return out.empty() ? "target" : out;
}

} // namespace

std::string matchesFilePath(const std::string& matchesDir, const std::string& targetName) {
    std::string name = targetName.empty() ? "matches.txt" : "matches-" + sanitizeForFilename(targetName) + ".txt";
    if (matchesDir.empty())
        return name;
    char last = matchesDir.back();
    return (last == '/' || last == '\\') ? matchesDir + name : matchesDir + "/" + name;
}

// Reads only the end of the file: a matches file is one line now, but older
// versions appended every Hash-A hit to it - hundreds a second on a fast
// GPU - for as long as a target ran, so one they left behind can be hundreds
// of MB until a search replaces it, and the Windows GUI calls this on every
// ~1s tick. Reading such a file whole took most of a second and most of a GB
// each time, which left the GUI's UI thread with no time for anything else.
std::vector<std::string> readLastLines(const std::string& path, size_t maxLines) {
    std::vector<std::string> lines;
    // Binary, since a text-mode stream can't be seeked to a byte offset
    // reliably on Windows - so the '\r' a Windows text-mode writer puts
    // before each '\n' is stripped below instead.
    std::ifstream in(path, std::ios::binary);
    if (!in || maxLines == 0)
        return lines;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0)
        return lines;

    // Walks back from the end a chunk at a time to the line break just before
    // the oldest wanted line - or to the start of the file. The file's own
    // final '\n' ends its last line rather than starting a new one, so it
    // doesn't count.
    std::streamoff start = 0;
    size_t breaks = 0;
    char chunk[4096];
    for (std::streamoff chunkEnd = size; chunkEnd > 0 && start == 0;) {
        std::streamoff chunkStart = std::max<std::streamoff>(0, chunkEnd - (std::streamoff) sizeof(chunk));
        in.seekg(chunkStart);
        if (!in.read(chunk, chunkEnd - chunkStart))
            return lines;
        for (std::streamoff i = chunkEnd - 1; i >= chunkStart; --i) {
            if (chunk[i - chunkStart] == '\n' && i != size - 1 && ++breaks == maxLines) {
                start = i + 1;
                break;
            }
        }
        chunkEnd = chunkStart;
    }

    std::string tail((size_t) (size - start), '\0');
    in.seekg(start);
    if (!in.read(&tail[0], (std::streamsize) tail.size()))
        return lines;
    for (size_t lineStart = 0; lineStart < tail.size();) {
        size_t lineEnd = std::min(tail.find('\n', lineStart), tail.size());
        size_t contentEnd = (lineEnd > lineStart && tail[lineEnd - 1] == '\r') ? lineEnd - 1 : lineEnd;
        lines.push_back(tail.substr(lineStart, contentEnd - lineStart));
        lineStart = lineEnd + 1;
    }
    return lines;
}

// Fraction (0.0-1.0) of the way `matchFilename` (a full prefix+candidate+
// suffix line from the matches file) sits between `lowerBound` and
// `upperBound` in `alphabet`'s enumeration order (candidate.h's
// stringToIndex - the same fixed odometer order runSearch() itself
// enumerates candidates in). Returns -1.0 if it can't be computed (mismatched
// candidate lengths - shouldn't happen for a single bounded-shaped range/
// search, but a match line surviving from a previous, differently-shaped run
// on the same file is possible - malformed alphabet, etc.); callers should
// treat that as "no usable progress signal yet" rather than a hard error.
double matchProgressFraction(const std::string& matchFilename, const std::string& prefix, const std::string& suffix,
                             const Insertion& insertFromStart, const Insertion& insertFromEnd, const std::string& alphabet,
                             const std::string& lowerBound, const std::string& upperBound) {
    std::string candidate, error;
    if (!candidateOfFilename(matchFilename, prefix, suffix, insertFromStart, insertFromEnd, candidate, error))
        return -1.0;
    if (candidate.length() != lowerBound.length() || lowerBound.length() != upperBound.length())
        return -1.0;
    uint64_t lowerIdx = 0, upperIdx = 0, matchIdx = 0;
    if (!stringToIndex(lowerBound, alphabet, lowerIdx, error) || !stringToIndex(upperBound, alphabet, upperIdx, error) ||
        !stringToIndex(candidate, alphabet, matchIdx, error))
        return -1.0;
    if (upperIdx <= lowerIdx)
        return -1.0;
    matchIdx = std::min(std::max(matchIdx, lowerIdx), upperIdx);
    return double(matchIdx - lowerIdx) / double(upperIdx - lowerIdx);
}
