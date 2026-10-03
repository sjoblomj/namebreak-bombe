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

// A matches file only ever grows a few KB to low MB over a session (one line
// per Hash-A hit), so re-reading it whole on every refresh (the Windows GUI's
// ~1s tick) is cheap enough not to need a real seek-from-end tail
// implementation.
std::vector<std::string> readLastLines(const std::string& path, size_t maxLines) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    if (!in)
        return lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    if (lines.size() > maxLines)
        lines.erase(lines.begin(), lines.begin() + (lines.size() - maxLines));
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
