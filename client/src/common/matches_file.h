#ifndef NAMEBREAK_COMMON_MATCHES_FILE_H
#define NAMEBREAK_COMMON_MATCHES_FILE_H

#include <cstddef>
#include <string>
#include <vector>

#include "engine/candidate.h"

// The files a search keeps its most recent Hash-A match in, as a full
// filename on one line (see MatchWriter, engine/match_writer.h - which also
// appends a match of both hashes to found.txt beside them): all of them live
// in one directory (config.conf's `matches_dir`, default kDefaultMatchesDir,
// relative to the current working directory unless absolute), created on
// first use. A local search writes matches.txt there; each coordinator
// target gets its own matches-<target name>.txt, so concurrent/successive
// targets don't clobber each other's matches.

inline constexpr const char* kDefaultMatchesDir = "matches";

// The matches file for `targetName` (a coordinator target's name) in
// `matchesDir` - or, if `targetName` is empty, the local search's.
std::string matchesFilePath(const std::string& matchesDir, const std::string& targetName);

// Reads up to the last `maxLines` lines of `path` (oldest first, without
// their "\n" or "\r\n") - empty if it doesn't exist (yet). Reads only the end
// of the file, however big it is.
std::vector<std::string> readLastLines(const std::string& path, size_t maxLines);

// Fraction (0.0-1.0) of the way `matchFilename` (a full prefix+candidate+
// suffix line from a matches file) sits between `lowerBound` and
// `upperBound`, or -1.0 if it can't be computed - see matches_file.cpp.
double matchProgressFraction(const std::string& matchFilename, const std::string& prefix, const std::string& suffix,
                             const Insertion& insertFromStart, const Insertion& insertFromEnd, const std::string& alphabet,
                             const std::string& lowerBound, const std::string& upperBound);

#endif // NAMEBREAK_COMMON_MATCHES_FILE_H
