#ifndef NAMEBREAK_ENGINE_SEARCH_H
#define NAMEBREAK_ENGINE_SEARCH_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "engine/backend.h"

// One request to search a candidate space for a target MPQ hash pair. All of
// startCandidate/lowerBound/upperBound are candidate-only strings (no
// prefix/suffix) - see removePrefixAndSuffix/getStartCandidate in
// candidate.h for stripping a full filename down to just the candidate.
struct SearchRequest {
    std::string alphabet;
    int maxBackslashCount = 0;
    std::string prefix;
    std::string suffix;
    // Empty means "from the beginning": the shortest candidates (length 1),
    // from lowerBound.
    std::string startCandidate;
    std::string lowerBound;
    std::string upperBound;
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
    bool pruneSymbolRuns = false;
    // Skip candidates that close a bracket nobody opened - see
    // hasUnopenedBracket_CPU (candidate.h). Brackets opened in `prefix`
    // count as open, so a candidate may close those.
    bool pruneUnopenedBrackets = false;
    // Skip candidates with fewer backslashes than this (0: none needed). Like
    // maxBackslashCount, the prefix's own don't count. Checked as "can the
    // characters not checked yet still make up the difference?", so a
    // candidate is only skipped once its checked characters leave too few
    // after them (see canReachMinBackslashes_CPU in candidate.h). At most
    // maxBackslashCount, unless that's 0.
    int minBackslashCount = 0;
    // Skip candidates with two backslashes next to each other - including a
    // backslash at the candidate's start right after one `prefix` ends with.
    bool pruneAdjacentBackslashes = false;
    // Fixed text inserted into every candidate at least as long as its
    // position: before the candidate's character at that index from the
    // start (its length: after the last one), and before the last
    // `position` characters from the end. Both are placed on the
    // candidate's own characters - where they meet, insertFromStart's text
    // comes first. The bounds and the start candidate are without them; the
    // filenames reported have them. The rules below check inserted text like
    // the characters around it - but text inserted after a candidate's last
    // character, which is part of the suffix.
    Insertion insertFromStart;
    Insertion insertFromEnd;
    // false: the rules above (pruneSymbolRuns, pruneUnopenedBrackets,
    // maxBackslashCount, minBackslashCount, pruneAdjacentBackslashes) look
    // only at a candidate's leading characters, which the engine enumerates
    // on the CPU. true: at every character but the last - the backend skips the rows of its trailing part that break one
    // (see README.md's "Design decisions"). A candidate's last character is
    // never checked: skipping one costs a GPU as much as searching it.
    bool pruneWholeCandidate = false;
    // false ("bounded"): stop once upperBound is exhausted at its own length.
    // true ("continuous"): keep going to longer candidateLens indefinitely.
    bool continuous = false;
    // Path (relative to the current working directory, unless absolute) of
    // the file that holds the most recent Hash-A match, or the match of both
    // hashes once there is one - see MatchWriter (engine/match_writer.h),
    // which also appends that one to found.txt next to it. Missing parent
    // directories are created. The program always sets this from
    // matchesFilePath() (common/matches_file.h) - this default only matters
    // to the tests.
    std::string outputFilePath = "matches.txt";
};

struct SearchResult {
    // False only for a validation/setup problem (bad alphabet, oversized
    // prefix, lowerBound >= upperBound, matches.txt not writable, etc.) -
    // see `error` for why. Never true at the same time as `found`.
    bool ok = true;
    std::string error;
    // True once a candidate is confirmed to match *both* hashes.
    bool found = false;
    std::string filename;
    // True if `abortRequested` was observed set mid-search - the caller
    // should treat this as neither found nor exhausted, just interrupted.
    bool aborted = false;
};

// Runs an exhaustive (bounded) or open-ended (continuous) search over
// `req`'s candidate space on `backend`, keeping its most recent Hash-A match
// in `req.outputFilePath` - or the match of both hashes, once it finds one.
// `abortRequested`, if given, is polled between batches so a caller can
// interrupt a long-running search (e.g. once a coordinator learns the target
// was solved elsewhere); `onPartialMatch`, if given, is invoked with the
// full filename of every Hash-A match as soon as it's known.
// `pauseRequested`, if given, is polled the same way abortRequested is - but
// instead of ending the search, blocks (still letting a caller's own
// heartbeat/etc. threads run) until it's cleared again, so the batch already
// in flight always finishes normally and only the *next* one is held back.
// See src/cli/main.cpp's main() for the key listener that drives this.
//
// Safe to call more than once in the same process, with the same backend
// (e.g. once per claimed coordinator range): all backend state this depends
// on is reset at the start of each call.
SearchResult runSearch(SearchBackend& backend, const SearchRequest& req,
                       std::atomic<bool>* abortRequested = nullptr,
                       std::function<void(const std::string&)> onPartialMatch = nullptr,
                       const std::atomic<bool>* pauseRequested = nullptr);

#endif // NAMEBREAK_ENGINE_SEARCH_H
