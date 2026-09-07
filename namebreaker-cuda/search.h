#ifndef NAMEBREAK_CUDA_SEARCH_H
#define NAMEBREAK_CUDA_SEARCH_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

// One request to search a candidate space for a target MPQ hash pair. All of
// startCandidate/lowerBound/upperBound are candidate-only strings (no
// prefix/suffix) - see remove_prefix_and_suffix/getStartCandidate in
// cpu-utils.h for stripping a full filename down to just the candidate.
struct SearchRequest {
    std::string alphabet;
    int maxBackslashCount = 0;
    std::string prefix;
    std::string suffix;
    std::string startCandidate;
    std::string lowerBound;
    std::string upperBound;
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
    bool pruneSymbolRuns = false;
    // false ("bounded"): stop once upperBound is exhausted at its own length.
    // true ("continuous"): keep going to longer candidateLens indefinitely.
    bool continuous = false;
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
// `req`'s candidate space, appending every Hash-A-only match to matches.txt
// in the current working directory (same file/behavior as before this was
// extracted out of main()). `abortRequested`, if given, is polled between
// batches so a caller can interrupt a long-running search (e.g. once a
// coordinator learns the target was solved elsewhere); `onPartialMatch`, if
// given, is invoked with the full filename of every Hash-A-only match as
// soon as it's known - the same checkpoint signal the old subprocess-based
// client used to get by scanning stdout for "Hash A matches: ", now
// delivered directly since the search runs in-process.
//
// Safe to call more than once in the same process (e.g. once per claimed
// coordinator range): all device state this depends on is reset at the
// start of each call.
SearchResult runSearch(const SearchRequest& req,
                        std::atomic<bool>* abortRequested = nullptr,
                        std::function<void(const std::string&)> onPartialMatch = nullptr);

#endif // NAMEBREAK_CUDA_SEARCH_H
