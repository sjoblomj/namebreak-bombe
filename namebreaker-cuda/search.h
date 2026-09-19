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
    // Path (relative to the current working directory, unless absolute)
    // every Hash-A-only match is appended to. Defaults to the plain local
    // name; coordinator_runner.cpp's toSearchRequest overrides this per
    // claimed range so concurrent/successive targets don't clobber each
    // other's matches.
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
// `req`'s candidate space, appending every Hash-A-only match to
// `req.outputFilePath`. `abortRequested`, if given, is polled
// between batches so a caller can interrupt a long-running search (e.g. once
// a coordinator learns the target was solved elsewhere); `onPartialMatch`,
// if given, is invoked with the full filename of every Hash-A-only match as
// soon as it's known. `pauseRequested`, if given, is polled the same way
// abortRequested is - but instead of ending the search, blocks (still
// letting a caller's own heartbeat/etc. threads run) until it's cleared
// again, so the batch already in flight always finishes normally and only
// the *next* one is held back. See namebreak.cu's main() for the key
// listener that drives this.
//
// Safe to call more than once in the same process (e.g. once per claimed
// coordinator range): all device state this depends on is reset at the
// start of each call.
SearchResult runSearch(const SearchRequest& req,
                        std::atomic<bool>* abortRequested = nullptr,
                        std::function<void(const std::string&)> onPartialMatch = nullptr,
                        const std::atomic<bool>* pauseRequested = nullptr);

#endif // NAMEBREAK_CUDA_SEARCH_H
