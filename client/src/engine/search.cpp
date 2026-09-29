#include "engine/search.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "engine/candidate.h"
#include "engine/limits.h"
#include "engine/mpq_hash.h"

std::string describeAlphabetSizes(const SearchBackend& backend) {
    std::vector<int> sizes = backend.supportedAlphabetSizes();
    if (sizes.empty())
        return "any size from 1 to " + std::to_string(MAX_ALPHABET_SIZE);
    std::string text;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0)
            text += (i + 1 == sizes.size()) ? " or " : ", ";
        text += std::to_string(sizes[i]);
    }
    return text;
}

namespace {

// Searches the trailing indices [startIdx, startIdx + count) (count > 0) with
// one backend.runBatch() call, and writes out what it found. Returns 0 (no
// match yet), 1 (found - both hashes matched, outFoundFilename is filled), or
// -1 (abortRequested was set, this batch was skipped).
//
// Every hashA hit is verified against hashB: if a batch has more hits than
// MAX_MATCHES (which needs a target hashA with over a thousand matches among
// the range's candidates - not something a real 32-bit hash produces, but the
// one outcome that must never happen is a both-hashes match going
// unchecked), the range is searched again as two halves, recursively.
int searchChunk(SearchBackend& backend, int trailingLen, uint64_t startIdx, uint64_t count, const BatchParams& params, FILE* fout,
                const std::atomic<bool>* abortRequested, const std::function<void(const std::string&)>& onPartialMatch,
                std::string& outFoundFilename, const std::atomic<bool>* pauseRequested) {
    if (abortRequested && abortRequested->load(std::memory_order_relaxed))
        return -1;

    // Called between batches only (runBatch returns once its batch is
    // finished), so a batch already in flight is never interrupted - pausing
    // here just means the *next* batch waits. Keeps polling abortRequested
    // too, so a pause that outlasts the target being solved elsewhere doesn't
    // block forever.
    while (pauseRequested && pauseRequested->load(std::memory_order_relaxed)) {
        if (abortRequested && abortRequested->load(std::memory_order_relaxed))
            return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    BatchOutcome outcome = backend.runBatch(trailingLen, startIdx, count, params);

    // Hits are rare (a few per thousand batches), so everything below is the
    // exception path.
    if (outcome.hitCount > MAX_MATCHES) {
        // More hits than a batch can record: the excess would never be
        // checked against hashB. Nothing from this batch has been verified or
        // reported yet, so discard it and search each half of the range on
        // its own instead (splitting again as needed). A one-candidate range
        // has at most one hit and MAX_MATCHES >= 1, so count >= 2 here and
        // this always terminates.
        fprintf(stderr, "note: %d hashA hits in one batch, more than the %d that can be recorded - searching its two halves separately\n",
                outcome.hitCount, MAX_MATCHES);
        const uint64_t half = count / 2;
        int r = searchChunk(backend, trailingLen, startIdx, half, params, fout, abortRequested, onPartialMatch, outFoundFilename,
                            pauseRequested);
        if (r != 0)
            return r;
        return searchChunk(backend, trailingLen, startIdx + half, count - half, params, fout, abortRequested, onPartialMatch,
                           outFoundFilename, pauseRequested);
    }
    for (const std::string& hit : outcome.hits) {
        printf("%s\n", hit.c_str());
        fprintf(fout, "%s\n", hit.c_str());
        fflush(fout);
        if (onPartialMatch)
            onPartialMatch(hit);
    }
    if (outcome.found) {
        printf("%s\n", outcome.foundFilename.c_str());
        printf("BOTH HASHES MATCH: %s\n", outcome.foundFilename.c_str());
        outFoundFilename = outcome.foundFilename;
        return 1;
    }
    return 0;
}

} // namespace

SearchResult runSearch(SearchBackend& backend, const SearchRequest& req, std::atomic<bool>* abortRequested,
                       std::function<void(const std::string&)> onPartialMatch, const std::atomic<bool>* pauseRequested) {
    SearchResult result;

    if (req.alphabet.empty() || req.alphabet.size() > MAX_ALPHABET_SIZE) {
        result.ok = false;
        result.error = "Alphabet must be non-empty and at most " + std::to_string(MAX_ALPHABET_SIZE) + " characters (got " + std::to_string(req.alphabet.size()) + ")";
        return result;
    }
    int alphabetSize = (int) req.alphabet.size();
    std::vector<int> supportedSizes = backend.supportedAlphabetSizes();
    if (!supportedSizes.empty() && std::find(supportedSizes.begin(), supportedSizes.end(), alphabetSize) == supportedSizes.end()) {
        result.ok = false;
        result.error = "Unsupported alphabet size: " + std::to_string(alphabetSize) + " (the " + backend.name() + " backend supports " +
                       describeAlphabetSizes(backend) + ")";
        return result;
    }
    // 0 means unlimited (see hasForbiddenSymbolRun_CPU/countBackslashes_CPU's
    // declaration comment in candidate.h for how and where this is applied).
    if (req.maxBackslashCount < 0) {
        result.ok = false;
        result.error = "maxBackslashCount must be >= 0 (0 means unlimited), got " + std::to_string(req.maxBackslashCount);
        return result;
    }

    // Compare lower/upper using the same alphabet ordering the rest of the search
    // relies on, rather than raw string comparison (which would break if the
    // alphabet's character order ever stopped matching ASCII order).
    bool lowerIsBeforeUpper = false;
    if (!isBeforeInAlphabet(req.lowerBound, req.upperBound, req.alphabet, lowerIsBeforeUpper, result.error)) {
        result.ok = false;
        return result;
    }
    // Equal bounds are allowed - not a special case, just the search space
    // collapsing to exactly one candidate at whatever length they're given
    // at (the rest of runSearch already handles a one-candidate batch/leading
    // value fine, since that's an ordinary shape for the *last* batch of any
    // search - this just makes it a legal shape for the *whole* search too).
    // In `continuous` mode this also has a second, useful reading: since
    // lowerBoundLimit/upperBoundLimit (getLowerBound/getUpperBound below)
    // extend outward from whatever's given as candidateLen grows, equal
    // bounds naturally become "every candidate with this exact string as a
    // prefix" once the search moves past this length - not a coincidence
    // worth special-casing, just what the existing widening already does.
    if (!lowerIsBeforeUpper && req.lowerBound != req.upperBound) {
        result.ok = false;
        result.error = "lower bound ('" + req.lowerBound + "') must not be greater than upper bound ('" + req.upperBound + "')";
        return result;
    }

    short prefix_size = req.prefix.size();
    short suffix_size = req.suffix.size();

    // Largest candidate length stringToIndex/indexToString can safely convert
    // to/from a uint64_t index without overflow (alphabetSize^N <= UINT64_MAX).
    // Computed from alphabetSize/MAX_CANDIDATE_LEN rather than hardcoded, so it
    // stays correct if either changes (a smaller alphabet allows a longer safe
    // length). This only bounds leadingLen below (see windowChars and
    // trailingLen's computation in the loop below) - trailingLen itself never
    // gets anywhere near this limit.
    int maxSafeIndexLen = 0;
    {
        uint64_t product = 1;
        while (maxSafeIndexLen < MAX_CANDIDATE_LEN && product <= UINT64_MAX / alphabetSize) {
            product *= alphabetSize;
            maxSafeIndexLen++;
        }
    }

    // How many trailing candidate characters go straight to the backend as a
    // native index each batch; the rest is folded into an extended prefix and
    // hashed on the CPU instead, incrementally (IncrementalPrefixHasher,
    // mpq_hash.h) rather than from scratch per leading value. Deliberately
    // small and fixed - NOT "as large as maxSafeIndexLen allows", which is
    // what this project used to do (and still needs to fall back toward for
    // very long candidates - see trailingLen's computation below). See
    // README.md's "Design decisions" section for why.
    const int windowChars = backend.windowChars();

    int maxLeadingLen = maxSafeIndexLen;
    printf("backend: %s, windowChars: %d, maxSafeIndexLen: %d (max leading/prefix-extension length: %d)\n",
           backend.name(), windowChars, maxSafeIndexLen, maxLeadingLen);

    if (prefix_size + maxLeadingLen >= kMaxPrefixSize || suffix_size >= kMaxSuffixSize) {
        result.ok = false;
        result.error = "prefix (up to " + std::to_string(prefix_size + maxLeadingLen) + " once extended by leading candidate characters) or suffix (" +
                        std::to_string(suffix_size) + ") too long for the search's buffers (max: " + std::to_string(kMaxPrefixSize) + " each)";
        return result;
    }
    if (prefix_size + suffix_size + MAX_CANDIDATE_LEN >= MAX_FILENAME_LEN) {
        result.ok = false;
        result.error = "prefix (" + std::to_string(prefix_size) + ") + suffix (" + std::to_string(suffix_size) + ") + candidate (up to " +
                        std::to_string(MAX_CANDIDATE_LEN) + ") would exceed MAX_FILENAME_LEN (" + std::to_string(MAX_FILENAME_LEN) + ")";
        return result;
    }

    std::string lowerBoundLimit, upperBoundLimit;
    if (!getLowerBound(req.lowerBound, req.alphabet, lowerBoundLimit, result.error) ||
        !getUpperBound(req.upperBound, req.alphabet, upperBoundLimit, result.error)) {
        result.ok = false;
        return result;
    }
    // upperBoundLimit is an *exclusive* end (the successor of req.upperBound), but an
    // upper bound of all-maximum characters has no successor: getUpperBound saturates
    // to that maximum itself, which used as an exclusive end would silently leave the
    // very last candidate of the candidate length unsearched. Detect that case so the
    // end is treated as "everything up to and including the last candidate" instead.
    const bool upperBoundIsAbsoluteMax =
        !req.upperBound.empty() && req.upperBound.find_first_not_of(req.alphabet.back()) == std::string::npos;

    printf("alphabet: '%s' (size %d)\n", req.alphabet.c_str(), alphabetSize);
    printf("candidate: '%s'\n", req.startCandidate.c_str());
    printf("prefix: '%s'\n", req.prefix.c_str());
    printf("suffix: '%s'\n", req.suffix.c_str());
    printf("lower: '%s'\n", req.lowerBound.c_str());
    printf("upper: '%s'\n", req.upperBound.c_str());
    printf("lowerBoundLimit: '%s'\n", lowerBoundLimit.c_str());
    printf("upperBoundLimit: '%s'\n", upperBoundLimit.c_str());
    printf("hashA: '%X'\n", req.targetHashA);
    printf("hashB: '%X'\n", req.targetHashB);
    printf("pruneSymbolRuns: %s (leading characters only)\n", req.pruneSymbolRuns ? "true" : "false");
    printf("pruneUnopenedBrackets: %s (leading characters only)\n", req.pruneUnopenedBrackets ? "true" : "false");
    printf("maxBackslashCount: %d%s (leading characters only)\n", req.maxBackslashCount,
           req.maxBackslashCount == 0 ? " (unlimited)" : "");

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);

    // Hash of req.prefix alone (never changes across candidateLen or
    // leadingIdx) - the base every leadingIdx loop's IncrementalPrefixHasher
    // extends by that iteration's leading characters.
    std::pair<uint32_t, uint32_t> prefixBaseState = mpqHashWithPrefixCache_CPU(req.prefix.c_str(), h_cryptTable);

    // Brackets req.prefix leaves open, which a candidate is free to close -
    // see req.pruneUnopenedBrackets.
    const OpenBrackets prefixOpenBrackets = openBracketsAfter_CPU(req.prefix);

    std::filesystem::path outputDir = std::filesystem::path(req.outputFilePath).parent_path();
    if (!outputDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outputDir, ec);
        if (ec) {
            result.ok = false;
            result.error = "cannot create " + outputDir.string() + ": " + ec.message();
            return result;
        }
    }
    FILE* fout = fopen(req.outputFilePath.c_str(), "a");
    if (!fout) {
        result.ok = false;
        result.error = std::string("fopen ") + req.outputFilePath + ": " + strerror(errno);
        return result;
    }

    SearchConstants constants;
    constants.alphabet = req.alphabet;
    constants.suffix = req.suffix;
    constants.cryptTable = h_cryptTable;
    constants.targetHashA = req.targetHashA;
    constants.targetHashB = req.targetHashB;
    backend.beginSearch(constants);

    bool found_match = false;
    bool aborted = false;
    std::string foundFilename;
    std::string start_candidate = req.startCandidate;
    int candidateLen = start_candidate.size();
    // Candidates per batch - see SearchBackend::batchSize.
    const uint64_t batchSize = backend.batchSize(alphabetSize);

    // The search space is walked by four nested levels, outermost to innermost:
    //  1. This `while` loop: over candidateLen itself - "try every 1-character
    //     candidate, then every 2-character one, ..." Only continuous mode
    //     (req.continuous) actually loops here more than once; bounded mode
    //     runs the body for req.startCandidate's own length and exits via the
    //     `if (!req.continuous)` check at the bottom.
    //  2. The `for (leadingIdx ...)` loop below: over the *leading* part of the
    //     candidate (see windowChars/leadingLen above) - every leading value
    //     is hashed on the CPU (incrementally - IncrementalPrefixHasher) and
    //     handed to the backend as its starting seed, so candidates longer than
    //     windowChars still get covered exhaustively. Since windowChars
    //     is small (throughput-tuned, not "as large as safely possible"),
    //     leadingLen > 0 - and this loop actually doing work - is the common
    //     case, not the exception; it only degenerates to a single iteration
    //     when candidateLen <= windowChars.
    //  3. The `for (i = trailStart ...)` loop: chops the (up to
    //     alphabetSize^trailingLen) remaining space for one leading value into
    //     batchSize-sized chunks (aligned to multiples of batchSize, so only a
    //     range's own first/last chunk can start/end mid-row) - each iteration
    //     is one searchChunk call, i.e. one SearchBackend::runBatch (for the
    //     CUDA backend: one kernel launch, plus a second, tiny one only if
    //     that launch had a hashA hit).
    //  4. Inside the backend - for the CUDA backend, one GPU thread per *row*
    //     of the chunk, each hashing only the few of its row's alphabetSize
    //     candidates its lookup filter lets through - see filteredRowsKernel's
    //     comment.
    if (candidateLen < 1) {
        // An empty start candidate (prefix + suffix alone) means "from the
        // very beginning": the shortest candidates, starting from the lower
        // bound the way every length after the first does below. The empty
        // candidate itself is skipped - a backend enumerates at least one
        // trailing character.
        candidateLen = 1;
        start_candidate = lowerBoundLimit;
    }
    while (true) {
        if (candidateLen > MAX_CANDIDATE_LEN) {
            fprintf(stderr, "candidateLen (%d) exceeds MAX_CANDIDATE_LEN (%d) - exiting\n", candidateLen, MAX_CANDIDATE_LEN);
            break;
        }

        // Split the candidate into a leading part (folded into the prefix, hashed
        // incrementally on the CPU) and a trailing part of windowChars characters
        // (brute-forced by the backend with native 64-bit indices) - or, once candidateLen
        // grows large enough that leadingLen would exceed maxSafeIndexLen (overflow the
        // index conversions below), a slightly larger trailing part, just big enough to
        // keep leadingLen within that safe limit. Every combination of the leading part
        // is enumerated too, so the full candidateLen-character space is still covered
        // exhaustively either way.
        int trailingLen = std::min(candidateLen, std::max(windowChars, candidateLen - maxSafeIndexLen));
        int leadingLen = candidateLen - trailingLen;
        if (trailingLen > backend.maxTrailingLen()) {
            // Only reachable if MAX_CANDIDATE_LEN/the alphabet sizes change so that
            // candidateLen - maxSafeIndexLen exceeds what the backend supports (for
            // CUDA, filteredRowsKernel's 32-bit row index) - refuse rather than overflow it.
            fprintf(stderr, "candidateLen (%d) needs a %d-character trailing part, which exceeds the %s backend's maximum (%d) - exiting\n",
                    candidateLen, trailingLen, backend.name(), backend.maxTrailingLen());
            break;
        }
        if (leadingLen > maxSafeIndexLen) {
            // Not expected to ever trigger given trailingLen's formula above (it's
            // constructed specifically to keep leadingLen <= maxSafeIndexLen) - kept as
            // a belt-and-suspenders check against future edits to that formula, since a
            // silent overflow here would mean silently skipped candidates, not a crash.
            fprintf(stderr, "candidateLen (%d) needs a %d-character leading part, which exceeds maxSafeIndexLen (%d) - exiting\n",
                    candidateLen, leadingLen, maxSafeIndexLen);
            break;
        }

        std::string start_full = makeBoundString(start_candidate, candidateLen);
        std::string end_full   = makeBoundString(upperBoundLimit, candidateLen);

        std::string start_leading = start_full.substr(0, leadingLen);
        std::string end_leading   =   end_full.substr(0, leadingLen);

        uint64_t startLeadingIdx = 0, endLeadingIdx = 0, trailSpaceSize = 0;
        if (!stringToIndex(start_leading, req.alphabet, startLeadingIdx, result.error) ||
            !stringToIndex(  end_leading, req.alphabet,   endLeadingIdx, result.error) ||
            !stringToIndex(std::string(trailingLen, req.alphabet.back()), req.alphabet, trailSpaceSize, result.error)) {
            result.ok = false;
            goto breakfree;
        }
        trailSpaceSize += 1;

        // Cap progress logging to roughly 1000 lines per candidateLen, regardless of how
        // large the leading space is - printing once per leading combination is fine
        // when leadingLen is 0 (a single iteration), but would flood stdout (and cost
        // real time) once leadingLen grows.
        uint64_t leadingCount = endLeadingIdx - startLeadingIdx + 1;
        uint64_t leadingLogInterval = std::max<uint64_t>(1, leadingCount / 1000);

        // Level 2 (see the walkthrough above the outer `while`). leadingHasher
        // walks every leading value from startLeadingIdx to endLeadingIdx in
        // the same order this loop does (always +1), so every step after the
        // first is an O(1)-amortized incremental update (IncrementalPrefixHasher,
        // mpq_hash.h) instead of a full from-scratch re-hash of the whole
        // leading string - the thing that makes leadingLen > 0 being the
        // common case (see windowChars above) affordable.
        IncrementalPrefixHasher leadingHasher(prefixBaseState, leadingLen, req.alphabet, h_cryptTable);
        leadingHasher.reset(startLeadingIdx);
        for (uint64_t leadingIdx = startLeadingIdx; leadingIdx <= endLeadingIdx; ++leadingIdx) {
            if (leadingIdx != startLeadingIdx) {
                leadingHasher.advance();
            }
            const std::string& leading = leadingHasher.leading();

            // req.pruneSymbolRuns/req.pruneUnopenedBrackets/req.maxBackslashCount examine `leading` -
            // the CPU-computed first leadingLen characters of the candidate
            // (see README.md's "Design decisions" section for why checking it
            // here, instead of in the backend, is worth doing). A prune here
            // skips this leading value's entire trailing batch (up to
            // batchSize candidates) without spending anything in the
            // backend (for a GPU, not even a kernel launch) - cheaper than
            // even one of those candidates would have cost individually.
            if (req.pruneSymbolRuns && hasForbiddenSymbolRun_CPU(leading))
                continue;
            if (req.maxBackslashCount != 0 && countBackslashes_CPU(leading) > req.maxBackslashCount)
                continue;
            if (req.pruneUnopenedBrackets && hasUnopenedBracket_CPU(leading, prefixOpenBrackets))
                continue;

            // Handed to every runBatch call below; see BatchParams.
            std::string extendedPrefix = req.prefix + leading;
            BatchParams params;
            memcpy(params.prefix, extendedPrefix.c_str(), extendedPrefix.size() + 1);
            params.prefixSize = (short) extendedPrefix.size();
            std::pair<uint32_t, uint32_t> pair = leadingHasher.state();
            params.seed1Start = pair.first;
            params.seed2Start = pair.second;

            uint64_t trailStart = 0, trailEnd = 0;
            if (leadingIdx == startLeadingIdx) {
                if (!stringToIndex(start_full.substr(leadingLen), req.alphabet, trailStart, result.error)) {
                    result.ok = false;
                    goto breakfree;
                }
            }
            if (leadingIdx == endLeadingIdx && !upperBoundIsAbsoluteMax) {
                if (!stringToIndex(end_full.substr(leadingLen), req.alphabet, trailEnd, result.error)) {
                    result.ok = false;
                    goto breakfree;
                }
            } else {
                trailEnd = trailSpaceSize;
            }

            // Level 3 (see the walkthrough above the outer `while`).
            for (uint64_t i = trailStart; i < trailEnd; ) {
                uint64_t chunkEnd = std::min(trailEnd, (i / batchSize + 1) * batchSize);
                uint64_t count = chunkEnd - i;
                int r = searchChunk(backend, trailingLen, i, count, params, fout, abortRequested, onPartialMatch, foundFilename,
                                    pauseRequested);
                i = chunkEnd;
                if (r == -1) {
                    aborted = true;
                    goto breakfree;
                }
                if (r == 1) {
                    found_match = true;
                    goto breakfree;
                }
            }
        }

        candidateLen += 1;
        start_candidate = lowerBoundLimit;
        if (!req.continuous) {
            printf("Reached the upper limit - exiting\n");
            goto breakfree;
        }
    }
breakfree:

    backend.endSearch();
    fclose(fout);

    result.aborted = aborted;
    result.found = found_match;
    if (found_match) {
        result.filename = foundFilename;
    }
    return result;
}
