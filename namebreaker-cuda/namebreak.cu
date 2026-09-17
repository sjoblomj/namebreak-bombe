#include <cuda_runtime.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include "cpu-utils.h"
#include "constants.h"
#include "search.h"
#include "config.h"
#include "hash_kernels.cuh"
#include "platform.h"
#ifdef NAMEBREAK_WITH_NETWORK
#include "coordinator_runner.h"
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// Keep this set in sync with the switch in runCudaBatch (and MAX_ALPHABET_SIZE in
// constants.h, which must be >= the largest size here). Checked early in main(),
// before any CUDA setup, so an unsupported size fails fast with a clear message
// instead of only surfacing deep inside the first batch.
bool isSupportedAlphabetSize(int size) {
    return size == 42 || size == 43 || size == 47 || size == 48 || size == 49 || size == 50;
}

// Terminology:
// * Candidate = The part of the name that we are brute-forcing
// * Filename  = The Prefix + Candidate + Suffix

__device__ volatile int d_foundMatchFlag = 0;
// Written by the kernel alongside d_foundMatchFlag, so the host can recover
// *which* candidate matched both hashes without scanning stdout for it - see
// buildCompleteFilename (hash_kernels.cuh) for how it's populated.
__device__ char d_foundFilename[MAX_FILENAME_LEN];

// No maxBackslashCount check, and no pruneSymbolRuns check, here - both are
// applied only to the leading characters, on the CPU, before this kernel is
// ever launched (see the leadingIdx loop in runSearch), not to the trailing
// characters this kernel brute-forces. See README.md's "Design decisions"
// section for why.
template<int AlphabetSize>
__global__ void bruteForceKernel(
    int candidateLen,
    uint64_t startIdx,
    uint64_t total,
    uint32_t targetA,
    uint32_t targetB,
    char* d_matches,
    int* d_matchCount
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    idx += startIdx;

    char candidate[MAX_CANDIDATE_LEN];
    indexToCandidate<AlphabetSize>(idx, candidateLen, candidate);

    uint32_t hashA = mpqHashCandidateAndSuffix(candidate, candidateLen);
    if (hashA == targetA) {
        char filename[MAX_FILENAME_LEN];
        buildCompleteFilename(candidate, candidateLen, filename);

        // Sanity check: hashA above was computed via the prefix-cache/incremental
        // path (mpqHashCandidateAndSuffix), which must agree with hashing the
        // complete filename from scratch. A mismatch would mean the cache is out
        // of sync with the actual filename - a real bug, not a candidate to skip.
        uint32_t verifyHashA = mpqHashSeed1(filename);
        if (verifyHashA != hashA) {
            printf("WARNING: hashA mismatch for '%s' - incremental hash 0x%08X, full-filename hash 0x%08X\n",
                   filename, hashA, verifyHashA);
        }

        printf("Hash A matches: %s\n", filename);

        uint32_t hashB = mpqHashSeed2(filename);
        if (hashB == targetB) {
            printf("BOTH HASHES MATCH: %s\n", filename);
            memcpy(d_foundFilename, filename, MAX_FILENAME_LEN);
            d_foundMatchFlag = 1;
        }

        int slot = atomicAdd(d_matchCount, 1);
        if (slot < MAX_MATCHES) {
            memcpy(&d_matches[slot * MAX_FILENAME_LEN], filename, MAX_FILENAME_LEN);
        }
    }
}


// Returns 0 (no match yet), 1 (found - both hashes matched, outFoundFilename
// is filled), or -1 (abortRequested was set, this batch was skipped).
int runCudaBatch(int candidateLen, uint64_t startIdx, uint64_t count, uint32_t targetA, uint32_t targetB, FILE* fout,
                  char* d_matches, int* d_matchCount, int alphabetSize,
                  const std::atomic<bool>* abortRequested, const std::function<void(const std::string&)>& onPartialMatch,
                  char* outFoundFilename, const std::atomic<bool>* pauseRequested) {
    if (abortRequested && abortRequested->load(std::memory_order_relaxed))
        return -1;

    // Called between batches only (every previous call already synchronized
    // via cudaDeviceSynchronize below before returning), so a batch already
    // in flight is never interrupted - pausing here just means the *next*
    // kernel launch waits. Keeps polling abortRequested too, so a pause that
    // outlasts the target being solved elsewhere doesn't block forever.
    while (pauseRequested && pauseRequested->load(std::memory_order_relaxed)) {
        if (abortRequested && abortRequested->load(std::memory_order_relaxed))
            return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    int h_flag = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_flag, d_foundMatchFlag, sizeof(int)));

    if (h_flag) {
        CUDA_CHECK(cudaMemcpyFromSymbol(outFoundFilename, d_foundFilename, MAX_FILENAME_LEN));
        return h_flag;
    }
    CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(int)));

    // This launches `blocks * threadsPerBlock` GPU threads for the chunk (up to
    // ~5.76M for a full batch - see batchSize in runSearch), one candidate per
    // thread. Every 32 consecutive threads form a "warp" that the hardware runs
    // in lockstep (SIMT) - that grouping is automatic (256 threads/block = 8
    // warps/block here), not something chosen at this call site.
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    // alphabetSize has to be dispatched to one of a fixed set of compile-time
    // template instantiations (see indexToCandidate's comment for why) - this is
    // the whole set this build supports. Add a case (and recompile/redistribute
    // namebreak to volunteers) to support a new size.
    #define LAUNCH_WITH_ALPHABET_SIZE(SIZE) \
        bruteForceKernel<SIZE><<<blocks, threadsPerBlock>>>( \
                candidateLen, startIdx, count, targetA, targetB, d_matches, d_matchCount)
    switch (alphabetSize) {
        case 42: LAUNCH_WITH_ALPHABET_SIZE(42); break;
        case 43: LAUNCH_WITH_ALPHABET_SIZE(43); break;
        case 47: LAUNCH_WITH_ALPHABET_SIZE(47); break;
        case 48: LAUNCH_WITH_ALPHABET_SIZE(48); break;
        case 49: LAUNCH_WITH_ALPHABET_SIZE(49); break;
        case 50: LAUNCH_WITH_ALPHABET_SIZE(50); break;
        default:
            fprintf(stderr, "Unsupported alphabet size: %d (this build only supports: 42, 43, 47, 48, 49, 50)\n", alphabetSize);
            exit(1);
    }
    #undef LAUNCH_WITH_ALPHABET_SIZE
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_matchCount = 0;
    CUDA_CHECK(cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost));
    h_matchCount = std::min(h_matchCount, MAX_MATCHES);

    char h_matches[MAX_MATCHES][MAX_FILENAME_LEN];
    CUDA_CHECK(cudaMemcpy(h_matches, d_matches, h_matchCount * MAX_FILENAME_LEN, cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_matchCount; ++i) {
        fprintf(fout, "%s\n", h_matches[i]);
        fflush(fout);
        if (onPartialMatch)
            onPartialMatch(h_matches[i]);
    }

    // Re-read rather than reusing the pre-launch value above: this batch's
    // own kernel (just synchronized) may be the one that just found the
    // match, and the early-return above only ever fires for a match found by
    // some *earlier* call - without this, a match found in a batch would go
    // undetected until whatever call happens to come after it, which may
    // never come (e.g. a "bounded" search whose very last batch is the one
    // that finds it would otherwise report "not found").
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_flag, d_foundMatchFlag, sizeof(int)));
    if (h_flag) {
        CUDA_CHECK(cudaMemcpyFromSymbol(outFoundFilename, d_foundFilename, MAX_FILENAME_LEN));
    }
    return h_flag;
}

SearchResult runSearch(const SearchRequest& req, std::atomic<bool>* abortRequested, std::function<void(const std::string&)> onPartialMatch,
                        const std::atomic<bool>* pauseRequested) {
    SearchResult result;

    if (req.alphabet.empty() || req.alphabet.size() > MAX_ALPHABET_SIZE) {
        result.ok = false;
        result.error = "Alphabet must be non-empty and at most " + std::to_string(MAX_ALPHABET_SIZE) + " characters (got " + std::to_string(req.alphabet.size()) + ")";
        return result;
    }
    int alphabetSize = (int) req.alphabet.size();
    if (!isSupportedAlphabetSize(alphabetSize)) {
        result.ok = false;
        result.error = "Unsupported alphabet size: " + std::to_string(alphabetSize) + " (this build only supports: 42, 43, 47, 48, 49, 50)";
        return result;
    }
    // 0 means unlimited (see hasForbiddenSymbolRun_CPU/countBackslashes_CPU's
    // declaration comment in cpu-utils.h for how and where this is applied).
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
    // length). This only bounds leadingLen below (see gpuWindowChars and
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

    // How many trailing candidate characters go straight to the GPU as a
    // native index each batch; the rest is folded into an extended prefix and
    // hashed on the CPU instead, incrementally (IncrementalPrefixHasher,
    // cpu-utils.h) rather than from scratch per leading value. Deliberately
    // small and fixed - NOT "as large as maxSafeIndexLen allows", which is
    // what this project used to do (and still needs to fall back toward for
    // very long candidates - see trailingLen's computation below). See
    // README.md's "Design decisions" section for why. Overridable at compile
    // time (-DNAMEBREAK_GPU_WINDOW_CHARS=N) purely for re-sweeping this
    // number against real hardware/alphabet combinations later
    // (tests/search_bench.cu) without hand-editing the source each time.
#ifndef NAMEBREAK_GPU_WINDOW_CHARS
#define NAMEBREAK_GPU_WINDOW_CHARS 4
#endif
    constexpr int gpuWindowChars = NAMEBREAK_GPU_WINDOW_CHARS;

    int maxLeadingLen = maxSafeIndexLen;
    printf("gpuWindowChars: %d, maxSafeIndexLen: %d (max leading/prefix-extension length: %d)\n",
           gpuWindowChars, maxSafeIndexLen, maxLeadingLen);

    if (prefix_size + maxLeadingLen >= (int) sizeof(d_prefix) || suffix_size >= (int) sizeof(d_suffix)) {
        result.ok = false;
        result.error = "prefix (up to " + std::to_string(prefix_size + maxLeadingLen) + " once extended by leading candidate characters) or suffix (" +
                        std::to_string(suffix_size) + ") too long for device buffers (max: " + std::to_string(sizeof(d_prefix)) + " each)";
        return result;
    }
    if (prefix_size + suffix_size + MAX_CANDIDATE_LEN >= MAX_FILENAME_LEN) {
        result.ok = false;
        result.error = "prefix (" + std::to_string(prefix_size) + ") + suffix (" + std::to_string(suffix_size) + ") + candidate (up to " +
                        std::to_string(MAX_CANDIDATE_LEN) + ") would exceed MAX_FILENAME_LEN (" + std::to_string(MAX_FILENAME_LEN) + ")";
        return result;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix_size, &suffix_size, sizeof(suffix_size)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix, req.suffix.c_str(), suffix_size + 1));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alphabet, req.alphabet.c_str(), req.alphabet.size() + 1));

    // A long-lived process (coordinator mode) can call runSearch() many times
    // over its lifetime, one per claimed range - reset explicitly rather than
    // relying on its process-startup zero value, so a previous range's match
    // can't make every subsequent range falsely report "found" immediately.
    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_foundMatchFlag, &zero, sizeof(zero)));

    std::string lowerBoundLimit, upperBoundLimit;
    if (!getLowerBound(req.lowerBound, req.alphabet, lowerBoundLimit, result.error) ||
        !getUpperBound(req.upperBound, req.alphabet, upperBoundLimit, result.error)) {
        result.ok = false;
        return result;
    }

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
    printf("maxBackslashCount: %d%s (leading characters only)\n", req.maxBackslashCount,
           req.maxBackslashCount == 0 ? " (unlimited)" : "");

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    CUDA_CHECK(cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable)));

    // Hash of req.prefix alone (never changes across candidateLen or
    // leadingIdx) - the base every leadingIdx loop's IncrementalPrefixHasher
    // extends by that iteration's leading characters.
    std::pair<uint32_t, uint32_t> prefixBaseState = mpqHashWithPrefixCache_CPU(req.prefix.c_str(), h_cryptTable);

    FILE* fout = fopen("matches.txt", "a");
    if (!fout) {
        result.ok = false;
        result.error = std::string("fopen matches.txt: ") + strerror(errno);
        return result;
    }

    // Allocated once and reused for every batch, instead of malloc/free per call -
    // these buffers are always the same size, so there's no reason to pay driver
    // allocation overhead on every single kernel launch.
    char* d_matches;
    int* d_matchCount;
    CUDA_CHECK(cudaMalloc(&d_matches, MAX_MATCHES * MAX_FILENAME_LEN));
    CUDA_CHECK(cudaMalloc(&d_matchCount, sizeof(int)));

    bool found_match = false;
    bool aborted = false;
    char foundFilename[MAX_FILENAME_LEN] = {0};
    std::string start_candidate = req.startCandidate;
    int candidateLen = start_candidate.size();
    // A fixed tuning constant (candidates per kernel launch) rather than derived
    // from alphabetSize (as it implicitly was when this was ALPHABET_SIZE^4) - a
    // small alphabet would otherwise produce pathologically tiny, overhead-heavy
    // batches. 49^4, matching this project's original default alphabet size.
    const uint64_t batchSize = 5'764'801;

    // The search space is walked by four nested levels, outermost to innermost:
    //  1. This `while` loop: over candidateLen itself - "try every 1-character
    //     candidate, then every 2-character one, ..." Only continuous mode
    //     (req.continuous) actually loops here more than once; bounded mode
    //     runs the body for req.startCandidate's own length and exits via the
    //     `if (!req.continuous)` check at the bottom.
    //  2. The `for (leadingIdx ...)` loop below: over the *leading* part of the
    //     candidate (see gpuWindowChars/leadingLen above) - every leading value
    //     is hashed on the CPU (incrementally - IncrementalPrefixHasher) and
    //     uploaded as the GPU's starting seed, so candidates longer than
    //     gpuWindowChars still get covered exhaustively. Since gpuWindowChars
    //     is small (throughput-tuned, not "as large as safely possible"),
    //     leadingLen > 0 - and this loop actually doing work - is the common
    //     case, not the exception; it only degenerates to a single iteration
    //     when candidateLen <= gpuWindowChars.
    //  3. The `for (i = trailStart ...)` loop: chops the (up to
    //     alphabetSize^trailingLen) remaining space for one leading value into
    //     batchSize-sized chunks, since that's too large for one kernel launch -
    //     each iteration is one runCudaBatch call, i.e. one kernel launch.
    //  4. Inside runCudaBatch: one GPU thread per candidate in the chunk (up to
    //     ~5.76M threads for a full batchSize chunk) - see its own comment.
    while (true) {
        if (candidateLen > MAX_CANDIDATE_LEN) {
            fprintf(stderr, "candidateLen (%d) exceeds MAX_CANDIDATE_LEN (%d) - exiting\n", candidateLen, MAX_CANDIDATE_LEN);
            break;
        }

        // Split the candidate into a leading part (folded into the prefix, hashed
        // incrementally on the CPU) and a trailing part of gpuWindowChars characters
        // (brute-forced by the GPU with native 64-bit indices) - or, once candidateLen
        // grows large enough that leadingLen would exceed maxSafeIndexLen (overflow the
        // index conversions below), a slightly larger trailing part, just big enough to
        // keep leadingLen within that safe limit. Every combination of the leading part
        // is enumerated too, so the full candidateLen-character space is still covered
        // exhaustively either way.
        int trailingLen = std::min(candidateLen, std::max(gpuWindowChars, candidateLen - maxSafeIndexLen));
        int leadingLen = candidateLen - trailingLen;
        if (leadingLen > maxSafeIndexLen) {
            // Not expected to ever trigger given trailingLen's formula above (it's
            // constructed specifically to keep leadingLen <= maxSafeIndexLen) - kept as
            // a belt-and-suspenders check against future edits to that formula, since a
            // silent overflow here would mean silently skipped candidates, not a crash.
            fprintf(stderr, "candidateLen (%d) needs a %d-character leading part, which exceeds maxSafeIndexLen (%d) - exiting\n",
                    candidateLen, leadingLen, maxSafeIndexLen);
            break;
        }

        std::string start_full = make_bound_string(start_candidate, candidateLen);
        std::string end_full   = make_bound_string(upperBoundLimit, candidateLen);

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
        // cpu-utils.h) instead of a full from-scratch re-hash of the whole
        // leading string - the thing that makes leadingLen > 0 being the
        // common case (see gpuWindowChars above) affordable.
        IncrementalPrefixHasher leadingHasher(prefixBaseState, leadingLen, req.alphabet, h_cryptTable);
        leadingHasher.reset(startLeadingIdx);
        for (uint64_t leadingIdx = startLeadingIdx; leadingIdx <= endLeadingIdx; ++leadingIdx) {
            if (leadingIdx != startLeadingIdx) {
                leadingHasher.advance();
            }
            const std::string& leading = leadingHasher.leading();

            // req.pruneSymbolRuns/req.maxBackslashCount examine `leading` -
            // the CPU-computed first leadingLen characters of the candidate
            // (see README.md's "Design decisions" section for why checking it
            // here, instead of on the GPU, is worth doing). A prune here
            // skips this leading value's entire trailing batch (up to
            // batchSize candidates) without spending anything on the GPU,
            // uploads included - cheaper than even one of those candidates
            // would have cost individually.
            if (req.pruneSymbolRuns && hasForbiddenSymbolRun_CPU(leading))
                continue;
            if (req.maxBackslashCount != 0 && countBackslashes_CPU(leading) > req.maxBackslashCount)
                continue;

            std::string extendedPrefix = req.prefix + leading;

            short extPrefixSize = extendedPrefix.size();
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix_size, &extPrefixSize, sizeof(extPrefixSize)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix, extendedPrefix.c_str(), extPrefixSize + 1));

            std::pair<uint32_t, uint32_t> pair = leadingHasher.state();
            uint32_t seed1_start = pair.first;
            uint32_t seed2_start = pair.second;
            CUDA_CHECK(cudaMemcpyToSymbol(d_seed1_start, &seed1_start, sizeof(seed1_start)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_seed2_start, &seed2_start, sizeof(seed2_start)));

            uint64_t trailStart = 0, trailEnd = 0;
            if (leadingIdx == startLeadingIdx) {
                if (!stringToIndex(start_full.substr(leadingLen), req.alphabet, trailStart, result.error)) {
                    result.ok = false;
                    goto breakfree;
                }
            }
            if (leadingIdx == endLeadingIdx) {
                if (!stringToIndex(end_full.substr(leadingLen), req.alphabet, trailEnd, result.error)) {
                    result.ok = false;
                    goto breakfree;
                }
            } else {
                trailEnd = trailSpaceSize;
            }

            // Level 3 (see the walkthrough above the outer `while`).
            for (uint64_t i = trailStart; i < trailEnd; i += batchSize) {
                uint64_t count = std::min(batchSize, trailEnd - i);
                int r = runCudaBatch(trailingLen, i, count, req.targetHashA, req.targetHashB, fout, d_matches, d_matchCount,
                                      alphabetSize, abortRequested, onPartialMatch, foundFilename, pauseRequested);
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

    CUDA_CHECK(cudaFree(d_matches));
    CUDA_CHECK(cudaFree(d_matchCount));
    fclose(fout);

    result.aborted = aborted;
    result.found = found_match;
    if (found_match) {
        result.filename = std::string(foundFilename);
    }
    return result;
}

// Guarded so tests/search_integration_test.cu can link against this file's
// runSearch() (the actual, unmodified function - not a reimplementation of
// it) without a duplicate main() symbol: built with -DNAMEBREAK_NO_MAIN. That
// build doesn't link platform.cpp either, so g_paused/pauseKeyListener below
// (which need it) have to stay inside this guard too.
#ifndef NAMEBREAK_NO_MAIN

// Toggled by pauseKeyListener below, polled by runCudaBatch (via runSearch's
// pauseRequested parameter) so a pause takes effect between batches rather
// than needing to interrupt one mid-flight.
std::atomic<bool> g_paused{false};

// Runs for the life of the process once main() starts it (only when stdin is
// an interactive terminal - see its call site), toggling g_paused on 'p'/
// 'P', the same key cgminer/xmrig and other long-running GPU compute tools
// already use for this. One key toggles both directions (like a media
// player's pause button) rather than separate pause/resume keys, since
// there's only ever one thing to remember.
void pauseKeyListener() {
    for (;;) {
        int key = readKeypressBlocking();
        if (key < 0)
            return; // stdin closed - nothing left to listen for
        if (key != 'p' && key != 'P')
            continue;
        bool nowPaused = !g_paused.load(std::memory_order_relaxed);
        g_paused.store(nowPaused, std::memory_order_relaxed);
        printf(nowPaused ? "\n[paused] finishing the current batch; no new batches will start until resumed (press 'p' to resume)\n"
                          : "\n[resumed]\n");
        fflush(stdout);
    }
}

// Installed as SIGINT's handler (overriding the plain restore-and-terminate
// one enableRawKeypressMode() already installed - see platform.h) so a
// reflexive first Ctrl+C pauses instead of losing the run outright: a pause
// is trivially undone (press 'p', or Ctrl+C once more), a quit isn't. A
// second Ctrl+C while already paused - however it got paused, this handler
// or the 'p' key - actually quits, restoring the terminal first exactly
// like the handler it replaced would have.
//
// Signal-handler context: only touches an atomic and async-signal-safe
// calls (write() via writeStdoutSignalSafe, tcsetattr via
// restoreKeypressMode, signal(), raise()) - no printf/fflush, which aren't
// guaranteed reentrant-safe if this signal interrupts another stdio call
// already in progress on the main or listener thread.
void handleSigintPauseOrQuit(int sig) {
    if (!g_paused.exchange(true, std::memory_order_relaxed)) {
        static constexpr char kMsg[] =
            "\n[paused] (Ctrl+C) finishing the current batch; no new batches will start until resumed "
            "(press 'p' to resume, or Ctrl+C again to quit)\n";
        writeStdoutSignalSafe(kMsg, sizeof(kMsg) - 1);
        return;
    }
    static constexpr char kMsg[] = "\n[quitting] (Ctrl+C again)\n";
    writeStdoutSignalSafe(kMsg, sizeof(kMsg) - 1);
    restoreKeypressMode();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

int main(int argc, char* argv[]) {
    // The only argument namebreak takes: an optional mode, overriding
    // config.conf's own `mode = ...` (see config.h). Everything else
    // lives in config.conf.
    std::string modeOverride = (argc >= 2) ? argv[1] : "";
    if (argc > 2 || (argc == 2 && modeOverride != "continuous" && modeOverride != "bounded" && modeOverride != "coordinator")) {
        fprintf(stderr, "Usage: %s [continuous|bounded|coordinator]\n"
                         "Reads %s from the current directory for everything else; the argument above,\n"
                         "if given, overrides that file's own 'mode = ...'.\n",
                argv[0], kConfigPath);
        return 1;
    }

    ConfigFile config;
    std::string error;
    if (!loadConfigFile(kConfigPath, config, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    std::string mode = !modeOverride.empty() ? modeOverride : config.mode;
    if (mode != "continuous" && mode != "bounded" && mode != "coordinator") {
        fprintf(stderr, "Unknown or missing mode '%s' - expected continuous, bounded, or coordinator "
                         "(set %s's 'mode = ...', or pass one as this program's argument)\n",
                mode.c_str(), kConfigPath);
        return 1;
    }

    // Only when stdin is actually a terminal - a piped/redirected/absent
    // stdin (cron, systemd, ...) has no keypresses to listen for, and
    // enableRawKeypressMode() would just fail anyway.
    if (isInteractiveTerminal() && enableRawKeypressMode()) {
        printf("Press 'p' to pause/resume the search. Ctrl+C pauses too - press it again to quit.\n");
        std::thread(pauseKeyListener).detach();
        std::signal(SIGINT, handleSigintPauseOrQuit);
    }

#ifdef NAMEBREAK_WITH_NETWORK
    if (mode == "coordinator") {
        CoordinatorArgs cargs;
        if (!buildCoordinatorArgs(config.coordinator, cargs, error)) {
            fprintf(stderr, "%s [coordinator]: %s\n", kConfigPath, error.c_str());
            return 1;
        }
        return runCoordinator(cargs, &g_paused);
    }
#else
    if (mode == "coordinator") {
        fprintf(stderr, "This build of %s was compiled without networking support (rebuild without NETWORK=0 to enable 'coordinator' mode).\n", argv[0]);
        return 1;
    }
#endif

    SearchRequest req;
    if (!buildSearchRequest(config.search, mode == "continuous", req, error)) {
        fprintf(stderr, "%s [search]: %s\n", kConfigPath, error.c_str());
        return 1;
    }

    SearchResult result = runSearch(req, nullptr, nullptr, &g_paused);

    if (!result.ok) {
        fprintf(stderr, "%s\n", result.error.c_str());
        return 1;
    }
    return result.found ? 0 : 2;
}
#endif // NAMEBREAK_NO_MAIN
