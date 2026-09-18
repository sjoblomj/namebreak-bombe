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
#include <type_traits>
#include <vector>
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
// * Trailing part = the last `trailingLen` characters of the candidate; the
//   part the GPU enumerates (the rest - the "leading" part - is folded into
//   the prefix on the CPU, see runSearch).
// * Trailing index = a candidate's position within the trailing space, in
//   the same last-character-fastest order as indexToString/stringToIndex.
// * Row = one combination of the trailing part's first `trailingLen - 1`
//   characters. A row holds `alphabetSize` candidates, one per value of the
//   last character, so row r's candidate with last character k has trailing
//   index r * alphabetSize + k. One GPU thread handles one whole row.

// Written by verifyMatchesKernel alongside BatchResults::foundFlag, so the
// host can recover *which* candidate matched both hashes without scanning
// stdout for it - see buildCompleteFilename (hash_kernels.cuh).
__device__ char d_foundFilename[MAX_FILENAME_LEN];

// The kernels' small per-batch outputs, kept together in one device buffer
// so the host reads them back with a single cudaMemcpy (each separate driver
// call is ~10us of GPU idle time between two consecutive kernels).
struct BatchResults {
    int matchCount; // hashA hits this batch (may exceed MAX_MATCHES; runCudaBatch then re-searches the range in halves)
    int foundFlag;  // 1 once any candidate has matched both hashes - never reset mid-search
};

struct DeviceBuffers {
    BatchResults* results;
    // matchIdx[i]: trailing index of the i-th hashA hit (only the first MAX_MATCHES
    // hits of a launch are recorded). Written by bruteForceKernel, consumed by
    // verifyMatchesKernel.
    uint64_t* matchIdx;
    // matches + i * MAX_FILENAME_LEN: the i-th hit's complete filename.
    // Written by verifyMatchesKernel, consumed by the host.
    char* matches;
};

constexpr int kThreadsPerBlock = 256; // must be >= MAX_ALPHABET_SIZE (bruteForceKernel fills its shared tables one entry per thread)
static_assert(kThreadsPerBlock >= MAX_ALPHABET_SIZE, "bruteForceKernel needs one thread per alphabet entry to fill its shared tables");
static_assert(NAMEBREAK_ROWS_PER_LAUNCH <= (1u << 30), "a launch's row count must stay well within 32 bits");

// Suffix lengths 0-8 (see dispatchSuffixLen) get their own compile-time
// instantiation of bruteForceKernel, fully unrolled - measured ~2x faster than
// looping over a runtime length; anything longer falls back to kRuntimeSuffix.
constexpr int kRuntimeSuffix = -1;

// One step of the MPQ hash recurrence (hashA table, offset 0x100) for a
// character whose crypt-table key `key` and value `ord` are already known.
// Must match mpqHashCandidateAndSuffix/cpu-utils.cpp exactly, or a match
// found on one side would never reproduce on the other.
__device__ __forceinline__ void mpqStep(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = ord + seed1 + seed2 + (seed2 << 5) + 3;
}

// Hashes the first N characters of `row` (its digits in base AlphabetSize,
// most significant first) into (seed1, seed2), via the block's shared tables.
// Consecutive lanes have consecutive rows, so consecutive lanes read
// consecutive shared-memory banks here (no bank conflicts) - unlike the
// __constant__ d_cryptTable, where every lane needing a different entry is
// serialized.
template<int AlphabetSize, int N>
__device__ __forceinline__ void hashRowDigits(uint32_t row, uint32_t& seed1, uint32_t& seed2,
                                                const uint32_t* sKey, const uint32_t* sOrd) {
    if constexpr (N > 0) {
        unsigned digit[N];
        #pragma unroll
        for (int i = N - 1; i >= 0; --i) {
            uint32_t q = row / AlphabetSize;
            digit[i] = row - q * AlphabetSize;
            row = q;
        }
        #pragma unroll
        for (int i = 0; i < N; ++i)
            mpqStep(seed1, seed2, sKey[digit[i]], sOrd[digit[i]]);
    }
}

// No maxBackslashCount check, and no pruneSymbolRuns check, here - both are
// applied only to the leading characters, on the CPU, before this kernel is
// ever launched (see the leadingIdx loop in runSearch), not to the trailing
// characters this kernel brute-forces. See README.md's "Design decisions"
// section for why.
//
// Thread t handles row `firstRow + t` (t < rowCount): every candidate of
// that row whose last character index k is in [kBegin, kEnd) - all of them
// for every row but the (at most two) at the edges of the launch's range
// (firstRowStartK / lastRowEndK). The row's shared prefix (its first
// trailingLen - 1 characters) is hashed once, then the last character is
// looped over the alphabet, so each candidate only costs one character step
// plus the suffix - instead of every candidate re-decoding and re-hashing
// all trailingLen characters itself. In that loop every table read is at a
// compile-time-constant index (the loop is fully unrolled), so the values are
// constant-bank operands of the ALU instructions themselves.
//
// A hashA hit only records its trailing index (bufs.matchIdx): building the
// filename and checking hashB happen in verifyMatchesKernel, launched only
// when there was a hit, so none of that (printf, a 128-byte filename buffer)
// bloats this hot kernel.
//
// `params` (prefix, seeds) is only ever read at constant offsets here - see
// buildCompleteFilename (hash_kernels.cuh) for why its address must not be taken.
template<int AlphabetSize, int SuffixLen>
__global__ void bruteForceKernel(
    int trailingLen,
    uint32_t firstRow,
    uint32_t rowCount,
    int firstRowStartK,
    int lastRowEndK,
    uint32_t targetA,
    BatchParams params,
    DeviceBuffers bufs
) {
    __shared__ uint32_t sKey[AlphabetSize];
    __shared__ uint32_t sOrd[AlphabetSize];
    if (threadIdx.x < AlphabetSize) {
        sKey[threadIdx.x] = d_alphabetKey[threadIdx.x];
        sOrd[threadIdx.x] = d_alphabetOrd[threadIdx.x];
    }
    __syncthreads(); // before the bounds check below, so every thread of the block reaches it

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rowCount)
        return;
    const uint32_t row = firstRow + t;
    const int kBegin = (t == 0) ? firstRowStartK : 0;
    const int kEnd = (t == rowCount - 1) ? lastRowEndK : AlphabetSize;

    uint32_t seed1 = params.seed1Start;
    uint32_t seed2 = params.seed2Start;
    switch (trailingLen - 1) {
        case 0: hashRowDigits<AlphabetSize, 0>(row, seed1, seed2, sKey, sOrd); break;
        case 1: hashRowDigits<AlphabetSize, 1>(row, seed1, seed2, sKey, sOrd); break;
        case 2: hashRowDigits<AlphabetSize, 2>(row, seed1, seed2, sKey, sOrd); break;
        case 3: hashRowDigits<AlphabetSize, 3>(row, seed1, seed2, sKey, sOrd); break;
        case 4: hashRowDigits<AlphabetSize, 4>(row, seed1, seed2, sKey, sOrd); break;
        case 5: hashRowDigits<AlphabetSize, 5>(row, seed1, seed2, sKey, sOrd); break;
        // trailingLen is validated against MAX_TRAILING_LEN (== 6) by runSearch
    }

    constexpr int kSuffixRegs = (SuffixLen > 0) ? SuffixLen : 1;
    uint32_t sufKey[kSuffixRegs];
    uint32_t sufOrd[kSuffixRegs];
    if constexpr (SuffixLen > 0) {
        #pragma unroll
        for (int i = 0; i < SuffixLen; ++i) {
            sufKey[i] = d_suffixKey[i];
            sufOrd[i] = (unsigned char) d_suffix[i];
        }
    }

    // hashA of (this row's prefix state) + one last character + the suffix.
    auto hashCandidate = [&](uint32_t key, uint32_t ord) -> uint32_t {
        uint32_t a = seed1, b = seed2;
        mpqStep(a, b, key, ord);
        if constexpr (SuffixLen == kRuntimeSuffix) {
            const int n = d_suffix_size;
            for (int i = 0; i < n; ++i)
                mpqStep(a, b, d_suffixKey[i], (unsigned char) d_suffix[i]);
        } else {
            #pragma unroll
            for (int i = 0; i < SuffixLen; ++i)
                mpqStep(a, b, sufKey[i], sufOrd[i]);
        }
        return a;
    };
    auto record = [&](int k) {
        int slot = atomicAdd(&bufs.results->matchCount, 1);
        if (slot < MAX_MATCHES)
            bufs.matchIdx[slot] = (uint64_t) row * AlphabetSize + k;
    };

    if (kBegin == 0 && kEnd == AlphabetSize) {
        #pragma unroll
        for (int k = 0; k < AlphabetSize; ++k) {
            if (hashCandidate(d_alphabetKey[k], d_alphabetOrd[k]) == targetA)
                record(k);
        }
    } else {
        // A partial row at the edge of the launch's range - only ever the
        // first and/or last thread, so its cost doesn't matter, only that
        // it's exactly right.
        #pragma unroll 1
        for (int k = kBegin; k < kEnd; ++k) {
            if (hashCandidate(sKey[k], sOrd[k]) == targetA)
                record(k);
        }
    }
}

// Second stage, launched only for a batch that had hashA hits (one thread per
// recorded hit): rebuilds each hit's complete filename and does everything
// that used to happen inline in the search kernel - the hashB check, the
// found flag/filename, the printf output. Deliberately implemented with the
// *original*, independent hashing path (indexToCandidate +
// mpqHashCandidateAndSuffix + from-scratch mpqHashSeed1/Seed2), not
// bruteForceKernel's row-based one, so every hit bruteForceKernel reports is
// cross-checked against a second implementation at runtime.
template<int AlphabetSize>
__global__ void verifyMatchesKernel(
    int trailingLen,
    int matchCount,
    uint32_t targetA,
    uint32_t targetB,
    BatchParams params,
    DeviceBuffers bufs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= matchCount)
        return;

    char candidate[MAX_CANDIDATE_LEN];
    indexToCandidate<AlphabetSize>(bufs.matchIdx[i], trailingLen, candidate);
    char filename[MAX_FILENAME_LEN];
    buildCompleteFilename(params, candidate, trailingLen, filename);

    // hashA via the prefix-cache/incremental path must agree with both the
    // target bruteForceKernel claimed to hit and hashing the complete filename
    // from scratch. A mismatch would mean one of them is out of sync with the
    // actual filename - a real bug, not a candidate to skip.
    uint32_t hashA = mpqHashCandidateAndSuffix(candidate, trailingLen, params.seed1Start, params.seed2Start);
    if (hashA != targetA) {
        printf("WARNING: bruteForceKernel reported a hashA hit for '%s' but the reference hashing path gives 0x%08X, not the target 0x%08X\n",
               filename, hashA, targetA);
    }
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
        bufs.results->foundFlag = 1;
    }

    memcpy(&bufs.matches[(size_t) i * MAX_FILENAME_LEN], filename, MAX_FILENAME_LEN);
}

// alphabetSize/suffix length have to be dispatched to one of a fixed set of
// compile-time template instantiations (see indexToCandidate's and
// bruteForceKernel's comments for why) - these two functions are the whole
// set this build supports, and call `f` with a std::integral_constant of the
// matching value. Add a case (and recompile/redistribute namebreak to
// volunteers) to support a new alphabet size. Returns false if unsupported.
template<typename F>
bool dispatchAlphabetSize(int alphabetSize, F&& f) {
    switch (alphabetSize) {
        case 42: f(std::integral_constant<int, 42>{}); return true;
        case 43: f(std::integral_constant<int, 43>{}); return true;
        case 47: f(std::integral_constant<int, 47>{}); return true;
        case 48: f(std::integral_constant<int, 48>{}); return true;
        case 49: f(std::integral_constant<int, 49>{}); return true;
        case 50: f(std::integral_constant<int, 50>{}); return true;
        default: return false;
    }
}

template<typename F>
void dispatchSuffixLen(int suffixLen, F&& f) {
    switch (suffixLen) {
        case 0: f(std::integral_constant<int, 0>{}); break;
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: f(std::integral_constant<int, kRuntimeSuffix>{}); break;
    }
}

// Searches the trailing indices [startIdx, startIdx + count) (count > 0).
// Returns 0 (no match yet), 1 (found - both hashes matched, outFoundFilename
// is filled), or -1 (abortRequested was set, this batch was skipped).
//
// Every hashA hit is verified against hashB: if a launch has more hits than
// MAX_MATCHES record (which needs a target hashA with over a thousand matches
// among the range's candidates - not something a real 32-bit hash produces,
// but the one outcome that must never happen is a both-hashes match going
// unchecked), the range is searched again as two halves, recursively.
int runCudaBatch(int trailingLen, uint64_t startIdx, uint64_t count, uint32_t targetA, uint32_t targetB, const BatchParams& params, FILE* fout,
                  const DeviceBuffers& bufs, int alphabetSize, int suffixLen,
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

    // No pre-launch check of bufs.results->foundFlag: runSearch zeroes it
    // before the first batch, and a batch that finds a match returns 1 (below)
    // - so runSearch stops calling this - rather than leaving it set for a
    // later call to notice. bufs.results->matchCount is likewise already 0
    // here (see the reset after the readback below), so there's no per-batch
    // memset either. Each of those was a synchronous driver call in the gap
    // between two kernels.

    // The range [startIdx, endIdx) covers rows firstRow..lastRow; only the
    // first and last row can be partial (see bruteForceKernel).
    const uint64_t endIdx = startIdx + count;
    const uint64_t firstRow = startIdx / alphabetSize;
    const uint64_t lastRow = (endIdx - 1) / alphabetSize;
    const uint64_t rowCount = lastRow - firstRow + 1;
    if (lastRow > UINT32_MAX || rowCount > (1ull << 31)) {
        fprintf(stderr, "Batch too large for the kernel's 32-bit row index (rows %llu..%llu) - exiting\n",
                (unsigned long long) firstRow, (unsigned long long) lastRow);
        exit(1);
    }
    const int firstRowStartK = (int) (startIdx - firstRow * alphabetSize);
    const int lastRowEndK = (int) (endIdx - lastRow * alphabetSize); // in [1, alphabetSize]

    // This launches one GPU thread per row (up to NAMEBREAK_ROWS_PER_LAUNCH+1
    // of them). Every 32 consecutive threads form a "warp" that the hardware
    // runs in lockstep (SIMT) - that grouping is automatic (256 threads/block
    // = 8 warps/block here), not something chosen at this call site.
    const unsigned blocks = (unsigned) ((rowCount + kThreadsPerBlock - 1) / kThreadsPerBlock);

    bool supported = dispatchAlphabetSize(alphabetSize, [&](auto alphabetC) {
        constexpr int AlphabetSize = decltype(alphabetC)::value;
        dispatchSuffixLen(suffixLen, [&](auto suffixC) {
            constexpr int SuffixLen = decltype(suffixC)::value;
            bruteForceKernel<AlphabetSize, SuffixLen><<<blocks, kThreadsPerBlock>>>(
                    trailingLen, (uint32_t) firstRow, (uint32_t) rowCount, firstRowStartK, lastRowEndK, targetA, params, bufs);
        });
    });
    if (!supported) {
        fprintf(stderr, "Unsupported alphabet size: %d (this build only supports: 42, 43, 47, 48, 49, 50)\n", alphabetSize);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    BatchResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, bufs.results, sizeof(h_results), cudaMemcpyDeviceToHost));

    // Hits are rare (a few per thousand batches), so everything below is the
    // exception path - the common batch costs exactly the one cudaMemcpy above.
    if (h_results.matchCount > MAX_MATCHES) {
        // More hits than bufs.matchIdx can record: the excess would never reach
        // verifyMatchesKernel, so they'd never be checked against hashB. Nothing
        // from this launch has been verified or reported yet, so discard it and
        // search each half of the range on its own instead (splitting again as
        // needed). A one-candidate range has at most one hit and MAX_MATCHES >= 1,
        // so count >= 2 here and this always terminates.
        fprintf(stderr, "note: %d hashA hits in one batch, more than the %d that can be recorded - searching its two halves separately\n",
                h_results.matchCount, MAX_MATCHES);
        CUDA_CHECK(cudaMemset(&bufs.results->matchCount, 0, sizeof(int)));
        const uint64_t half = count / 2;
        int r = runCudaBatch(trailingLen, startIdx, half, targetA, targetB, params, fout, bufs, alphabetSize, suffixLen,
                              abortRequested, onPartialMatch, outFoundFilename, pauseRequested);
        if (r != 0)
            return r;
        return runCudaBatch(trailingLen, startIdx + half, count - half, targetA, targetB, params, fout, bufs, alphabetSize, suffixLen,
                             abortRequested, onPartialMatch, outFoundFilename, pauseRequested);
    }
    if (h_results.matchCount > 0) {
        const int recorded = h_results.matchCount;

        dispatchAlphabetSize(alphabetSize, [&](auto alphabetC) {
            constexpr int AlphabetSize = decltype(alphabetC)::value;
            verifyMatchesKernel<AlphabetSize><<<(recorded + 63) / 64, 64>>>(trailingLen, recorded, targetA, targetB, params, bufs);
        });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<char> h_matches((size_t) recorded * MAX_FILENAME_LEN);
        CUDA_CHECK(cudaMemcpy(h_matches.data(), bufs.matches, h_matches.size(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < recorded; ++i) {
            const char* match = &h_matches[(size_t) i * MAX_FILENAME_LEN];
            fprintf(fout, "%s\n", match);
            fflush(fout);
            if (onPartialMatch)
                onPartialMatch(match);
        }

        // verifyMatchesKernel is what sets foundFlag, so re-read it now.
        CUDA_CHECK(cudaMemcpy(&h_results, bufs.results, sizeof(h_results), cudaMemcpyDeviceToHost));

        // Restores the "matchCount is 0 at launch" invariant the next batch relies on.
        CUDA_CHECK(cudaMemset(&bufs.results->matchCount, 0, sizeof(int)));
    }

    // Read after this batch's own kernels have finished, so a match found by
    // *this* batch is reported now instead of going undetected until
    // whatever call happens to come after it, which may never come (e.g. a
    // "bounded" search whose very last batch is the one that finds it would
    // otherwise report "not found").
    if (h_results.foundFlag) {
        CUDA_CHECK(cudaMemcpyFromSymbol(outFoundFilename, d_foundFilename, MAX_FILENAME_LEN));
    }
    return h_results.foundFlag;
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
    // README.md's "Design decisions" section for why. Defined (and
    // overridable at compile time, -DNAMEBREAK_GPU_WINDOW_CHARS=N) in
    // constants.h, so tests/benchmarks share this exact value instead of
    // duplicating it.
    constexpr int gpuWindowChars = NAMEBREAK_GPU_WINDOW_CHARS;

    int maxLeadingLen = maxSafeIndexLen;
    printf("gpuWindowChars: %d, maxSafeIndexLen: %d (max leading/prefix-extension length: %d)\n",
           gpuWindowChars, maxSafeIndexLen, maxLeadingLen);

    if (prefix_size + maxLeadingLen >= (int) kMaxPrefixSize || suffix_size >= (int) sizeof(d_suffix)) {
        result.ok = false;
        result.error = "prefix (up to " + std::to_string(prefix_size + maxLeadingLen) + " once extended by leading candidate characters) or suffix (" +
                        std::to_string(suffix_size) + ") too long for device buffers (max: " + std::to_string(kMaxPrefixSize) + " each)";
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
    printf("maxBackslashCount: %d%s (leading characters only)\n", req.maxBackslashCount,
           req.maxBackslashCount == 0 ? " (unlimited)" : "");

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    CUDA_CHECK(cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable)));

    // The per-search tables bruteForceKernel reads at compile-time-constant
    // indices - see their declaration in hash_kernels.cuh.
    {
        uint32_t h_alphabetKey[MAX_ALPHABET_SIZE] = {0};
        uint32_t h_alphabetOrd[MAX_ALPHABET_SIZE] = {0};
        uint32_t h_suffixKey[64] = {0};
        for (int k = 0; k < alphabetSize; ++k) {
            h_alphabetOrd[k] = (unsigned char) req.alphabet[k];
            h_alphabetKey[k] = h_cryptTable[0x100 + h_alphabetOrd[k]];
        }
        for (int i = 0; i < suffix_size; ++i)
            h_suffixKey[i] = h_cryptTable[0x100 + (unsigned char) req.suffix[i]];
        CUDA_CHECK(cudaMemcpyToSymbol(d_alphabetKey, h_alphabetKey, sizeof(h_alphabetKey)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_alphabetOrd, h_alphabetOrd, sizeof(h_alphabetOrd)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_suffixKey, h_suffixKey, sizeof(h_suffixKey)));
    }

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
    DeviceBuffers bufs;
    CUDA_CHECK(cudaMalloc(&bufs.matches, MAX_MATCHES * MAX_FILENAME_LEN));
    CUDA_CHECK(cudaMalloc(&bufs.matchIdx, MAX_MATCHES * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&bufs.results, sizeof(BatchResults)));
    // A long-lived process (coordinator mode) can call runSearch() many times
    // over its lifetime, one per claimed range - zeroing this fresh allocation
    // every call (rather than relying on any process-lifetime state) is what
    // keeps a previous range's match from making every subsequent range
    // falsely report "found" immediately. It also establishes the
    // "matchCount is 0 at launch" invariant runCudaBatch maintains from here.
    CUDA_CHECK(cudaMemset(bufs.results, 0, sizeof(BatchResults)));

    bool found_match = false;
    bool aborted = false;
    char foundFilename[MAX_FILENAME_LEN] = {0};
    std::string start_candidate = req.startCandidate;
    int candidateLen = start_candidate.size();
    // Candidates per kernel launch: a whole number of rows (see the
    // terminology comment above bruteForceKernel), NAMEBREAK_ROWS_PER_LAUNCH
    // of them (constants.h). Launch boundaries always land on row boundaries
    // (except at a range's own start/end - see runCudaBatch), so almost every
    // row a launch handles takes bruteForceKernel's fast path.
    const uint64_t batchSize = (uint64_t) alphabetSize * NAMEBREAK_ROWS_PER_LAUNCH;

    // The search space is walked by four nested levels, outermost to innermost:
    //  1. This `while` loop: over candidateLen itself - "try every 1-character
    //     candidate, then every 2-character one, ..." Only continuous mode
    //     (req.continuous) actually loops here more than once; bounded mode
    //     runs the body for req.startCandidate's own length and exits via the
    //     `if (!req.continuous)` check at the bottom.
    //  2. The `for (leadingIdx ...)` loop below: over the *leading* part of the
    //     candidate (see gpuWindowChars/leadingLen above) - every leading value
    //     is hashed on the CPU (incrementally - IncrementalPrefixHasher) and
    //     handed to the GPU as its starting seed, so candidates longer than
    //     gpuWindowChars still get covered exhaustively. Since gpuWindowChars
    //     is small (throughput-tuned, not "as large as safely possible"),
    //     leadingLen > 0 - and this loop actually doing work - is the common
    //     case, not the exception; it only degenerates to a single iteration
    //     when candidateLen <= gpuWindowChars.
    //  3. The `for (i = trailStart ...)` loop: chops the (up to
    //     alphabetSize^trailingLen) remaining space for one leading value into
    //     batchSize-sized chunks (aligned to multiples of batchSize, so only a
    //     range's own first/last chunk can start/end mid-row) - each iteration
    //     is one runCudaBatch call, i.e. one kernel launch (plus a second,
    //     tiny one only if that launch had a hashA hit).
    //  4. Inside runCudaBatch: one GPU thread per *row* of the chunk, each
    //     looping over its row's alphabetSize candidates - see
    //     bruteForceKernel's comment.
    if (candidateLen < 1) {
        // bruteForceKernel enumerates at least one trailing character; a zero-length
        // candidate (a range whose lower bound is empty) isn't something it can search.
        result.ok = false;
        result.error = "the start candidate must not be empty";
        goto breakfree;
    }
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
        if (trailingLen > MAX_TRAILING_LEN) {
            // Only reachable if MAX_CANDIDATE_LEN/the alphabet sizes change so that
            // candidateLen - maxSafeIndexLen exceeds what bruteForceKernel's 32-bit row
            // index supports (see MAX_TRAILING_LEN) - refuse rather than overflow it.
            fprintf(stderr, "candidateLen (%d) needs a %d-character trailing part, which exceeds MAX_TRAILING_LEN (%d) - exiting\n",
                    candidateLen, trailingLen, MAX_TRAILING_LEN);
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
            // kernel launch included - cheaper than even one of those
            // candidates would have cost individually.
            if (req.pruneSymbolRuns && hasForbiddenSymbolRun_CPU(leading))
                continue;
            if (req.maxBackslashCount != 0 && countBackslashes_CPU(leading) > req.maxBackslashCount)
                continue;

            // Handed to every runCudaBatch call below as a kernel argument; see BatchParams.
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
                int r = runCudaBatch(trailingLen, i, count, req.targetHashA, req.targetHashB, params, fout, bufs,
                                      alphabetSize, suffix_size, abortRequested, onPartialMatch, foundFilename, pauseRequested);
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

    CUDA_CHECK(cudaFree(bufs.matches));
    CUDA_CHECK(cudaFree(bufs.matchIdx));
    CUDA_CHECK(cudaFree(bufs.results));
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
