#include <cuda_runtime.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include "cpu-utils.h"
#include "constants.h"
#include "search.h"
#include "config.h"
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

// Sized to the largest alphabet this build supports (see MAX_ALPHABET_SIZE);
// populated at runtime via cudaMemcpyToSymbol from the CLI's <alphabet> argument,
// the same pattern already used for d_prefix/d_suffix below.
__device__ __constant__ char d_alphabet[MAX_ALPHABET_SIZE + 1];

__device__ volatile int d_foundMatchFlag = 0;
// Written by the kernel alongside d_foundMatchFlag, so the host can recover
// *which* candidate matched both hashes without scanning stdout for it - see
// buildCompleteFilename below for how it's populated.
__device__ char d_foundFilename[MAX_FILENAME_LEN];
__device__ __constant__ char d_prefix[64];
__device__ __constant__ char d_suffix[64];
__device__ __constant__ short d_prefix_size;
__device__ __constant__ short d_suffix_size;
__device__ __constant__ uint32_t d_seed1_start;
__device__ __constant__ uint32_t d_seed2_start;
// Max '\' occurrences allowed in a candidate before it's discarded unhashed; 0
// means unlimited (no candidate is ever discarded on this basis - use an
// alphabet without '\' in it if none should ever appear at all). A plain
// runtime constant rather than a template parameter like AlphabetSize: this is
// just an integer compare, not a division, so there's no compile-time-constant
// codegen benefit to chase here.
__device__ __constant__ int d_maxBackslashCount;

__device__ __constant__ uint32_t d_cryptTable[0x500];

// Hashes `candidate` followed by d_suffix directly, without ever concatenating them
// into a scratch buffer first. Starts from d_seed1_start/d_seed2_start, which already
// account for the (extended) prefix's contribution - see mpqHashWithPrefixCache_CPU.
// This is the hot path (every thread runs it), so avoiding the extra buffer write+read
// that buildFilenameWithoutPrefix + a buffer-based hash would need is worth it; the
// full filename is only built (via buildCompleteFilename) on the rare hashA match below.
__device__ uint32_t mpqHashCandidateAndSuffix(const char* candidate, int candidateLen) {
    uint32_t seed1 = d_seed1_start;
    uint32_t seed2 = d_seed2_start;

    // unsigned so a byte >= 0x80 zero-extends into the crypt-table index/seed
    // arithmetic instead of sign-extending to a negative value - must match
    // cpu-utils.cpp's host-side hash exactly, or a match found on one side
    // would never reproduce on the other.
    for (int i = 0; i < candidateLen; ++i) {
        unsigned char ch = candidate[i];
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    for (int i = 0; i < d_suffix_size; ++i) {
        unsigned char ch = d_suffix[i];
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

__device__ uint32_t mpqHashSeed2(const char* str) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    // unsigned - see mpqHashCandidateAndSuffix above.
    unsigned char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x200 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

// AlphabetSize is a compile-time template parameter (mirroring PruneSymbolRuns
// below) so this modulus/division - run once per candidate character, for every
// thread - stays a cheap compiler-optimized constant instead of a real (much
// slower) GPU integer division. See runCudaBatch for the fixed set of sizes this
// gets instantiated for and the runtime dispatch between them.
template<int AlphabetSize>
__device__ void indexToCandidate(uint64_t index, int candidateLen, char* outCandidate) {
    for (int i = candidateLen - 1; i >= 0; --i) {
        outCandidate[i] = d_alphabet[index % AlphabetSize];
        index /= AlphabetSize;
    }
}

__device__ __forceinline__ bool isAlnumMpq(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z');
}

// Real MPQ filename components essentially never contain three consecutive
// non-alphanumeric, non-space characters (e.g. "']&_") - used to prune obviously-
// implausible candidates before spending a hash chain on them. Spaces are exempted
// since " - " and " & " are common real word separators (e.g. "Arathi - Lake",
// "Gold Separates East & West") that would otherwise be wrongly pruned. Only
// inspects the candidate itself, not where it joins the (fixed, user-supplied)
// prefix/suffix.
__device__ __forceinline__ bool hasForbiddenSymbolRun(const char* candidate, int candidateLen) {
    int run = 0;
    for (int i = 0; i < candidateLen; ++i) {
        if (isAlnumMpq(candidate[i]) || candidate[i] == ' ') {
            run = 0;
        } else if (++run >= 3) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ int countBackslashes(const char* candidate, int candidateLen) {
    int count = 0;
    for (int i = 0; i < candidateLen; ++i) {
        if (candidate[i] == '\\') count++;
    }
    return count;
}

__device__ void buildCompleteFilename(const char* candidate, int candidateLen, char* out) {
    memcpy(out, d_prefix, d_prefix_size);
    short i = d_prefix_size;

    memcpy(out + i, candidate, candidateLen);
    i += candidateLen;

    memcpy(out + i, d_suffix, d_suffix_size);
    i += d_suffix_size;

    out[i] = '\0';
}

// PruneSymbolRuns is a compile-time template parameter rather than a runtime bool:
// the two instantiations are separate compiled kernels, so the disabled variant
// contains no trace of the check (not even a dead branch) and costs zero cycles
// on this hot path. Which one runs is decided once per batch on the host, in
// runCudaBatch, so the flag is still a normal runtime toggle from the caller's
// point of view. AlphabetSize is the same trick applied to indexToCandidate below.
template<int AlphabetSize, bool PruneSymbolRuns>
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
    if (idx >= total) return;

    idx += startIdx;

    char candidate[MAX_CANDIDATE_LEN];
    indexToCandidate<AlphabetSize>(idx, candidateLen, candidate);

    if constexpr (PruneSymbolRuns) {
        if (hasForbiddenSymbolRun(candidate, candidateLen)) return;
    }

    if (d_maxBackslashCount != 0 && countBackslashes(candidate, candidateLen) > d_maxBackslashCount) return;

    uint32_t hashA = mpqHashCandidateAndSuffix(candidate, candidateLen);
    if (hashA == targetA) {
        char filename[MAX_FILENAME_LEN];
        buildCompleteFilename(candidate, candidateLen, filename);
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
                  char* d_matches, int* d_matchCount, bool pruneSymbolRuns, int alphabetSize,
                  const std::atomic<bool>* abortRequested, const std::function<void(const std::string&)>& onPartialMatch,
                  char* outFoundFilename) {
    if (abortRequested && abortRequested->load(std::memory_order_relaxed)) return -1;

    int h_flag = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_flag, d_foundMatchFlag, sizeof(int)));

    if (h_flag) {
        CUDA_CHECK(cudaMemcpyFromSymbol(outFoundFilename, d_foundFilename, MAX_FILENAME_LEN));
        return h_flag;
    }
    CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(int)));

    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    // alphabetSize has to be dispatched to one of a fixed set of compile-time
    // template instantiations (see indexToCandidate's comment for why) - this is
    // the whole set this build supports. Add a case (and recompile/redistribute
    // namebreak to volunteers) to support a new size.
    #define LAUNCH_WITH_ALPHABET_SIZE(SIZE) \
        if (pruneSymbolRuns) { \
            bruteForceKernel<SIZE, true><<<blocks, threadsPerBlock>>>( \
                    candidateLen, startIdx, count, targetA, targetB, d_matches, d_matchCount); \
        } else { \
            bruteForceKernel<SIZE, false><<<blocks, threadsPerBlock>>>( \
                    candidateLen, startIdx, count, targetA, targetB, d_matches, d_matchCount); \
        }
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
        if (onPartialMatch) onPartialMatch(h_matches[i]);
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

SearchResult runSearch(const SearchRequest& req, std::atomic<bool>* abortRequested, std::function<void(const std::string&)> onPartialMatch) {
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
    // 0 means unlimited (see d_maxBackslashCount's declaration comment).
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
    if (!lowerIsBeforeUpper) {
        result.ok = false;
        result.error = "lower bound ('" + req.lowerBound + "') must be smaller than upper bound ('" + req.upperBound + "')";
        return result;
    }

    short prefix_size = req.prefix.size();
    short suffix_size = req.suffix.size();

    // A candidate's trailing `windowSize` characters are brute-forced directly by the
    // GPU using native 64-bit indices. Any characters beyond that are treated as an
    // extension of the prefix: their contribution to the hash is folded in once per
    // outer iteration on the CPU (see mpqHashWithPrefixCache_CPU below), so a candidate
    // can grow up to MAX_CANDIDATE_LEN without the per-thread index ever overflowing
    // uint64_t. Computed from alphabetSize/MAX_CANDIDATE_LEN rather than hardcoded, so
    // it stays correct if either changes (a smaller alphabet allows a larger window).
    int windowSize = 0;
    {
        uint64_t product = 1;
        while (windowSize < MAX_CANDIDATE_LEN && product <= UINT64_MAX / alphabetSize) {
            product *= alphabetSize;
            windowSize++;
        }
    }
    int maxLeadingLen = MAX_CANDIDATE_LEN - windowSize;
    printf("windowSize: %d (max leading/prefix-extension length: %d)\n", windowSize, maxLeadingLen);

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
    int maxBackslashCount = req.maxBackslashCount;
    CUDA_CHECK(cudaMemcpyToSymbol(d_maxBackslashCount, &maxBackslashCount, sizeof(maxBackslashCount)));

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
    printf("pruneSymbolRuns: %s\n", req.pruneSymbolRuns ? "true" : "false");
    printf("maxBackslashCount: %d%s\n", req.maxBackslashCount, req.maxBackslashCount == 0 ? " (unlimited)" : "");

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    CUDA_CHECK(cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable)));

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

    while (true) {
        if (candidateLen > MAX_CANDIDATE_LEN) {
            fprintf(stderr, "candidateLen (%d) exceeds MAX_CANDIDATE_LEN (%d) - exiting\n", candidateLen, MAX_CANDIDATE_LEN);
            break;
        }

        // Split the candidate into a leading part (folded into the prefix, hashed once
        // per value on the CPU) and a trailing part of at most `windowSize` characters
        // (brute-forced by the GPU with native 64-bit indices). Every combination of the
        // leading part is enumerated too, so the full candidateLen-character space is
        // still covered exhaustively - it's just indexed in two safely-sized pieces
        // instead of one that could overflow uint64_t.
        int trailingLen = std::min(candidateLen, windowSize);
        int leadingLen = candidateLen - trailingLen;
        if (leadingLen > windowSize) {
            fprintf(stderr, "candidateLen (%d) needs a %d-character leading part, which exceeds windowSize (%d) - exiting\n",
                    candidateLen, leadingLen, windowSize);
            break;
        }

        std::string start_full = make_bound_string(start_candidate, candidateLen);
        std::string end_full   = make_bound_string(upperBoundLimit, candidateLen);

        std::string start_leading = start_full.substr(0, leadingLen);
        std::string end_leading   = end_full.substr(0, leadingLen);

        uint64_t startLeadingIdx = 0, endLeadingIdx = 0, trailSpaceSize = 0;
        if (!stringToIndex(start_leading, req.alphabet, startLeadingIdx, result.error) ||
            !stringToIndex(end_leading, req.alphabet, endLeadingIdx, result.error) ||
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

        for (uint64_t leadingIdx = startLeadingIdx; leadingIdx <= endLeadingIdx; ++leadingIdx) {
            std::string leading = indexToString(leadingIdx, leadingLen, req.alphabet);
            std::string extendedPrefix = req.prefix + leading;

            short extPrefixSize = extendedPrefix.size();
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix_size, &extPrefixSize, sizeof(extPrefixSize)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix, extendedPrefix.c_str(), extPrefixSize + 1));

            std::pair<uint32_t, uint32_t> pair = mpqHashWithPrefixCache_CPU(extendedPrefix.c_str(), h_cryptTable);
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

            if ((leadingIdx - startLeadingIdx) % leadingLogInterval == 0) {
                printf("Leading '%s'. Char length = %d → Trailing combinations: %llu\n",
                       leading.c_str(), candidateLen, (unsigned long long)(trailEnd - trailStart));
            }

            for (uint64_t i = trailStart; i < trailEnd; i += batchSize) {
                uint64_t count = std::min(batchSize, trailEnd - i);
                int r = runCudaBatch(trailingLen, i, count, req.targetHashA, req.targetHashB, fout, d_matches, d_matchCount,
                                      req.pruneSymbolRuns, alphabetSize, abortRequested, onPartialMatch, foundFilename);
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

int main(int argc, char* argv[]) {
    // The only argument namebreak takes: an optional mode, overriding
    // config.conf's own `mode = ...` (see config.h). Everything else - which
    // used to be nine positional/flag arguments differing per mode - now
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

#ifdef NAMEBREAK_WITH_NETWORK
    if (mode == "coordinator") {
        CoordinatorArgs cargs;
        if (!buildCoordinatorArgs(config.coordinator, cargs, error)) {
            fprintf(stderr, "%s [coordinator]: %s\n", kConfigPath, error.c_str());
            return 1;
        }
        return runCoordinator(cargs);
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

    SearchResult result = runSearch(req);

    if (!result.ok) {
        fprintf(stderr, "%s\n", result.error.c_str());
        return 1;
    }
    return result.found ? 0 : 2;
}
