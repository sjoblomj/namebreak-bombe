// The CUDA backend: the search kernels, and the SearchBackend that launches
// them (see engine/backend.h for what the engine asks of a backend). The
// same file, compiled with HIP for AMD GPUs, is the HIP backend - see
// gpu_runtime.h.

#include "backends/cuda/cuda_backend.h"

#include "backends/cuda/gpu_runtime.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>
#include "backends/cuda/hash_kernels.cuh"
#include "backends/cuda/tuning.h"
#include "engine/backend.h"
#include "engine/limits.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, NAMEBREAK_GPU_RUNTIME_NAME " error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// Terminology (see also engine/backend.h):
// * Trailing part = the last `trailingLen` characters of the candidate; the
//   part the GPU enumerates (the rest - the "leading" part - is folded into
//   the prefix on the CPU, see runSearch).
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
    int matchCount; // hashA hits this batch (may exceed MAX_MATCHES; the engine then re-searches the range in halves)
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

static_assert(kThreadsPerBlock >= MAX_ALPHABET_SIZE, "bruteForceKernel needs one thread per alphabet entry to fill its shared tables");

// Suffix lengths 0-8 (see dispatchSuffixLen) get their own compile-time
// instantiation of bruteForceKernel, fully unrolled - measured ~2x faster than
// looping over a runtime length; anything longer falls back to kRuntimeSuffix.
constexpr int kRuntimeSuffix = -1;

// One step of the MPQ hash recurrence (hashA table, offset 0x100) for a
// character whose crypt-table key `key` and value `ord` are already known.
// Must match mpqHashCandidateAndSuffix/mpq_hash.cpp exactly, or a match
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

// No maxBackslashCount, pruneSymbolRuns or pruneUnopenedBrackets check here -
// all are applied only to the leading characters, on the CPU, before this kernel is
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
        // trailingLen is validated against kMaxTrailingLen (== 6) by runSearch
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
// found flag/filename. Deliberately implemented with the
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

    uint32_t hashB = mpqHashSeed2(filename);
    if (hashB == targetB) {
        memcpy(d_foundFilename, filename, MAX_FILENAME_LEN);
        bufs.results->foundFlag = 1;
    }

    memcpy(&bufs.matches[(size_t) i * MAX_FILENAME_LEN], filename, MAX_FILENAME_LEN);
}

// alphabetSize/suffix length have to be dispatched to one of a fixed set of
// compile-time template instantiations (see indexToCandidate's and
// bruteForceKernel's comments for why) - dispatchAlphabetSize/
// dispatchSuffixLen call `f` with a std::integral_constant of the matching
// value.
//
// The alphabet sizes this backend supports, ascending: add one here (and
// recompile/redistribute namebreak to volunteers) to support a new size.
// Must all be <= MAX_ALPHABET_SIZE.
template<int... Sizes>
struct AlphabetSizeList {};
using SupportedAlphabetSizes = AlphabetSizeList<42, 43, 47, 48, 49, 50>;

template<typename F, int... Sizes>
bool dispatchAlphabetSizeIn(AlphabetSizeList<Sizes...>, int alphabetSize, F&& f) {
    static_assert(((Sizes <= MAX_ALPHABET_SIZE) && ...), "an alphabet size exceeds MAX_ALPHABET_SIZE");
    return ((alphabetSize == Sizes ? (f(std::integral_constant<int, Sizes>{}), true) : false) || ...);
}

template<int... Sizes>
std::vector<int> alphabetSizesIn(AlphabetSizeList<Sizes...>) {
    return {Sizes...};
}

// Returns false if `alphabetSize` isn't in SupportedAlphabetSizes.
template<typename F>
bool dispatchAlphabetSize(int alphabetSize, F&& f) {
    return dispatchAlphabetSizeIn(SupportedAlphabetSizes{}, alphabetSize, f);
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

namespace {

class CudaBackend : public SearchBackend {
public:
#ifdef NAMEBREAK_HIP
    const char* name() const override { return "hip"; }
#else
    const char* name() const override { return "cuda"; }
#endif
    std::vector<int> supportedAlphabetSizes() const override { return alphabetSizesIn(SupportedAlphabetSizes{}); }
    int windowChars() const override { return NAMEBREAK_GPU_WINDOW_CHARS; }
    int maxTrailingLen() const override { return kMaxTrailingLen; }
    // A whole number of rows (see the terminology comment above
    // bruteForceKernel), NAMEBREAK_ROWS_PER_LAUNCH of them (tuning.h). Launch
    // boundaries then always land on row boundaries (except at a range's own
    // start/end), so almost every row a launch handles takes
    // bruteForceKernel's fast path.
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * NAMEBREAK_ROWS_PER_LAUNCH; }

    void beginSearch(const SearchConstants& constants) override;
    BatchOutcome runBatch(int trailingLen, uint64_t startIdx, uint64_t count, const BatchParams& params) override;
    void endSearch() override;

private:
    int alphabetSize_ = 0;
    int suffixLen_ = 0;
    uint32_t targetA_ = 0;
    uint32_t targetB_ = 0;
    DeviceBuffers bufs_ = {};
};

void CudaBackend::beginSearch(const SearchConstants& constants) {
    alphabetSize_ = (int) constants.alphabet.size();
    suffixLen_ = (int) constants.suffix.size();
    targetA_ = constants.targetHashA;
    targetB_ = constants.targetHashB;

    short suffixSize = (short) suffixLen_;
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix_size, &suffixSize, sizeof(suffixSize)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix, constants.suffix.c_str(), suffixLen_ + 1));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alphabet, constants.alphabet.c_str(), alphabetSize_ + 1));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cryptTable, constants.cryptTable, 0x500 * sizeof(uint32_t)));

    // The per-search tables bruteForceKernel reads at compile-time-constant
    // indices - see their declaration in hash_kernels.cuh.
    {
        uint32_t h_alphabetKey[MAX_ALPHABET_SIZE] = {0};
        uint32_t h_alphabetOrd[MAX_ALPHABET_SIZE] = {0};
        uint32_t h_suffixKey[kMaxSuffixSize] = {0};
        for (int k = 0; k < alphabetSize_; ++k) {
            h_alphabetOrd[k] = (unsigned char) constants.alphabet[k];
            h_alphabetKey[k] = constants.cryptTable[0x100 + h_alphabetOrd[k]];
        }
        for (int i = 0; i < suffixLen_; ++i)
            h_suffixKey[i] = constants.cryptTable[0x100 + (unsigned char) constants.suffix[i]];
        CUDA_CHECK(cudaMemcpyToSymbol(d_alphabetKey, h_alphabetKey, sizeof(h_alphabetKey)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_alphabetOrd, h_alphabetOrd, sizeof(h_alphabetOrd)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_suffixKey, h_suffixKey, sizeof(h_suffixKey)));
    }

    // Allocated once per search and reused for every batch, instead of
    // malloc/free per call - these buffers are always the same size, so
    // there's no reason to pay driver allocation overhead on every single
    // kernel launch.
    CUDA_CHECK(cudaMalloc(&bufs_.matches, MAX_MATCHES * MAX_FILENAME_LEN));
    CUDA_CHECK(cudaMalloc(&bufs_.matchIdx, MAX_MATCHES * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&bufs_.results, sizeof(BatchResults)));
    // A long-lived process (coordinator mode) runs many searches over its
    // lifetime, one per claimed range - zeroing this fresh allocation every
    // search (rather than relying on any process-lifetime state) is what
    // keeps a previous range's match from making every subsequent range
    // falsely report "found" immediately. It also establishes the
    // "matchCount is 0 at launch" invariant runBatch maintains from here.
    CUDA_CHECK(cudaMemset(bufs_.results, 0, sizeof(BatchResults)));
}

void CudaBackend::endSearch() {
    CUDA_CHECK(cudaFree(bufs_.matches));
    CUDA_CHECK(cudaFree(bufs_.matchIdx));
    CUDA_CHECK(cudaFree(bufs_.results));
    bufs_ = {};
}

BatchOutcome CudaBackend::runBatch(int trailingLen, uint64_t startIdx, uint64_t count, const BatchParams& params) {
    const int alphabetSize = alphabetSize_;
    BatchOutcome outcome;

    // No pre-launch check of bufs_.results->foundFlag: beginSearch zeroes it
    // before the first batch, and a batch that finds a match reports it
    // (below) - so runSearch stops calling this - rather than leaving it set
    // for a later call to notice. bufs_.results->matchCount is likewise already 0
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
        dispatchSuffixLen(suffixLen_, [&](auto suffixC) {
            constexpr int SuffixLen = decltype(suffixC)::value;
            bruteForceKernel<AlphabetSize, SuffixLen><<<blocks, kThreadsPerBlock>>>(
                    trailingLen, (uint32_t) firstRow, (uint32_t) rowCount, firstRowStartK, lastRowEndK, targetA_, params, bufs_);
        });
    });
    if (!supported) {
        // runSearch checks supportedAlphabetSizes() before any batch.
        fprintf(stderr, "Unsupported alphabet size: %d\n", alphabetSize);
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    BatchResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, bufs_.results, sizeof(h_results), cudaMemcpyDeviceToHost));
    outcome.hitCount = h_results.matchCount;

    // Hits are rare (a few per thousand batches), so everything below is the
    // exception path - the common batch costs exactly the one cudaMemcpy above.
    if (h_results.matchCount > MAX_MATCHES) {
        // More hits than bufs_.matchIdx can record: the excess would never reach
        // verifyMatchesKernel. Nothing from this launch has been verified, so
        // leave it all to the engine to search again in halves.
        CUDA_CHECK(cudaMemset(&bufs_.results->matchCount, 0, sizeof(int)));
        return outcome;
    }
    if (h_results.matchCount > 0) {
        const int recorded = h_results.matchCount;

        dispatchAlphabetSize(alphabetSize, [&](auto alphabetC) {
            constexpr int AlphabetSize = decltype(alphabetC)::value;
            verifyMatchesKernel<AlphabetSize><<<(recorded + 63) / 64, 64>>>(trailingLen, recorded, targetA_, targetB_, params, bufs_);
        });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<char> h_matches((size_t) recorded * MAX_FILENAME_LEN);
        CUDA_CHECK(cudaMemcpy(h_matches.data(), bufs_.matches, h_matches.size(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < recorded; ++i)
            outcome.hits.emplace_back(&h_matches[(size_t) i * MAX_FILENAME_LEN]);

        // verifyMatchesKernel is what sets foundFlag, so re-read it now.
        CUDA_CHECK(cudaMemcpy(&h_results, bufs_.results, sizeof(h_results), cudaMemcpyDeviceToHost));

        // Restores the "matchCount is 0 at launch" invariant the next batch relies on.
        CUDA_CHECK(cudaMemset(&bufs_.results->matchCount, 0, sizeof(int)));
    }

    // Read after this batch's own kernels have finished, so a match found by
    // *this* batch is reported now instead of going undetected until
    // whatever call happens to come after it, which may never come (e.g. a
    // "bounded" search whose very last batch is the one that finds it would
    // otherwise report "not found").
    if (h_results.foundFlag) {
        char foundFilename[MAX_FILENAME_LEN];
        CUDA_CHECK(cudaMemcpyFromSymbol(foundFilename, d_foundFilename, MAX_FILENAME_LEN));
        outcome.found = true;
        outcome.foundFilename = foundFilename;
    }
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeCudaBackend(std::string& error) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        error = std::string("no usable " NAMEBREAK_GPU_RUNTIME_NAME " device: ") + cudaGetErrorString(err);
        return nullptr;
    }
    if (deviceCount == 0) {
        error = "no " NAMEBREAK_GPU_RUNTIME_NAME " device";
        return nullptr;
    }
    return std::make_unique<CudaBackend>();
}
