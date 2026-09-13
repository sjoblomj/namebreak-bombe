// Measures whether shrinking the GPU's per-thread candidate window (and
// folding the rest into a CPU-cached, incrementally-hashed "leading" prefix -
// see IncrementalPrefixHasher in cpu-utils.h) is actually worth it here, and
// if so, how small the window should be.
//
// The tradeoff being measured: a smaller GPU window means each GPU thread
// hashes fewer characters (cheaper kernel), but it also means a new leading
// value - and therefore a new d_prefix/d_seed1_start/d_seed2_start upload -
// has to happen far more often (once per *batch* instead of once per
// *search*). Whether that's a net win depends on real host<->device
// round-trip costs in this environment, which is exactly what this measures
// instead of assuming.
//
// Every variant (baseline and every window size) launches bruteForceKernel's
// real decode -> prune -> hash pipeline (via hash_kernels.cuh, not a copy of
// it) and calls cudaDeviceSynchronize() after every single kernel launch,
// mirroring runCudaBatch's real behavior (namebreak.cu) rather than letting
// launches pipeline asynchronously, which the real code doesn't do either.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include "../constants.h"
#include "../hash_kernels.cuh"
#include "../cpu-utils.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

template<int AlphabetSize>
__global__ void benchKernel(int candidateLen, uint64_t startIdx, uint64_t total, uint32_t targetA, unsigned long long* sink) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    idx += startIdx;

    char candidate[MAX_CANDIDATE_LEN];
    indexToCandidate<AlphabetSize>(idx, candidateLen, candidate);

    // No pruning here - matches bruteForceKernel's real behavior
    // (namebreak.cu): pruneSymbolRuns/maxBackslashCount are CPU-side only now.
    uint32_t hashA = mpqHashCandidateAndSuffix(candidate, candidateLen);
    if (hashA == targetA) {
        atomicAdd(sink, 1ULL);
    }
}

static void launchAndSync(int candidateLen, uint64_t startIdx, uint64_t count, unsigned long long* d_sink) {
    int threadsPerBlock = 256;
    int blocks = (int) ((count + threadsPerBlock - 1) / threadsPerBlock);
    benchKernel<49><<<blocks, threadsPerBlock>>>(candidateLen, startIdx, count, 0xFFFFFFFFu, d_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct Result {
    std::string label;
    double seconds;
    uint64_t candidates;
};

static void report(const Result& r) {
    double throughputG = r.candidates / r.seconds / 1e9;
    printf("%-42s  %8.3f s   %10llu candidates   %6.3f G/s\n",
           r.label.c_str(), r.seconds, (unsigned long long) r.candidates, throughputG);
}

int main() {
    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    CUDA_CHECK(cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable)));

    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"; // real, 49 chars
    CUDA_CHECK(cudaMemcpyToSymbol(d_alphabet, alphabet.c_str(), alphabet.size() + 1));

    const std::string prefix = "REZ\\";
    const std::string suffix = ".WAV";
    short suffixSize = (short) suffix.size();
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix_size, &suffixSize, sizeof(suffixSize)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_suffix, suffix.c_str(), suffixSize + 1));

    unsigned long long* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(unsigned long long)));

    const int candidateLen = 10;
    const uint64_t iterations = 5000; // number of kernel launches, held constant across every variant

    std::vector<Result> results;

    // --- Baseline: today's behavior - whole 10-character candidate hashed
    // per GPU thread, prefix/seed uploaded exactly once, total. ---
    {
        auto baseState = mpqHashWithPrefixCache_CPU(prefix.c_str(), h_cryptTable);
        short prefixSize = (short) prefix.size();
        CUDA_CHECK(cudaMemcpyToSymbol(d_prefix_size, &prefixSize, sizeof(prefixSize)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_prefix, prefix.c_str(), prefixSize + 1));
        CUDA_CHECK(cudaMemcpyToSymbol(d_seed1_start, &baseState.first, sizeof(baseState.first)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_seed2_start, &baseState.second, sizeof(baseState.second)));

        const uint64_t batchSize = 5'764'801; // 49^4, this project's existing fixed batch size
        launchAndSync(candidateLen, 0, batchSize, d_sink); // warm-up, untimed

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (uint64_t b = 0; b < iterations; ++b) {
            launchAndSync(candidateLen, b * batchSize, batchSize, d_sink);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        results.push_back({"baseline (candidateLen=10, prefix set once)", ms / 1000.0, iterations * batchSize});
    }

    // --- Candidate GPU window sizes: only windows where alphabetSize^W fits
    // in one batch (<= 5,764,801) are tried, so each leading value maps to
    // exactly one kernel launch (no extra chunking-loop layer needed). For
    // alphabet size 49 that's W in {1,2,3,4} (49^4 == 5,764,801 exactly). ---
    for (int W : {1, 2, 3, 4}) {
        int leadingLen = candidateLen - W;
        uint64_t windowSpace = 1;
        for (int i = 0; i < W; ++i) windowSpace *= alphabet.size();

        auto baseState = mpqHashWithPrefixCache_CPU(prefix.c_str(), h_cryptTable);
        IncrementalPrefixHasher hasher(baseState, leadingLen, alphabet, h_cryptTable);
        hasher.reset(0);

        // Warm-up, untimed.
        {
            std::string extended = prefix + hasher.leading();
            short prefixSize = (short) extended.size();
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix_size, &prefixSize, sizeof(prefixSize)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix, extended.c_str(), prefixSize + 1));
            auto s = hasher.state();
            CUDA_CHECK(cudaMemcpyToSymbol(d_seed1_start, &s.first, sizeof(s.first)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_seed2_start, &s.second, sizeof(s.second)));
            launchAndSync(W, 0, windowSpace, d_sink);
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (uint64_t b = 0; b < iterations; ++b) {
            hasher.advance();
            std::string extended = prefix + hasher.leading();
            short prefixSize = (short) extended.size();
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix_size, &prefixSize, sizeof(prefixSize)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_prefix, extended.c_str(), prefixSize + 1));
            auto s = hasher.state();
            CUDA_CHECK(cudaMemcpyToSymbol(d_seed1_start, &s.first, sizeof(s.first)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_seed2_start, &s.second, sizeof(s.second)));

            launchAndSync(W, 0, windowSpace, d_sink);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        char label[128];
        snprintf(label, sizeof(label), "GPU window=%d chars (leading=%d, %llu/launch)", W, leadingLen,
                  (unsigned long long) windowSpace);
        results.push_back({label, ms / 1000.0, iterations * windowSpace});
    }

    printf("\n%-42s  %10s   %14s   %8s\n", "variant", "time", "candidates", "throughput");
    for (const auto& r : results) report(r);

    printf("\nAll variants ran %llu kernel launches (with a sync after each, like runCudaBatch does).\n",
           (unsigned long long) iterations);
    printf("Compare 'GPU window=4' against 'baseline' directly - both launch batches of the same\n");
    printf("size (5,764,801 threads), so that pair isolates exactly the tradeoff being tested:\n");
    printf("cheaper per-thread hashing (4 characters vs 10) against the added per-batch cost of\n");
    printf("advance() + the 4 cudaMemcpyToSymbol uploads that batch-scoped d_prefix/d_seed*_start now need.\n");

    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
