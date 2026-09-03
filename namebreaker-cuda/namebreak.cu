#include <cuda_runtime.h>
#include "cpu-utils.h"
#include "constants.h"

// Terminology:
// * Candidate = The part of the name that we are brute-forcing
// * Filename  = The Prefix + Candidate + Suffix

__device__ __constant__ char d_alphabet[ALPHABET_SIZE + 1] = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_";
const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_";

__device__ volatile int d_foundMatchFlag = 0;
__device__ __constant__ char d_prefix[64];
__device__ __constant__ char d_suffix[64];
__device__ __constant__ char d_lowerBound[64];
__device__ __constant__ char d_upperBound[64];
__device__ __constant__ char lowerBound[64];
__device__ __constant__ char upperBound[64];
__device__ __constant__ short d_prefix_size;
__device__ __constant__ short d_suffix_size;
__device__ __constant__ uint32_t d_seed1_start;
__device__ __constant__ uint32_t d_seed2_start;

__device__ __constant__ uint32_t d_cryptTable[0x500];

__device__ uint32_t mpqHash(const char* str) {
    uint32_t seed1 = d_seed1_start;
    uint32_t seed2 = d_seed2_start;
    char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

__device__ uint32_t mpqHashSeed2(const char* str) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    char ch;

    while ((ch = *str++) != '\0') {
        seed1 = d_cryptTable[0x200 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    return seed1;
}

__device__ void indexToCandidate(uint64_t index, int candidateLen, char* outCandidate) {
    for (int i = candidateLen - 1; i >= 0; --i) {
        outCandidate[i] = d_alphabet[index % ALPHABET_SIZE];
        index /= ALPHABET_SIZE;
    }
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

__device__ void buildFilenameWithoutPrefix(const char* candidate, int candidateLen, char* out) {
    memcpy(out, candidate, candidateLen);
    memcpy(out + candidateLen, d_suffix, d_suffix_size);

    out[candidateLen + d_suffix_size] = '\0';
}

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
    char filename[MAX_FILENAME_LEN];

    indexToCandidate(idx, candidateLen, candidate);

    buildFilenameWithoutPrefix(candidate, candidateLen, filename);

    uint32_t hashA = mpqHash(filename);
    if (hashA == targetA) {
        buildCompleteFilename(candidate, candidateLen, filename);
        printf("Hash A matches: %s\n", filename);

        uint32_t hashB = mpqHashSeed2(filename);
        if (hashB == targetB) {
            printf("BOTH HASHES MATCH: %s\n", filename);
            d_foundMatchFlag = 1;
        }

        int slot = atomicAdd(d_matchCount, 1);
        if (slot < MAX_MATCHES) {
            memcpy(&d_matches[slot * MAX_FILENAME_LEN], filename, MAX_FILENAME_LEN);
        }
    }
}


int runCudaBatch(int candidateLen, uint64_t startIdx, uint64_t count, uint32_t targetA, uint32_t targetB, FILE* fout) {
    int h_flag = 0;
    cudaMemcpyFromSymbol(&h_flag, d_foundMatchFlag, sizeof(int));

    if (h_flag) return h_flag;
    char* d_matches;
    int* d_matchCount;
    cudaMalloc(&d_matches, MAX_MATCHES * MAX_FILENAME_LEN);
    cudaMalloc(&d_matchCount, sizeof(int));
    cudaMemset(d_matchCount, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    bruteForceKernel<<<blocks, threadsPerBlock>>>(
            candidateLen, startIdx, count, targetA, targetB, d_matches, d_matchCount
    );
    cudaDeviceSynchronize();

    int h_matchCount = 0;
    cudaMemcpy(&h_matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost);
    h_matchCount = std::min(h_matchCount, MAX_MATCHES);

    char h_matches[MAX_MATCHES][MAX_FILENAME_LEN];
    cudaMemcpy(h_matches, d_matches, h_matchCount * MAX_FILENAME_LEN, cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_matchCount; ++i) {
        fprintf(fout, "%s\n", h_matches[i]);
        fflush(fout);
    }

    cudaFree(d_matches);
    cudaFree(d_matchCount);
    return h_flag;
}

int main(int argc, char* argv[]) {
    if (argc < 9 || (strcmp(argv[1], "continuous") != 0 && strcmp(argv[1], "bounded") != 0)) {
        fprintf(stderr, "Usage: %s <continuous|bounded> <startCandidate> <prefix> <suffix> <lowerBound> <upperBound> <targetHashA> <targetHashB>\n", argv[0]);
        return 1;
    }
    if (alphabet.size() != ALPHABET_SIZE) {
        // This is just a check to make sure we don't change the alphabet without updating its size
        fprintf(stderr, "Alphabet size mismatch. Expected %d, got %zu\n", ALPHABET_SIZE, alphabet.size());
        return 1;
    }

    std::string operation = argv[1];
    std::string prefix = argv[3];
    std::string suffix = argv[4];
    std::string start_candidate = getStartCandidate(argv[2], prefix, suffix);
    std::string lowerBound = argv[5];
    std::string upperBound = argv[6];
    uint32_t target_hash_A = std::stoul(argv[7], nullptr, 16);
    uint32_t target_hash_B = std::stoul(argv[8], nullptr, 16);

    std::string lower = remove_prefix_and_suffix(lowerBound, prefix, suffix);
    std::string upper = remove_prefix_and_suffix(upperBound, prefix, suffix);

    short prefix_size = prefix.size();
    short suffix_size = suffix.size();
    cudaMemcpyToSymbol(d_prefix_size, &prefix_size, sizeof(prefix_size));
    cudaMemcpyToSymbol(d_suffix_size, &suffix_size, sizeof(suffix_size));
    cudaMemcpyToSymbol(d_prefix, prefix.c_str(), prefix_size + 1);
    cudaMemcpyToSymbol(d_suffix, suffix.c_str(), suffix_size + 1);
    cudaMemcpyToSymbol(d_lowerBound, lowerBound.c_str(), lowerBound.size() + 1);
    cudaMemcpyToSymbol(d_upperBound, upperBound.c_str(), upperBound.size() + 1);
    cudaMemcpyToSymbol(lowerBound, lower.c_str(), lower.size() + 1);
    cudaMemcpyToSymbol(upperBound, upper.c_str(), upper.size() + 1);

    std::string lowerBoundLimit = getLowerBound(lower, alphabet);
    std::string upperBoundLimit = getUpperBound(upper, alphabet);

    printf("candidate: '%s'\n", start_candidate.c_str());
    printf("prefix: '%s'\n", prefix.c_str());
    printf("suffix: '%s'\n", suffix.c_str());
    printf("lowerBound: '%s'\n", lowerBound.c_str());
    printf("upperBound: '%s'\n", upperBound.c_str());
    printf("lower: '%s'\n", lower.c_str());
    printf("upper: '%s'\n", upper.c_str());
    printf("lowerBoundLimit: '%s'\n", lowerBoundLimit.c_str());
    printf("upperBoundLimit: '%s'\n", upperBoundLimit.c_str());
    printf("hashA: '%X'\n", target_hash_A);
    printf("hashB: '%X'\n", target_hash_B);

    uint32_t h_cryptTable[0x500];
    prepareCryptTable(h_cryptTable);
    cudaMemcpyToSymbol(d_cryptTable, h_cryptTable, sizeof(h_cryptTable));

    std::pair<uint32_t, uint32_t> pair = mpqHashWithPrefixCache_CPU(prefix.c_str(), h_cryptTable);
    uint32_t seed1_start = pair.first;
    uint32_t seed2_start = pair.second;
    cudaMemcpyToSymbol(d_seed1_start, &seed1_start, sizeof(seed1_start));
    cudaMemcpyToSymbol(d_seed2_start, &seed2_start, sizeof(seed2_start));

    FILE* fout = fopen("matches.txt", "a");
    if (!fout) {
        perror("fopen");
        return 1;
    }

    bool found_match = false;
    int candidateLen = start_candidate.size();

    while (true) {
        std::string start_bound = make_bound_string(start_candidate, candidateLen);
        std::string end_bound   = make_bound_string(upperBoundLimit, candidateLen);

        uint64_t startIdx = stringToIndex(start_bound, alphabet);
        uint64_t endIdx   = stringToIndex(end_bound, alphabet);
        printf("Starting at '%s'. Char length = %d → Total combinations: %llu\n", start_bound.c_str(), candidateLen, (unsigned long long)(endIdx - startIdx));

        const uint64_t batchSize = ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE;
        for (uint64_t i = startIdx; i < endIdx; i += batchSize) {
            uint64_t count = std::min(batchSize, endIdx - i);
            if (runCudaBatch(candidateLen, i, count, target_hash_A, target_hash_B, fout) == 1) {
                found_match = true;
                goto breakfree;
            }
        }

        candidateLen += 1;
        start_candidate = lowerBoundLimit;
        if (operation == "bounded") {
            printf("Reached the upper limit - exiting\n");
            goto breakfree;
        }
    }
breakfree:

    fclose(fout);
    return found_match ? 0 : 2;
}
