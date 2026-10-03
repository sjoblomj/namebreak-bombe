#ifndef NAMEBREAK_BACKENDS_CUDA_HASH_KERNELS_CUH
#define NAMEBREAK_BACKENDS_CUDA_HASH_KERNELS_CUH

#include <cstdint>
#include "engine/backend.h"
#include "engine/limits.h"

// The per-search constants cuda_backend.cu's search kernel reads.
//
// What changes from one launch to the next (its batches' rows and seeds -
// LaunchBatches, cuda_backend.cu) is passed to the kernel by value as a
// kernel argument rather than uploaded to __constant__ symbols with
// cudaMemcpyToSymbol. Those
// uploads were synchronous driver calls that each cost ~6us of GPU idle time
// between two consecutive kernels - a kernel argument rides along with the
// launch itself for free. Only what stays the same for a whole search lives
// in the symbols below.

__device__ __constant__ char d_suffix[kMaxSuffixSize];
__device__ __constant__ short d_suffix_size;

// Per-search tables precomputed on the host from the crypt table, so the
// search kernel never does a data-dependent crypt-table lookup per
// character, which every lane of a warp would then serialize on
// (filteredRowsKernel copies the alphabet's into shared memory for the row's
// digits, and reads the suffix's at compile-time-constant indices, which
// fold into its instructions as constant-bank operands):
//   d_alphabetKey[k] = cryptTable[0x100 + alphabet[k]]   (hashA's per-char key)
//   d_alphabetOrd[k] = (unsigned char) alphabet[k]
//   d_suffixKey[i]   = cryptTable[0x100 + suffix[i]]
__device__ __constant__ uint32_t d_alphabetKey[MAX_ALPHABET_SIZE];
__device__ __constant__ uint32_t d_alphabetOrd[MAX_ALPHABET_SIZE];
__device__ __constant__ uint32_t d_suffixKey[kMaxSuffixSize];
// The text inserted into the trailing part (SearchConstants::
// trailingInsertions): up to two insertions among a row group's characters,
// each before its last groupCharsAfter[n] trailing characters (0: none),
// and one after a row's own character, before the candidate's last. Their
// characters' keys and values are d_insertKey/d_insertOrd[start ..
// start + len).
struct TrailingInsertLayout {
    int groupCharsAfter[2];
    int groupStart[2];
    int groupLen[2];
    int rowStart;
    int rowLen;
};
__device__ __constant__ TrailingInsertLayout d_insertLayout;
__device__ __constant__ uint32_t d_insertKey[2 * kMaxInsertLen];
__device__ __constant__ uint32_t d_insertOrd[2 * kMaxInsertLen];

#endif // NAMEBREAK_BACKENDS_CUDA_HASH_KERNELS_CUH
