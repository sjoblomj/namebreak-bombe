#ifndef NAMEBREAK_CUDA_CONSTANTS_H
#define NAMEBREAK_CUDA_CONSTANTS_H

// Upper bound on the alphabet's character count, sizing the device-side
// d_alphabet buffer. Must be >= the largest size in the fixed set of
// compile-time-templated sizes runCudaBatch() dispatches on in namebreak.cu -
// bump this and add a matching dispatch branch to support a bigger alphabet.
#define MAX_ALPHABET_SIZE 50
#define MAX_CANDIDATE_LEN 16
#define MAX_FILENAME_LEN 128
#define MAX_MATCHES 1024

#endif //NAMEBREAK_CUDA_CONSTANTS_H
