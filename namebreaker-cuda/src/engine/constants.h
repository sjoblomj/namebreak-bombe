#ifndef NAMEBREAK_ENGINE_CONSTANTS_H
#define NAMEBREAK_ENGINE_CONSTANTS_H

// Upper bound on the alphabet's character count, sizing the device-side
// d_alphabet buffer. Must be >= the largest size in the fixed set of
// compile-time-templated sizes runCudaBatch() dispatches on in cuda_backend.cu -
// bump this and add a matching dispatch branch to support a bigger alphabet.
#define MAX_ALPHABET_SIZE 50
#define MAX_CANDIDATE_LEN 16
#define MAX_FILENAME_LEN 128
// How many hashA hits one kernel launch can record. More than this in a single
// launch is handled by re-searching that launch's range in halves (see
// runCudaBatch in cuda_backend.cu), so it only bounds a buffer, not correctness.
// Overridable at compile time (-DMAX_MATCHES=N) so tests/search_overflow_test.cu
// can exercise that path with just a couple of colliding candidates instead of
// needing >1024 of them. Must be >= 1.
#ifndef MAX_MATCHES
#define MAX_MATCHES 1024
#endif

// How many trailing characters of a candidate the GPU enumerates directly
// (the rest - the "leading" part - is folded into the prefix on the CPU, one
// value per group of GPU launches). Overridable at compile time
// (-DNAMEBREAK_GPU_WINDOW_CHARS=N) so tests can force a different leading/
// trailing split and benchmarks can re-sweep it. Lives here, not in
// cuda_backend.cu, so the tests derive their expectations from the same value
// instead of duplicating it. See cuda_backend.cu (runSearch) and README.md's
// "Design decisions" for what it trades off.
#ifndef NAMEBREAK_GPU_WINDOW_CHARS
#define NAMEBREAK_GPU_WINDOW_CHARS 5
#endif

// How many *rows* one kernel launch covers at most (a row = every value of a
// candidate's last character, for one combination of its other trailing
// characters - see bruteForceKernel in cuda_backend.cu), so one launch covers
// at most kRowsPerLaunch * alphabetSize candidates. Bounds how long a single
// launch can run (pause/abort are only polled between launches). Overridable
// at compile time (-DNAMEBREAK_ROWS_PER_LAUNCH=N) so tests can force many
// small launches and exercise the chunk boundaries with tiny ranges.
#ifndef NAMEBREAK_ROWS_PER_LAUNCH
#define NAMEBREAK_ROWS_PER_LAUNCH (1u << 23)
#endif

// Largest trailing (GPU-enumerated) length the kernel supports. A row index
// (alphabetSize^(trailingLen-1)) must fit in 32 bits: 50^5 < 2^32 <= 50^6.
#define MAX_TRAILING_LEN 6
static_assert(NAMEBREAK_GPU_WINDOW_CHARS >= 1 && NAMEBREAK_GPU_WINDOW_CHARS <= MAX_TRAILING_LEN,
              "NAMEBREAK_GPU_WINDOW_CHARS must be between 1 and MAX_TRAILING_LEN");
static_assert(MAX_MATCHES >= 1, "MAX_MATCHES must be at least 1 (a one-candidate range can have one hit)");
static_assert(NAMEBREAK_ROWS_PER_LAUNCH >= 1, "NAMEBREAK_ROWS_PER_LAUNCH must be >= 1");

#endif // NAMEBREAK_ENGINE_CONSTANTS_H
