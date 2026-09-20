#ifndef NAMEBREAK_BACKENDS_CUDA_TUNING_H
#define NAMEBREAK_BACKENDS_CUDA_TUNING_H

// The CUDA backend's tuning values. The engine and the tests read them
// through SearchBackend (windowChars(), batchSize(), ...), not from here.

// How many trailing characters of a candidate the GPU enumerates directly
// (the rest - the "leading" part - is folded into the prefix on the CPU, one
// value per group of GPU launches). Overridable at compile time
// (-DNAMEBREAK_GPU_WINDOW_CHARS=N) so tests can force a different leading/
// trailing split and benchmarks can re-sweep it. See runSearch (search.cpp)
// and README.md's "Design decisions" for what it trades off.
#ifndef NAMEBREAK_GPU_WINDOW_CHARS
#define NAMEBREAK_GPU_WINDOW_CHARS 5
#endif

// How many *rows* one kernel launch covers at most (a row = every value of a
// candidate's last character, for one combination of its other trailing
// characters - see bruteForceKernel in cuda_backend.cu), so one launch covers
// at most NAMEBREAK_ROWS_PER_LAUNCH * alphabetSize candidates. Bounds how long
// a single launch can run (pause/abort are only polled between launches).
// Overridable at compile time (-DNAMEBREAK_ROWS_PER_LAUNCH=N) so tests can
// force many small launches and exercise the chunk boundaries with tiny ranges.
#ifndef NAMEBREAK_ROWS_PER_LAUNCH
#define NAMEBREAK_ROWS_PER_LAUNCH (1u << 23)
#endif

// Largest trailing (GPU-enumerated) length the kernel supports. A row index
// (alphabetSize^(trailingLen-1)) must fit in 32 bits: 50^5 < 2^32 <= 50^6.
constexpr int kMaxTrailingLen = 6;

// Threads per block. Must be >= MAX_ALPHABET_SIZE: bruteForceKernel fills its
// shared tables one entry per thread.
constexpr int kThreadsPerBlock = 256;

static_assert(NAMEBREAK_GPU_WINDOW_CHARS >= 1 && NAMEBREAK_GPU_WINDOW_CHARS <= kMaxTrailingLen,
              "NAMEBREAK_GPU_WINDOW_CHARS must be between 1 and kMaxTrailingLen");
static_assert(NAMEBREAK_ROWS_PER_LAUNCH >= 1, "NAMEBREAK_ROWS_PER_LAUNCH must be >= 1");
static_assert(NAMEBREAK_ROWS_PER_LAUNCH <= (1u << 30), "a launch's row count must stay well within 32 bits");

#endif // NAMEBREAK_BACKENDS_CUDA_TUNING_H
