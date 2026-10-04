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
// characters - see filteredRowsKernel in cuda_backend.cu), so one launch covers
// at most NAMEBREAK_ROWS_PER_LAUNCH * alphabetSize candidates. Bounds how long
// a single launch can run (pause/abort are only polled between launches).
// Overridable at compile time (-DNAMEBREAK_ROWS_PER_LAUNCH=N) so tests can
// force many small launches and exercise the chunk boundaries with tiny ranges.
#ifndef NAMEBREAK_ROWS_PER_LAUNCH
#define NAMEBREAK_ROWS_PER_LAUNCH (1u << 23)
#endif

// How many batches - each up to NAMEBREAK_ROWS_PER_LAUNCH rows of one leading
// value - one kernel launch searches at most (SearchBackend::maxBatchesPerCall).
// A batch of the usual 5-character window is one leading value, 49^4 rows,
// about 0.13 ms of GPU time, and the GPU idles about 10 us between two
// launches; several batches per launch make that a smaller share, without
// the window growing (a wider window would leave the CPU less to prune). On
// the RTX 3080 Ti Laptop, 16 took the GPU's idle time from 9.4% to 1.6%, and
// search_bench from 1 to 16 gained about 6.7% (six alternating pairs); 4, 8,
// 16 and 32 were within the noise of each other. Overridable at compile time (-DNAMEBREAK_BATCHES_PER_LAUNCH=N) so the tests
// can exercise other groupings, and benchmarks re-sweep it.
#ifndef NAMEBREAK_BATCHES_PER_LAUNCH
#define NAMEBREAK_BATCHES_PER_LAUNCH 16
#endif
// The most NAMEBREAK_BATCHES_PER_LAUNCH can be: a launch's batches are passed
// to the kernel as an argument, by value.
constexpr int kMaxBatchesPerLaunch = 32;

// About how many consecutive rows one thread searches (see kChunksPerGroup
// in cuda_backend.cu): it hashes the characters they share once, then one
// step per row. More rows per thread means less of that, but fewer threads
// per launch. On the RTX 3080 Ti Laptop (search_bench --scale 20, three runs
// each), 1 row per thread did 1,133 G candidates/s, 4 did 1,477, 7 did 1,575,
// 10-16 about 1,605, 25 did 1,659 and 49 did 1,611 - 25, i.e. two chunks per
// row group for the real 49-character alphabet, was best in every run. Since
// the kernel hashes a chunk's flagged candidates after all of its rows
// (searchChunk), longer chunks share those out better among a warp's lanes:
// a whole row group per thread (64, more than any alphabet's size) measured
// 2-6% faster than 25 at alphabet sizes 40, 43 and 49, with 4- and
// 11-character suffixes alike, and 15 about 6-9% slower.
// Overridable at compile time (-DNAMEBREAK_ROWS_PER_THREAD=N) so the tests can
// exercise other splits, and benchmarks re-sweep it.
#ifndef NAMEBREAK_ROWS_PER_THREAD
#define NAMEBREAK_ROWS_PER_THREAD 64
#endif

// A search that prunes the whole candidate gets, for each batch, the list of
// its row groups that survive the rules (backends/common/row_pruning.h). A
// launch walks those lists only if they leave out at least this percentage
// of its batches' row groups; if not, it walks every group, as a search that
// doesn't prune the whole candidate does, and the hits the lists would have
// left out are dropped on the host (runBatches) - so what a search reports
// is the same either way. Walking a list costs about 5% more per row group
// (Nsight Compute: 1.60 ms instead of 1.53 for the same launch), more than
// the groups some rules leave out: the alphabet " -.0-9A-Z_" with symbol
// runs pruned loses about 0.3%. 0 always walks the lists, 101 never does.
// Overridable at compile time (-DNAMEBREAK_LIST_MIN_PRUNED_PERCENT=N) so the
// tests can exercise both ways.
#ifndef NAMEBREAK_LIST_MIN_PRUNED_PERCENT
#define NAMEBREAK_LIST_MIN_PRUNED_PERCENT 5
#endif

// Largest trailing (GPU-enumerated) length the kernel supports. A row index
// (alphabetSize^(trailingLen-1)) must fit in 32 bits: 63^5 < 2^32 <= 63^6.
constexpr int kMaxTrailingLen = 6;

// Threads per block. Must be >= MAX_ALPHABET_SIZE: filteredRowsKernel fills its
// shared tables one entry per thread.
constexpr int kThreadsPerBlock = 256;

static_assert(NAMEBREAK_GPU_WINDOW_CHARS >= 1 && NAMEBREAK_GPU_WINDOW_CHARS <= kMaxTrailingLen,
              "NAMEBREAK_GPU_WINDOW_CHARS must be between 1 and kMaxTrailingLen");
static_assert(NAMEBREAK_ROWS_PER_LAUNCH >= 1, "NAMEBREAK_ROWS_PER_LAUNCH must be >= 1");
static_assert(NAMEBREAK_BATCHES_PER_LAUNCH >= 1 && NAMEBREAK_BATCHES_PER_LAUNCH <= kMaxBatchesPerLaunch,
              "NAMEBREAK_BATCHES_PER_LAUNCH must be between 1 and kMaxBatchesPerLaunch");
static_assert(NAMEBREAK_ROWS_PER_THREAD >= 1, "NAMEBREAK_ROWS_PER_THREAD must be >= 1");
static_assert(NAMEBREAK_LIST_MIN_PRUNED_PERCENT >= 0 && NAMEBREAK_LIST_MIN_PRUNED_PERCENT <= 101,
              "NAMEBREAK_LIST_MIN_PRUNED_PERCENT must be between 0 (always walk the lists) and 101 (never)");
static_assert(NAMEBREAK_ROWS_PER_LAUNCH <= (1u << 30), "a launch's row count must stay well within 32 bits");

#endif // NAMEBREAK_BACKENDS_CUDA_TUNING_H
