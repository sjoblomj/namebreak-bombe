#ifndef NAMEBREAK_BACKENDS_COMMON_JUNCTION_LAUNCH_H
#define NAMEBREAK_BACKENDS_COMMON_JUNCTION_LAUNCH_H

#include <cstdint>

#include "engine/backend.h"
#include "engine/junction.h"

// What the backends that launch a junction kernel (CUDA/HIP and OpenCL)
// share: how big a launch is, and how a batch's tails split into the rows
// its work-items take.

// About how many tails one launch rewinds
// (SearchBackend::junctionTailsPerCall). A tail costs one step of the hash
// undone and one lookup in the table, so a GPU does billions a second: this
// is a few milliseconds' work, enough to amortize a launch and short enough
// for pause and abort to feel immediate.
#ifndef NAMEBREAK_JUNCTION_TAILS_PER_LAUNCH
#define NAMEBREAK_JUNCTION_TAILS_PER_LAUNCH (1ull << 26)
#endif
// How many of a launch's hits it has room for. A launch that has more is
// launched again with room for them all, so this only bounds a buffer, not
// correctness - the junction search tests build with 1, so that a single hit
// overflows it (see CMakeLists.txt).
#ifndef NAMEBREAK_JUNCTION_HIT_CAPACITY
#define NAMEBREAK_JUNCTION_HIT_CAPACITY 4096
#endif
static_assert(NAMEBREAK_JUNCTION_HIT_CAPACITY >= 1, "NAMEBREAK_JUNCTION_HIT_CAPACITY must be at least 1");

// A batch's rows. A row is the `alphabetSize` tails that share every
// character but the one rewound last - the first character of the tail,
// which is the directory's own - so one work-item takes a whole row, and
// only the batch's first and last row can be partial: the first starts at
// firstRowStartK, the last ends before lastRowEndK. The tail of no
// characters is one row of one tail.
struct JunctionRows {
    uint64_t firstRow;
    uint64_t lastRow;
    uint32_t firstRowStartK;
    uint32_t lastRowEndK;
    uint64_t rowCount() const { return lastRow - firstRow + 1; }
};

inline JunctionRows junctionRowsOf(const JunctionBatch& batch, uint64_t alphabetSize) {
    const uint64_t last = batch.start + batch.count - 1;
    if (batch.tailLength == 0)
        return {0, 0, 0, 1};
    return {batch.start / alphabetSize, last / alphabetSize, (uint32_t) (batch.start % alphabetSize),
            (uint32_t) (last % alphabetSize) + 1};
}

#endif // NAMEBREAK_BACKENDS_COMMON_JUNCTION_LAUNCH_H
