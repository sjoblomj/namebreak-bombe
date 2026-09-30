#ifndef NAMEBREAK_BACKENDS_COMMON_ROW_BATCH_H
#define NAMEBREAK_BACKENDS_COMMON_ROW_BATCH_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/backend.h"

// Shared by the backends that search in *rows*, the way the CUDA one does
// (see the terminology comment in backends/cuda/cuda_backend.cu): a row is
// one combination of the trailing part's first trailingLen - 1 characters,
// holding one candidate per value of the last character, so row r's
// candidate with last character k has trailing index r * alphabetSize + k.
// Hashing a row's shared characters once, then only the last character and
// the suffix per candidate, is what makes them fast.

// The rows a batch's trailing indices [start, start + count) cover: rows
// firstRow .. firstRow + rowCount - 1, of which only the first and last can
// be partial - the first starts at last character firstRowStartK, the last
// ends (exclusively) at lastRowEndK.
struct RowRange {
    uint64_t firstRow;
    uint64_t rowCount;
    int firstRowStartK;
    int lastRowEndK; // in [1, alphabetSize]
};
RowRange rowRangeFor(uint64_t start, uint64_t count, int alphabetSize);

// A row-based backend's window (SearchBackend::windowChars) and rows per
// batch, unless the build overrides them for every row-based backend at once
// (-DNAMEBREAK_GPU_WINDOW_CHARS=N / -DNAMEBREAK_ROWS_PER_LAUNCH=N - the tests
// do, to exercise other geometries; see CMakeLists.txt).
constexpr int windowCharsOr([[maybe_unused]] int backendDefault) {
#ifdef NAMEBREAK_GPU_WINDOW_CHARS
    return NAMEBREAK_GPU_WINDOW_CHARS;
#else
    return backendDefault;
#endif
}
constexpr uint64_t rowsPerBatchOr([[maybe_unused]] uint64_t backendDefault) {
#ifdef NAMEBREAK_ROWS_PER_LAUNCH
    return NAMEBREAK_ROWS_PER_LAUNCH;
#else
    return backendDefault;
#endif
}
// Likewise how many rows of a row group one GPU thread searches (see
// kChunksPerGroup in backends/cuda/cuda_backend.cu), for the backends that
// split row groups into chunks (-DNAMEBREAK_ROWS_PER_THREAD=N).
constexpr int rowsPerThreadOr([[maybe_unused]] int backendDefault) {
#ifdef NAMEBREAK_ROWS_PER_THREAD
    return NAMEBREAK_ROWS_PER_THREAD;
#else
    return backendDefault;
#endif
}
// And how many batches one launch searches at most
// (SearchBackend::maxBatchesPerCall), for the backends that search several at
// once (-DNAMEBREAK_BATCHES_PER_LAUNCH=N).
constexpr int batchesPerLaunchOr([[maybe_unused]] int backendDefault) {
#ifdef NAMEBREAK_BATCHES_PER_LAUNCH
    return NAMEBREAK_BATCHES_PER_LAUNCH;
#else
    return backendDefault;
#endif
}

// Turns the trailing indices of a batch's hashA hits into what runBatch
// returns: rebuilds each hit's complete filename on the host and checks it
// against hashB. Also hashes every filename from scratch, as a cross-check
// of the backend's own hashing - independent code, on the CPU - printing a
// WARNING (which the tests treat as a failure) if the two disagree. Every
// backend but `reference` checks its hits with it.
class HitVerifier {
public:
    void begin(const SearchConstants& constants);
    // Adds the hits at `trailingIndices` (at most MAX_MATCHES of them) to
    // `outcome` - hitCount is left for the caller.
    void addHits(const std::vector<uint64_t>& trailingIndices, int trailingLen, const BatchParams& params, BatchOutcome& outcome) const;

private:
    uint32_t hashFromScratch(const std::string& filename, int tableOffset) const;

    std::string alphabet_;
    std::string suffix_;
    std::vector<uint32_t> cryptTable_;
    uint32_t targetA_ = 0;
    uint32_t targetB_ = 0;
};

#endif // NAMEBREAK_BACKENDS_COMMON_ROW_BATCH_H
