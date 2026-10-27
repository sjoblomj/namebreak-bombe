#ifndef NAMEBREAK_BACKENDS_GPU_TUNING_H
#define NAMEBREAK_BACKENDS_GPU_TUNING_H

#include <optional>
#include <string>
#include <vector>

// The launch geometry of a GPU backend that compiles its kernels at runtime
// (OpenCL, Metal), which config.conf's [opencl] and [metal] sections can set
// (see buildBackendTunings, common/config.h) and `namebreak --autotune`
// measures (common/autotune.h). None of it changes what a search finds, only
// how its work is cut up - every backend still has to pass its self-tests
// with it. The CUDA backend compiles its kernels ahead of time, with these
// as template arguments and launch bounds (see its tuning.h), so it takes
// none of them.
//
// Unset is the backend's own default: what it was measured best with on
// the RTX 3080 Ti Laptop, or what the build's -DNAMEBREAK_... overrides say
// (backends/common/row_batch.h, dictionary_batch.h).
struct GpuTuning {
    // A row search's: about how many rows of a row group one work-item
    // searches (ROWS_PER_THREAD in search.cl and search.metal)...
    std::optional<int> rowsPerThread;
    // ...how many batches one launch searches at most
    // (SearchBackend::maxBatchesPerCall - OpenCL only: Metal launches one)...
    std::optional<int> batchesPerLaunch;
    // ...how many trailing characters the GPU enumerates
    // (SearchBackend::windowChars)...
    std::optional<int> windowChars;
    // ...and its work-items per work-group (threadgroup), unless the device
    // allows fewer.
    std::optional<int> workGroupSize;
    // A dictionary search's: cells per work-item, work-items per work-group,
    // and tails per cell (see DictionaryLaunchBatch,
    // backends/common/dictionary_batch.h).
    std::optional<int> dictionaryWordsPerThread;
    std::optional<int> dictionaryWorkGroupSize;
    std::optional<int> dictionaryTailsPerCell;
};

// One tuning value: its config key, the GpuTuning field it sets, and the
// range it must be in.
struct GpuTuningKey {
    const char* name;
    std::optional<int> GpuTuning::* field;
    int min;
    int max;
};

// The keys `backend` ("opencl" or "metal") takes, in the order the config
// and --autotune list them. Empty for any other backend.
const std::vector<GpuTuningKey>& gpuTuningKeys(const std::string& backend);

// Whether `a` and `b` set the same values, and leave the same ones unset.
bool operator==(const GpuTuning& a, const GpuTuning& b);
inline bool operator!=(const GpuTuning& a, const GpuTuning& b) {
    return !(a == b);
}

// `tuning` with every value it leaves unset taken from `defaults`.
GpuTuning withDefaults(const GpuTuning& tuning, const GpuTuning& defaults);

// The values `tuning` sets, as "key value, key value" - for the backends'
// start-up line.
std::string describeGpuTuning(const std::string& backend, const GpuTuning& tuning);

#endif // NAMEBREAK_BACKENDS_GPU_TUNING_H
