#include "backends/gpu_tuning.h"

#include "engine/limits.h"

const std::vector<GpuTuningKey>& gpuTuningKeys(const std::string& backend) {
    // A work-group fills its row search's tables one entry per work-item, so
    // it has at least as many as the largest alphabet. A row index is 32-bit
    // in both kernels: 63^5 < 2^32 <= 63^6. A launch's batches go in a buffer
    // of 32 (kMaxBatchesPerLaunch). More rows per work-item than an alphabet
    // has is a whole row group each, as 64 is.
    static const std::vector<GpuTuningKey> opencl = {
        {"rows_per_thread", &GpuTuning::rowsPerThread, 1, 64},
        {"batches_per_launch", &GpuTuning::batchesPerLaunch, 1, 32},
        {"window_chars", &GpuTuning::windowChars, 1, 6},
        {"work_group_size", &GpuTuning::workGroupSize, MAX_ALPHABET_SIZE + 1, 1024},
        {"dictionary_words_per_thread", &GpuTuning::dictionaryWordsPerThread, 1, 256},
        {"dictionary_work_group_size", &GpuTuning::dictionaryWorkGroupSize, 1, 1024},
        {"dictionary_tails_per_cell", &GpuTuning::dictionaryTailsPerCell, 1, 256},
    };
    static const std::vector<GpuTuningKey> metal = {
        {"rows_per_thread", &GpuTuning::rowsPerThread, 1, 64},
        {"window_chars", &GpuTuning::windowChars, 1, 6},
        {"work_group_size", &GpuTuning::workGroupSize, MAX_ALPHABET_SIZE + 1, 1024},
        {"dictionary_words_per_thread", &GpuTuning::dictionaryWordsPerThread, 1, 256},
        {"dictionary_work_group_size", &GpuTuning::dictionaryWorkGroupSize, 1, 1024},
        {"dictionary_tails_per_cell", &GpuTuning::dictionaryTailsPerCell, 1, 256},
    };
    static const std::vector<GpuTuningKey> none;
    return backend == "opencl" ? opencl : backend == "metal" ? metal : none;
}

GpuTuning withDefaults(const GpuTuning& tuning, const GpuTuning& defaults) {
    GpuTuning out = tuning;
    for (std::optional<int> GpuTuning::* field :
         {&GpuTuning::rowsPerThread, &GpuTuning::batchesPerLaunch, &GpuTuning::windowChars, &GpuTuning::workGroupSize,
          &GpuTuning::dictionaryWordsPerThread, &GpuTuning::dictionaryWorkGroupSize, &GpuTuning::dictionaryTailsPerCell}) {
        if (!(out.*field))
            out.*field = defaults.*field;
    }
    return out;
}

std::string describeGpuTuning(const std::string& backend, const GpuTuning& tuning) {
    std::string out;
    for (const GpuTuningKey& key : gpuTuningKeys(backend)) {
        if (const std::optional<int>& value = tuning.*key.field)
            out += std::string(out.empty() ? "" : ", ") + key.name + " " + std::to_string(*value);
    }
    return out;
}
