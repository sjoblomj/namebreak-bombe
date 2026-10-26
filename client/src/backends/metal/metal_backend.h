#ifndef NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H
#define NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H

#include <memory>
#include <string>

#include "backends/gpu_tuning.h"
#include "engine/backend.h"

// Searches on the Mac's GPU (Apple Silicon, or an Intel Mac's) through
// Metal, with the same lookup filter (backends/common/lowbits_filter.h) and
// row groups as the CUDA and OpenCL backends, launched as `tuning` says
// (metalTuningDefaults() where it says nothing). Null, with `error` set, if
// there's no Metal device.
std::unique_ptr<SearchBackend> makeMetalBackend(const GpuTuning& tuning, std::string& error);

// What the backend uses for any value a GpuTuning leaves unset - every one
// it takes set.
GpuTuning metalTuningDefaults();

#endif // NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H
