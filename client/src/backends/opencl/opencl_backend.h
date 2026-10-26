#ifndef NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H
#define NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H

#include <memory>
#include <string>

#include "backends/gpu_tuning.h"
#include "engine/backend.h"

// Searches on the first GPU any OpenCL platform offers - AMD, Intel or
// NVIDIA, with nothing but the vendor's regular driver installed - with the
// same lookup filter as the CUDA backend (backends/common/lowbits_filter.h),
// launched as `tuning` says (openClTuningDefaults() where it says nothing).
// Null, with `error` set, if there's no such GPU.
std::unique_ptr<SearchBackend> makeOpenClBackend(const GpuTuning& tuning, std::string& error);

// What the backend uses for any value a GpuTuning leaves unset - every one
// set.
GpuTuning openClTuningDefaults();

#endif // NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H
