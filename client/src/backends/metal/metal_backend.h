#ifndef NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H
#define NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "backends/gpu_device.h"
#include "backends/gpu_tuning.h"
#include "engine/backend.h"

// Searches on the Mac's GPU numbered `device` in metalDevices() (Apple
// Silicon's, or one of an Intel Mac's) through Metal, with the same lookup
// filter (backends/common/lowbits_filter.h) and row groups as the CUDA and
// OpenCL backends, launched as `tuning` says (metalTuningDefaults() where it
// says nothing). Null, with `error` set, if there's no such Metal device.
std::unique_ptr<SearchBackend> makeMetalBackend(int device, const GpuTuning& tuning, std::string& error);

// The Mac's Metal devices (MTLCopyAllDevices), the system's default one -
// the one the display uses - first. An Intel Mac's integrated GPU is the
// low-power one; Apple Silicon's is integrated too, and its only one.
std::vector<GpuDevice> metalDevices();

// What the backend uses for any value a GpuTuning leaves unset - every one
// it takes set.
GpuTuning metalTuningDefaults();

#endif // NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H
