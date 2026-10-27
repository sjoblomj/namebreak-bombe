#ifndef NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H
#define NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "backends/gpu_device.h"
#include "engine/backend.h"

// Searches on the NVIDIA GPU numbered `device` in cudaDevices(). Null, with
// `error` set, if there's no such CUDA device (or no driver).
//
// Compiled with HIP instead (NAMEBREAK_GPU=hip), the same code is the `hip`
// backend, searching on an AMD GPU - backends.cpp registers these functions
// under that name then.
std::unique_ptr<SearchBackend> makeCudaBackend(int device, std::string& error);

// The CUDA devices, numbered in PCI order (as nvidia-smi numbers them) rather
// than the runtime's fastest first, so that a number stays the same card
// until cards are added, taken out or moved - whatever the driver thinks is
// fastest. Of those CUDA_VISIBLE_DEVICES leaves. Empty if there's no device,
// or no driver.
std::vector<GpuDevice> cudaDevices();

#endif // NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H
