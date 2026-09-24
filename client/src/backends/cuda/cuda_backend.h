#ifndef NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H
#define NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H

#include <memory>

#include "engine/backend.h"

#include <string>

// Searches on an NVIDIA GPU. Null, with `error` set, if there's no CUDA
// device (or driver) to search on.
//
// Compiled with HIP instead (NAMEBREAK_GPU=hip), the same code is the `hip`
// backend, searching on an AMD GPU - backends.cpp registers this function
// under that name then.
std::unique_ptr<SearchBackend> makeCudaBackend(std::string& error);

#endif // NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H
