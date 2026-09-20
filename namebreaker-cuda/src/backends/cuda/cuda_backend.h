#ifndef NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H
#define NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H

#include <memory>

#include "engine/backend.h"

// Searches on an NVIDIA GPU. Constructing one doesn't touch the GPU - that
// only happens once a search begins.
std::unique_ptr<SearchBackend> makeCudaBackend();

#endif // NAMEBREAK_BACKENDS_CUDA_CUDA_BACKEND_H
