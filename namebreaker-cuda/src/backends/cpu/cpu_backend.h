#ifndef NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H
#define NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H

#include <memory>
#include <string>

#include "engine/backend.h"

// Searches on the CPU, on every core, with the same row trick as the CUDA
// backend and the last character's candidates hashed several at a time with
// SIMD instructions. Always available - a GPU is one or two orders of
// magnitude faster, but this lets any machine contribute.
std::unique_ptr<SearchBackend> makeCpuBackend(std::string& error);

#endif // NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H
