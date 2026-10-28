#ifndef NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H
#define NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H

#include <memory>
#include <string>

#include "engine/backend.h"

// Searches on the CPU, with `threads` threads (0 for one per hardware
// thread - every core), with the same row trick and lookup filter as the
// CUDA backend: on an x86-64 CPU with AVX2, eight rows' states and table
// entries at a time, and the candidates the filter lets through hashed eight
// at a time. Always available - a GPU is one or two orders of magnitude
// faster, but this lets any machine contribute.
std::unique_ptr<SearchBackend> makeCpuBackend(int threads, std::string& error);

#endif // NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H
