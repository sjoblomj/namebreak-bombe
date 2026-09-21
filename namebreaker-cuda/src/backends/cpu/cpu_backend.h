#ifndef NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H
#define NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H

#include <memory>

#include "engine/backend.h"

// Searches on the CPU, one candidate at a time on one thread - the reference
// every other backend is tested against (see tests/search_integration_test.cpp),
// and what `make BACKEND=cpu` builds with. Written to be obviously right, not
// fast: it's orders of magnitude slower than a GPU.
std::unique_ptr<SearchBackend> makeCpuBackend();

#endif // NAMEBREAK_BACKENDS_CPU_CPU_BACKEND_H
