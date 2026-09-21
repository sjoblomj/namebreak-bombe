#ifndef NAMEBREAK_BACKENDS_REFERENCE_REFERENCE_BACKEND_H
#define NAMEBREAK_BACKENDS_REFERENCE_REFERENCE_BACKEND_H

#include <memory>

#include "engine/backend.h"

// Searches on the CPU, one candidate at a time on one thread - the reference
// every other backend is tested against (see tests/search_integration_test.cpp).
// Written to be obviously right, not fast: it's orders of magnitude slower
// than even the multithreaded `cpu` backend. Always available.
std::unique_ptr<SearchBackend> makeReferenceBackend();

#endif // NAMEBREAK_BACKENDS_REFERENCE_REFERENCE_BACKEND_H
