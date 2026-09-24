#ifndef NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H
#define NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H

#include <memory>
#include <string>

#include "engine/backend.h"

// Searches on the Mac's GPU (Apple Silicon, or an Intel Mac's) through
// Metal. Null, with `error` set, if there's no Metal device.
std::unique_ptr<SearchBackend> makeMetalBackend(std::string& error);

#endif // NAMEBREAK_BACKENDS_METAL_METAL_BACKEND_H
