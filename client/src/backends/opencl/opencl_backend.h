#ifndef NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H
#define NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H

#include <memory>
#include <string>

#include "engine/backend.h"

// Searches on the first GPU any OpenCL platform offers - AMD, Intel or
// NVIDIA, with nothing but the vendor's regular driver installed. Null, with
// `error` set, if there's no such GPU.
std::unique_ptr<SearchBackend> makeOpenClBackend(std::string& error);

#endif // NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H
