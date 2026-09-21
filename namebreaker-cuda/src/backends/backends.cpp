#include "backends/backends.h"

#include "backends/cpu/cpu_backend.h"
#ifdef NAMEBREAK_WITH_CUDA
#include "backends/cuda/cuda_backend.h"
#endif

// The CPU backend is in every build - as the reference the tests hold the
// others to - but last, so it's only the default when there's nothing else.
std::vector<std::string> availableBackends() {
    std::vector<std::string> names;
#ifdef NAMEBREAK_WITH_CUDA
    names.push_back("cuda");
#endif
    names.push_back("cpu");
    return names;
}

std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error) {
#ifdef NAMEBREAK_WITH_CUDA
    if (name == "cuda")
        return makeCudaBackend();
#endif
    if (name == "cpu")
        return makeCpuBackend();
    error = "this build has no '" + name + "' backend";
    return nullptr;
}

std::unique_ptr<SearchBackend> createDefaultBackend() {
    std::string error;
    return createBackend(availableBackends().front(), error);
}
