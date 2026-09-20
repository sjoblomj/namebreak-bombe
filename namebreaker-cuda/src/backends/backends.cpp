#include "backends/backends.h"

#ifdef NAMEBREAK_WITH_CUDA
#include "backends/cuda/cuda_backend.h"
#endif

std::vector<std::string> availableBackends() {
    std::vector<std::string> names;
#ifdef NAMEBREAK_WITH_CUDA
    names.push_back("cuda");
#endif
    return names;
}

std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error) {
#ifdef NAMEBREAK_WITH_CUDA
    if (name == "cuda")
        return makeCudaBackend();
#endif
    error = "this build has no '" + name + "' backend";
    return nullptr;
}

std::unique_ptr<SearchBackend> createDefaultBackend() {
    std::string error;
    return createBackend(availableBackends().front(), error);
}
