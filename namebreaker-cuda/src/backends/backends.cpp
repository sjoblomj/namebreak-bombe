#include "backends/backends.h"

#include "backends/cpu/cpu_backend.h"
#include "backends/reference/reference_backend.h"
#if defined(NAMEBREAK_WITH_CUDA) || defined(NAMEBREAK_WITH_HIP)
#include "backends/cuda/cuda_backend.h"
#endif
#ifdef NAMEBREAK_WITH_OPENCL
#include "backends/opencl/opencl_backend.h"
#endif

namespace {

// One backend this build has: its name, and how to make one (null, with
// `error` set, if it can't run on this machine).
struct BackendEntry {
    const char* name;
    std::unique_ptr<SearchBackend> (*make)(std::string& error);
};

std::unique_ptr<SearchBackend> makeReference(std::string&) {
    return makeReferenceBackend();
}

// Most preferred first.
const std::vector<BackendEntry>& backendEntries() {
    static const std::vector<BackendEntry> entries = {
#ifdef NAMEBREAK_WITH_CUDA
        {"cuda", makeCudaBackend},
#endif
#ifdef NAMEBREAK_WITH_HIP
        // The CUDA backend's source, compiled with HIP - see cuda_backend.h.
        {"hip", makeCudaBackend},
#endif
#ifdef NAMEBREAK_WITH_OPENCL
        {"opencl", makeOpenClBackend},
#endif
        {"cpu", makeCpuBackend},
        {"reference", makeReference},
    };
    return entries;
}

} // namespace

std::vector<std::string> backendNames() {
    std::vector<std::string> names;
    for (const BackendEntry& entry : backendEntries())
        names.push_back(entry.name);
    return names;
}

std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error) {
    if (name.empty()) {
        // The reference backend is always there, so something always succeeds.
        std::string reasons;
        for (const BackendEntry& entry : backendEntries()) {
            std::string why;
            if (std::unique_ptr<SearchBackend> backend = entry.make(why))
                return backend;
            reasons += std::string(reasons.empty() ? "" : "; ") + entry.name + ": " + why;
        }
        error = "no backend can run on this machine (" + reasons + ")";
        return nullptr;
    }
    for (const BackendEntry& entry : backendEntries()) {
        if (name == entry.name) {
            std::string why;
            std::unique_ptr<SearchBackend> backend = entry.make(why);
            if (!backend)
                error = "the " + name + " backend can't run on this machine: " + why;
            return backend;
        }
    }
    std::string names;
    for (const BackendEntry& entry : backendEntries())
        names += std::string(names.empty() ? "" : ", ") + entry.name;
    error = "this build has no '" + name + "' backend (it has: " + names + ")";
    return nullptr;
}
