#include "backends/backends.h"

#include <cstdio>

#include "backends/cpu/cpu_backend.h"
#include "backends/reference/reference_backend.h"
#include "backends/self_test.h"
#if defined(NAMEBREAK_WITH_CUDA) || defined(NAMEBREAK_WITH_HIP)
#include "backends/cuda/cuda_backend.h"
#endif
#ifdef NAMEBREAK_WITH_METAL
#include "backends/metal/metal_backend.h"
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
#ifdef NAMEBREAK_WITH_METAL
        {"metal", makeMetalBackend},
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

std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error, bool* selfTestFailed) {
    if (selfTestFailed)
        *selfTestFailed = false;
    if (name.empty()) {
        // The reference backend is always there, so something always succeeds
        // (unless even it fails its self-test).
        std::string reasons;
        for (const BackendEntry& entry : backendEntries()) {
            std::string why;
            std::unique_ptr<SearchBackend> backend = entry.make(why);
            if (backend && selfTestBackend(*backend, why))
                return backend;
            if (backend) {
                // It could run, but searches wrongly here: say so, rather than
                // quietly settling for a slower backend.
                fprintf(stderr, "The %s backend failed its self-test on this machine, so it won't be used: %s\n", entry.name, why.c_str());
                why = "failed its self-test: " + why;
            }
            reasons += std::string(reasons.empty() ? "" : "; ") + entry.name + ": " + why;
        }
        error = "no backend can run on this machine (" + reasons + ")";
        return nullptr;
    }
    for (const BackendEntry& entry : backendEntries()) {
        if (name == entry.name) {
            std::string why;
            std::unique_ptr<SearchBackend> backend = entry.make(why);
            if (!backend) {
                error = "the " + name + " backend can't run on this machine: " + why;
            } else if (!selfTestBackend(*backend, why)) {
                error = "the " + name + " backend failed its self-test on this machine: " + why;
                if (selfTestFailed)
                    *selfTestFailed = true;
                backend = nullptr;
            }
            return backend;
        }
    }
    std::string names;
    for (const BackendEntry& entry : backendEntries())
        names += std::string(names.empty() ? "" : ", ") + entry.name;
    error = "this build has no '" + name + "' backend (it has: " + names + ")";
    return nullptr;
}
