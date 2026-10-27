#include "backends/backends.h"

#include <cstdio>
#include <mutex>

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
    std::unique_ptr<SearchBackend> (*make)(const BackendTunings& tunings, std::string& error);
};

// setBackendTunings's, which one thread (the Windows GUI's) may set while
// another (its search's) makes backends.
std::mutex g_tuningsMutex;
BackendTunings g_tunings;

std::unique_ptr<SearchBackend> makeReference(const BackendTunings&, std::string&) {
    return makeReferenceBackend();
}
#if defined(NAMEBREAK_WITH_CUDA) || defined(NAMEBREAK_WITH_HIP)
std::unique_ptr<SearchBackend> makeCuda(const BackendTunings&, std::string& error) {
    return makeCudaBackend(error);
}
#endif
#ifdef NAMEBREAK_WITH_METAL
std::unique_ptr<SearchBackend> makeMetal(const BackendTunings& tunings, std::string& error) {
    return makeMetalBackend(tunings.metal, error);
}
#endif
#ifdef NAMEBREAK_WITH_OPENCL
std::unique_ptr<SearchBackend> makeOpenCl(const BackendTunings& tunings, std::string& error) {
    return makeOpenClBackend(tunings.opencl, error);
}
#endif
std::unique_ptr<SearchBackend> makeCpu(const BackendTunings&, std::string& error) {
    return makeCpuBackend(error);
}

// Most preferred first.
const std::vector<BackendEntry>& backendEntries() {
    static const std::vector<BackendEntry> entries = {
#ifdef NAMEBREAK_WITH_CUDA
        {"cuda", makeCuda},
#endif
#ifdef NAMEBREAK_WITH_HIP
        // The CUDA backend's source, compiled with HIP - see cuda_backend.h.
        {"hip", makeCuda},
#endif
#ifdef NAMEBREAK_WITH_METAL
        {"metal", makeMetal},
#endif
#ifdef NAMEBREAK_WITH_OPENCL
        {"opencl", makeOpenCl},
#endif
        {"cpu", makeCpu},
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

void setBackendTunings(const BackendTunings& value) {
    std::lock_guard<std::mutex> lock(g_tuningsMutex);
    g_tunings = value;
}

BackendTunings backendTunings() {
    std::lock_guard<std::mutex> lock(g_tuningsMutex);
    return g_tunings;
}

GpuTuning defaultGpuTuning([[maybe_unused]] const std::string& name) {
#ifdef NAMEBREAK_WITH_METAL
    if (name == "metal")
        return metalTuningDefaults();
#endif
#ifdef NAMEBREAK_WITH_OPENCL
    if (name == "opencl")
        return openClTuningDefaults();
#endif
    return GpuTuning();
}

namespace {

// Which mode a backend is being picked for - each has its own self-test, and
// its own question of whether a backend can search that way at all.
enum class Mode { Plain, Dictionary, MeetInTheMiddle };

bool supportsMode(const SearchBackend& backend, Mode mode) {
    switch (mode) {
    case Mode::Dictionary:
        return backend.supportsDictionary();
    case Mode::MeetInTheMiddle:
        return backend.supportsMeetInTheMiddle();
    default:
        return true;
    }
}

const char* cannotSearch(Mode mode) {
    return mode == Mode::Dictionary ? "can't search dictionaries" : "can't search meet-in-the-middle targets";
}

// Whether `backend` passes its self-tests - the mode's own too. If not,
// `why` says why.
bool passesSelfTests(SearchBackend& backend, Mode mode, std::string& why) {
    if (!selfTestBackend(backend, why))
        return false;
    if (mode == Mode::Dictionary && !selfTestDictionaryBackend(backend, why)) {
        why = "dictionary search: " + why;
        return false;
    }
    if (mode == Mode::MeetInTheMiddle && !selfTestMeetInTheMiddleBackend(backend, why))
        return false;
    return true;
}

std::unique_ptr<SearchBackend> create(const std::string& name, std::string& error, bool* selfTestFailed, Mode mode, const BackendTunings* given) {
    const BackendTunings tunings = given ? *given : backendTunings();
    if (selfTestFailed)
        *selfTestFailed = false;
    if (name.empty()) {
        // The reference backend is always there, so something always succeeds
        // (unless even it fails its self-test).
        std::string reasons;
        for (const BackendEntry& entry : backendEntries()) {
            std::string why;
            std::unique_ptr<SearchBackend> backend = entry.make(tunings, why);
            if (backend && !supportsMode(*backend, mode)) {
                // Not one to pick for this mode - and not one that failed
                // anything either.
                reasons += std::string(reasons.empty() ? "" : "; ") + entry.name + ": " + cannotSearch(mode);
                continue;
            }
            if (backend && passesSelfTests(*backend, mode, why))
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
            std::unique_ptr<SearchBackend> backend = entry.make(tunings, why);
            if (!backend) {
                error = "the " + name + " backend can't run on this machine: " + why;
            } else if (!supportsMode(*backend, mode)) {
                error = "the " + name + " backend " + cannotSearch(mode);
                backend = nullptr;
            } else if (!passesSelfTests(*backend, mode, why)) {
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

} // namespace

std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error, bool* selfTestFailed, const BackendTunings* tunings) {
    return create(name, error, selfTestFailed, Mode::Plain, tunings);
}

std::unique_ptr<SearchBackend> createDictionaryBackend(const std::string& name, std::string& error, bool* selfTestFailed,
                                                       const BackendTunings* tunings) {
    return create(name, error, selfTestFailed, Mode::Dictionary, tunings);
}

std::unique_ptr<SearchBackend> createMeetInTheMiddleBackend(const std::string& name, std::string& error, bool* selfTestFailed,
                                                            const BackendTunings* tunings) {
    return create(name, error, selfTestFailed, Mode::MeetInTheMiddle, tunings);
}
