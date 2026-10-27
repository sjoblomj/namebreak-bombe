#include "backends/backends.h"

#include <algorithm>
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

// One backend this build has: its name, how to make one (null, with `error`
// set, if it can't run on this machine) on its GPU numbered `device` (see
// GpuDevice) - which a backend without GPUs ignores - and, for a GPU
// backend, how to list its GPUs.
struct BackendEntry {
    const char* name;
    std::unique_ptr<SearchBackend> (*make)(const HardwareConfig& hardware, int device, std::string& error);
    std::vector<GpuDevice> (*devices)();
};

// setHardwareConfig's, which one thread (the Windows GUI's) may set while
// another (its search's) makes backends.
std::mutex g_hardwareMutex;
HardwareConfig g_hardware;

// The settings `devices`' GPU numbered `device` is made with, as
// gpuTuningFor says - on stderr, if its own are for another GPU.
[[maybe_unused]] GpuTuning tuningForDevice(const HardwareConfig& hardware, const std::vector<GpuDevice>& devices, int device) {
    for (const GpuDevice& gpu : devices) {
        if (gpu.index == device) {
            std::string ignored;
            const GpuTuning tuning = gpuTuningFor(hardware, gpu, &ignored);
            if (!ignored.empty())
                fprintf(stderr, "%s\n", ignored.c_str());
            return tuning;
        }
    }
    return hardware.tuning;
}

std::unique_ptr<SearchBackend> makeReference(const HardwareConfig&, int, std::string&) {
    return makeReferenceBackend();
}
#if defined(NAMEBREAK_WITH_CUDA) || defined(NAMEBREAK_WITH_HIP)
std::unique_ptr<SearchBackend> makeCuda(const HardwareConfig&, int device, std::string& error) {
    return makeCudaBackend(device, error);
}
#endif
#ifdef NAMEBREAK_WITH_METAL
std::unique_ptr<SearchBackend> makeMetal(const HardwareConfig& hardware, int device, std::string& error) {
    return makeMetalBackend(device, tuningForDevice(hardware, metalDevices(), device), error);
}
#endif
#ifdef NAMEBREAK_WITH_OPENCL
std::unique_ptr<SearchBackend> makeOpenCl(const HardwareConfig& hardware, int device, std::string& error) {
    return makeOpenClBackend(device, tuningForDevice(hardware, openClDevices(), device), error);
}
#endif
std::unique_ptr<SearchBackend> makeCpu(const HardwareConfig& hardware, int, std::string& error) {
    return makeCpuBackend(hardware.cpuThreads, error);
}

// Most preferred first.
const std::vector<BackendEntry>& backendEntries() {
    static const std::vector<BackendEntry> entries = {
#ifdef NAMEBREAK_WITH_CUDA
        {"cuda", makeCuda, cudaDevices},
#endif
#ifdef NAMEBREAK_WITH_HIP
        // The CUDA backend's source, compiled with HIP - see cuda_backend.h.
        {"hip", makeCuda, cudaDevices},
#endif
#ifdef NAMEBREAK_WITH_METAL
        {"metal", makeMetal, metalDevices},
#endif
#ifdef NAMEBREAK_WITH_OPENCL
        {"opencl", makeOpenCl, openClDevices},
#endif
        {"cpu", makeCpu, nullptr},
        {"reference", makeReference, nullptr},
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

void setHardwareConfig(const HardwareConfig& value) {
    std::lock_guard<std::mutex> lock(g_hardwareMutex);
    g_hardware = value;
}

HardwareConfig hardwareConfig() {
    std::lock_guard<std::mutex> lock(g_hardwareMutex);
    return g_hardware;
}

std::vector<GpuDevice> gpuDevices(const std::string& name) {
    for (const BackendEntry& entry : backendEntries()) {
        if (name == entry.name)
            return entry.devices ? entry.devices() : std::vector<GpuDevice>();
    }
    return {};
}

std::vector<GpuDevice> selectedGpus(const std::vector<GpuDevice>& devices, const HardwareConfig& hardware) {
    std::vector<GpuDevice> selected;
    switch (hardware.gpuChoice) {
    case HardwareConfig::GpuChoice::All:
        return devices;
    case HardwareConfig::GpuChoice::Listed:
        for (const GpuDevice& device : devices) {
            if (std::find(hardware.gpus.begin(), hardware.gpus.end(), device.index) != hardware.gpus.end())
                selected.push_back(device);
        }
        return selected;
    case HardwareConfig::GpuChoice::Auto:
        // An integrated GPU shares the CPU's power and memory - and the CPU
        // feeds the discrete GPUs - so it's only searched with on its own.
        for (const GpuDevice& device : devices) {
            if (!device.integrated)
                selected.push_back(device);
        }
        return selected.empty() ? devices : selected;
    }
    return selected;
}

std::string gpuNumberMismatch(const HardwareConfig& hardware, const GpuDevice& device) {
    if (hardware.gpuChoice != HardwareConfig::GpuChoice::Listed ||
        std::find(hardware.gpus.begin(), hardware.gpus.end(), device.index) == hardware.gpus.end())
        return "";
    const auto name = hardware.gpuNames.find(device.index);
    if (name == hardware.gpuNames.end() || name->second == device.name)
        return "";
    const std::string number = std::to_string(device.index);
    return "[hardware]'s gpus picks GPU " + number + ", which is " + device.name + " - but gpu" + number + ".name says GPU " + number +
           " was " + name->second + ". If the GPUs have been renumbered, namebreak --list-gpus shows their numbers now";
}

GpuTuning gpuTuningFor(const HardwareConfig& hardware, const GpuDevice& device, std::string* ignoredWhy) {
    const auto own = hardware.gpuTunings.find(device.index);
    if (own == hardware.gpuTunings.end())
        return hardware.tuning;
    const auto name = hardware.gpuNames.find(device.index);
    if (name != hardware.gpuNames.end() && name->second != device.name) {
        if (ignoredWhy)
            *ignoredWhy = "[hardware]'s gpu" + std::to_string(device.index) + " settings are for " + name->second + ", but GPU " +
                          std::to_string(device.index) + " is " + device.name + " - not using them";
        return hardware.tuning;
    }
    return withDefaults(own->second, hardware.tuning);
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

// The GPU `entry` searches on as `device` asks (see createBackend): -1 for
// the first of those `hardware` selects - warning on stderr of any it picks
// by a number that was another card's (gpuNumberMismatch). Also -1, with `why` set, if there's
// none to search on; any other backend's is 0.
int deviceFor(const BackendEntry& entry, const HardwareConfig& hardware, int device, std::string& why) {
    if (!entry.devices)
        return 0;
    if (device >= 0)
        return device;
    const std::vector<GpuDevice> devices = entry.devices();
    const std::vector<GpuDevice> selected = selectedGpus(devices, hardware);
    for (const GpuDevice& gpu : selected) {
        const std::string mismatch = gpuNumberMismatch(hardware, gpu);
        if (!mismatch.empty())
            fprintf(stderr, "%s\n", mismatch.c_str());
    }
    if (!selected.empty())
        return selected.front().index;
    why = devices.empty() ? "no GPU of its kind here" : "none of its GPUs is selected by [hardware]'s gpus";
    return -1;
}

std::unique_ptr<SearchBackend> create(const std::string& name, std::string& error, bool* selfTestFailed, Mode mode, const HardwareConfig* given,
                                      int device) {
    const HardwareConfig hardware = given ? *given : hardwareConfig();
    if (selfTestFailed)
        *selfTestFailed = false;
    if (name.empty()) {
        // The reference backend is always there, so something always succeeds
        // (unless even it fails its self-test).
        std::string reasons;
        for (const BackendEntry& entry : backendEntries()) {
            std::string why;
            const int on = deviceFor(entry, hardware, device, why);
            std::unique_ptr<SearchBackend> backend = on >= 0 ? entry.make(hardware, on, why) : nullptr;
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
            const int on = deviceFor(entry, hardware, device, why);
            std::unique_ptr<SearchBackend> backend = on >= 0 ? entry.make(hardware, on, why) : nullptr;
            if (!backend) {
                error = "the " + name + " backend can't run on this machine: " + why;
            } else if (!supportsMode(*backend, mode)) {
                error = "the " + name + " backend " + cannotSearch(mode);
                backend = nullptr;
            } else if (!passesSelfTests(*backend, mode, why)) {
                error = "the " + name + " backend failed its self-test on this machine" +
                        (entry.devices ? " (on " + backend->deviceName() + ")" : std::string()) + ": " + why;
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

std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error, bool* selfTestFailed, const HardwareConfig* hardware,
                                             int device) {
    return create(name, error, selfTestFailed, Mode::Plain, hardware, device);
}

std::unique_ptr<SearchBackend> createDictionaryBackend(const std::string& name, std::string& error, bool* selfTestFailed,
                                                       const HardwareConfig* hardware, int device) {
    return create(name, error, selfTestFailed, Mode::Dictionary, hardware, device);
}

std::unique_ptr<SearchBackend> createMeetInTheMiddleBackend(const std::string& name, std::string& error, bool* selfTestFailed,
                                                            const HardwareConfig* hardware, int device) {
    return create(name, error, selfTestFailed, Mode::MeetInTheMiddle, hardware, device);
}
