#ifndef NAMEBREAK_BACKENDS_BACKENDS_H
#define NAMEBREAK_BACKENDS_BACKENDS_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backends/gpu_device.h"
#include "backends/gpu_tuning.h"
#include "engine/backend.h"

// The search backends compiled into this build (see NAMEBREAK_GPU etc. in
// CMakeLists.txt), and picking one of them.

// The names of the backends in this build, most preferred first: GPU
// backends, then `cpu`, then `reference` (the slow one the others are tested
// against, which is always there).
std::vector<std::string> backendNames();

// config.conf's [hardware] section (see buildHardwareConfig, config.h): which
// of this machine's GPUs to search with, how many threads the cpu backend
// uses, and the OpenCL and Metal backends' launch geometry (see GpuTuning,
// gpu_tuning.h). Set at start-up, before the first backend is made (see
// setHardwareConfig), and again by the Windows GUI once Optimize for
// hardware has saved new launch settings, which a coordinator run then makes
// its backends again with before their next range (see runCoordinator).
struct HardwareConfig {
    // `gpus`: Auto (the default) searches with every discrete GPU the
    // backend has, or its integrated ones if it has nothing else; All with
    // every GPU; Listed with the ones `gpus` lists by number - none at all
    // for `gpus = none`, which leaves the cpu backend.
    enum class GpuChoice { Auto, All, Listed };
    GpuChoice gpuChoice = GpuChoice::Auto;
    std::vector<int> gpus;
    // `cpu_threads`: how many threads the cpu backend searches with - 0 (the
    // default) for one per hardware thread.
    int cpuThreads = 0;
    // The launch settings every GPU is made with, and one GPU's own
    // (`gpu<N>.<key>`) over them - only for a GPU named `gpu<N>.name`, if
    // that's set, so that settings tuned for one card aren't used for
    // another that has its number now.
    GpuTuning tuning;
    std::map<int, GpuTuning> gpuTunings;
    std::map<int, std::string> gpuNames;
};
void setHardwareConfig(const HardwareConfig& hardware);
HardwareConfig hardwareConfig();

// Every GPU the backend called `name` can search on here, by number. Empty
// for one without GPUs (cpu, reference), one this build doesn't have, or no
// GPU of its kind.
std::vector<GpuDevice> gpuDevices(const std::string& name);

// Which of `devices` `hardware` says to search with (see
// HardwareConfig::gpuChoice), in their order. A number `gpus` lists that
// isn't among them is left out.
std::vector<GpuDevice> selectedGpus(const std::vector<GpuDevice>& devices, const HardwareConfig& hardware);

// If `hardware`'s gpus picks `device` by its number, but gpu<N>.name says
// that number was another card's when it was tuned - the GPUs having been
// renumbered since, most likely - a warning saying so. Empty otherwise.
std::string gpuNumberMismatch(const HardwareConfig& hardware, const GpuDevice& device);

// The launch settings `device` is made with: `hardware`'s own, with its
// gpu<N> ones over them - unless they're for a GPU of another name, which
// `ignoredWhy`, if given, then says.
GpuTuning gpuTuningFor(const HardwareConfig& hardware, const GpuDevice& device, std::string* ignoredWhy = nullptr);

// The values the backend called `name` uses for whatever its GpuTuning leaves
// unset - every key of gpuTuningKeys(name) set. Empty for a backend this
// build doesn't have, or that takes no tuning.
GpuTuning defaultGpuTuning(const std::string& name);

// The backend called `name` - or, if `name` is empty, the first of
// backendNames() that can run on this machine. Null, with `error` set, if
// this build has no backend by that name, or it can't run here (e.g. a GPU
// backend on a machine without that kind of GPU, or with none of them
// selected by [hardware]'s `gpus`).
//
// A GPU backend searches on its GPU numbered `device` (see GpuDevice) - or,
// with `device` -1, on the first of those [hardware] selects (see
// selectedGpus). Any other backend takes no device.
//
// Every backend it returns has just passed selfTestBackend (self_test.h) on
// this machine. One that fails it is never returned: with `name` empty, the
// next backend is tried instead, after a message on stderr; with `name` given,
// the result is null and `*selfTestFailed` (if given) is set - so a test can
// tell a backend that searches wrongly from one that just can't run here.
//
// It's made as `hardware` says if given, else as hardwareConfig() does -
// `namebreak --autotune` tries its values this way, without touching what
// the rest of the process uses.
std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error, bool* selfTestFailed = nullptr,
                                             const HardwareConfig* hardware = nullptr, int device = -1);

// createBackend for a dictionary search (engine/dictionary_search.h): the
// backend called `name`, or the first of backendNames() that can search
// dictionaries and run on this machine - having also passed
// selfTestDictionaryBackend (self_test.h). Null, with `error` set, if the
// one called `name` can't search dictionaries either.
std::unique_ptr<SearchBackend> createDictionaryBackend(const std::string& name, std::string& error, bool* selfTestFailed = nullptr,
                                                       const HardwareConfig* hardware = nullptr, int device = -1);

// createBackend for a meet-in-the-middle search
// (engine/meet_in_the_middle_search.h): the backend called `name`, or the
// first of backendNames() that can search meet-in-the-middle targets and run
// on this machine - having also passed selfTestMeetInTheMiddleBackend
// (self_test.h). Null, with `error` set, if the one called `name` can't
// search meet-in-the-middle targets either.
std::unique_ptr<SearchBackend> createMeetInTheMiddleBackend(const std::string& name, std::string& error, bool* selfTestFailed = nullptr,
                                                            const HardwareConfig* hardware = nullptr, int device = -1);

#endif // NAMEBREAK_BACKENDS_BACKENDS_H
