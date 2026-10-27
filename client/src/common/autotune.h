#ifndef NAMEBREAK_COMMON_AUTOTUNE_H
#define NAMEBREAK_COMMON_AUTOTUNE_H

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "backends/gpu_tuning.h"

// `namebreak --autotune`: measures the OpenCL or Metal backend's launch
// geometry (GpuTuning, backends/gpu_tuning.h) on this machine's GPU and
// writes what searched fastest to the config file's [opencl] or [metal]
// section (see buildBackendTunings, config.h) - the backend's defaults were
// only ever measured on one RTX 3080 Ti Laptop.
//
// Each trial makes the backend afresh with the values tried (so it passes
// its self-tests with them, or the values are skipped), warms it up, and
// times a fixed search: for the row search values, a search like
// search_bench's (the real 49-character alphabet, ten characters, nothing
// pruned, an unreachable target); for the dictionary ones, one like
// dictionary_bench's (english-1, up to three words a candidate, with an
// encryption key). Their sizes are set to take about `secondsPerTrial`
// each, after a search of about half a minute with the defaults has heated
// the GPU up - one that starts cool searches faster until it's warm, a
// laptop's up to three times as fast. Files are written to a temporary
// directory, which is removed afterwards.
//
// The values are tuned one at a time (see tuneAxes), all of them starting
// from the backend's defaults. Then defaults and result are timed twice more
// each: only if the result is more than 2% faster are its values written,
// and only those that differ from the defaults - a value equal to one is
// taken out of the
// section, so later releases' defaults apply. dictionary_tails_per_cell,
// which only searches with tails would show, is left as the file has it.
//
// `backend` is the backend to tune, as autotuneBackend says.
// `abortRequested`, if given, ends the tuning after the trial in progress,
// and writes nothing. What it does it reports through `hooks` - without
// them, results go to stdout and errors to stderr, as for the CLI. Returns
// the process's exit code: 0 once done, 1 if stopped or on an error, which
// it has reported.
struct AutotuneHooks {
    // A line of results, or a message - what the CLI prints. Empty lines
    // separate the parts.
    std::function<void(const std::string&)> onLine;
    // What made it give up.
    std::function<void(const std::string&)> onError;
    // What it's doing now, e.g. "Timing a row search with rows_per_thread 25".
    std::function<void(const std::string&)> onStatus;
    // How far it has got, from 0 to 1, and about how many seconds are left -
    // a rough guess, from how long the trials so far took.
    std::function<void(double fraction, double secondsLeft)> onProgress;
};
int runAutotune(const std::string& configPath, const std::string& backend, double secondsPerTrial, std::atomic<bool>* abortRequested,
                const AutotuneHooks& hooks = AutotuneHooks());

// Where scripts/autotune/ - which tunes the CUDA and HIP backends'
// compiled-in values - and its README are online.
inline constexpr const char* kAutotuneScriptUrl = "https://github.com/sjoblomj/namebreak-bombe/tree/main/client/scripts/autotune";

// The backend runAutotune would tune: `requested`, or with that empty, the
// one a search would use here (createBackend's choice, backends.h) - a CUDA
// build with OpenCL too searches with CUDA, and tuning OpenCL wouldn't speed
// that up. Empty, with `why` saying why for a person, if it can't: CUDA and
// HIP, whose values are compiled in (pointing to kAutotuneScriptUrl), the
// CPU backends, or a backend this build doesn't have.
std::string autotuneBackend(const std::string& requested, std::string& why);

// One value to tune, and the values to try for it.
struct TuningAxis {
    GpuTuningKey key;
    std::vector<int> values;
};

// Tunes `axes` one at a time, `start` holding every one's starting value:
// for each axis, every one of its values is measured with the other axes'
// as they stand, the value it stands at first and once more after the
// others (the two averaged, so that a GPU warming up as it goes doesn't
// favor whichever went first), and the fastest one taken if it's more than
// `minGain` (e.g. 0.02 for 2%) faster than that. Once every axis has been
// through, it goes round again, as long as one changed - at most
// `maxRounds` times. `measure` returns a rate (higher is better), or 0 for
// values that can't be used, which are never taken. `log`, if given, gets a
// line per measurement, and `trying`, before each, what it's about to
// measure ("rows_per_thread 25").
GpuTuning tuneAxes(const GpuTuning& start, const std::vector<TuningAxis>& axes, double minGain, int maxRounds,
                   const std::function<double(const GpuTuning&)>& measure, const std::function<void(const std::string&)>& log = nullptr,
                   const std::function<void(const std::string&)>& trying = nullptr);

#endif // NAMEBREAK_COMMON_AUTOTUNE_H
