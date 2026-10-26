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
// `backend` is the backend to tune - empty for the one a search would use
// here (createBackend's choice, backends.h). CUDA and HIP are refused, with
// a pointer to scripts/autotune/, which tunes their compiled-in values. `abortRequested`, if given, ends the tuning early (and
// writes nothing). Returns the process's exit code: 0 once done, 1 on an
// error, which it has printed.
int runAutotune(const std::string& configPath, const std::string& backend, double secondsPerTrial, std::atomic<bool>* abortRequested);

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
// line per measurement.
GpuTuning tuneAxes(const GpuTuning& start, const std::vector<TuningAxis>& axes, double minGain, int maxRounds,
                   const std::function<double(const GpuTuning&)>& measure, const std::function<void(const std::string&)>& log = nullptr);

#endif // NAMEBREAK_COMMON_AUTOTUNE_H
