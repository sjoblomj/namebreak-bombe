#ifndef NAMEBREAK_GUI_WIN32_OPTIMIZE_DIALOG_H
#define NAMEBREAK_GUI_WIN32_OPTIMIZE_DIALOG_H

#include <atomic>
#include <functional>
#include <string>

#include "common/autotune.h"
#include "gui/win32/win32.h"

namespace gui {

// The setup dialog's "Optimize for hardware" button: what `namebreak
// --autotune` does (common/autotune.h), behind a window. If the machine
// searches with a backend that can't be tuned this way - CUDA and HIP, whose
// settings are compiled in, or the CPU - a message box says so, with where to
// get the script that tunes CUDA and HIP. Otherwise, once the user agrees, it
// times the backend's settings on the GPU with a progress window
// (showAutotuneProgress) - each GPU [hardware] selects in turn - saves the
// fastest to `configPath`'s [hardware] section, as each GPU's own, and makes
// them the process's (setHardwareConfig).
// `backend` is the config's `backend` (empty for the first that can run).
//
// `runningMode` is the mode of the search running already - from the tray
// menu's Settings... - or empty at first run, before any has started; it
// decides what the user is told about when the settings are used: a
// search that hasn't started yet starts with them, a coordinator run makes
// its backend again with them once the range in hand is done (see
// runCoordinator), and a local search, which has no such point, uses them
// once namebreak is restarted. `searchPause`, if given, is that search's
// pause flag: it's paused while the GPU is being timed - it would make every
// measurement meaningless - and resumed after, unless it was paused already.
void optimizeForHardware(HWND owner, const std::string& configPath, const std::string& backend, const std::string& runningMode,
                         std::atomic<bool>* searchPause);

// The work showAutotuneProgress runs on a background thread: reports through
// the hooks, stops (soon) once `abortRequested` is set, and returns 0 when
// it's done - runAutotune's shape.
using AutotuneJob = std::function<int(const AutotuneHooks& hooks, std::atomic<bool>* abortRequested)>;

// A modal window over `owner` while `job` runs: what it's doing now, a
// progress bar, about how long is left, the lines it reports, and a Cancel
// button - Close once it's over, with `finishedText` under the bar if it
// succeeded. Returns the job's result, which is not 0 if it was cancelled
// or failed.
int showAutotuneProgress(HWND owner, AutotuneJob job, const std::string& finishedText);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_OPTIMIZE_DIALOG_H
