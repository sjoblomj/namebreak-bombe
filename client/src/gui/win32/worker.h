#ifndef NAMEBREAK_GUI_WIN32_WORKER_H
#define NAMEBREAK_GUI_WIN32_WORKER_H

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "gui/win32/app_config.h"
#include "gui/win32/win32.h"

namespace gui {

// Fired once the worker thread's runCoordinator()/runSearch()/
// runDictionarySearch() call returns,
// whichever mode this run is in - see workerThreadMain. (WM_APP + 1 is the
// main window's tray icon message.)
constexpr UINT WM_APP_WORKER_STOPPED = WM_APP + 2;

// ---------------------------------------------------------------------
// Cross-thread state - written by CoordinatorCallbacks on the worker
// thread, read by the UI thread's once-a-second timer. No Win32 calls ever
// happen inside the callbacks themselves (see coordinator_runner.h's own
// warning on this) - they only ever touch this mutex-protected struct.
// ---------------------------------------------------------------------

struct SharedStatus {
    std::mutex mutex;
    std::string targetName;
    std::string outputFilePath;
    // Populated only for "bounded-shaped" work - a coordinator range, or
    // local bounded mode - which has a real finish line (a fixed, single
    // candidate length between lowerBound/upperBound); left empty for
    // continuous mode, which has none by design (see runLocalSearch), and
    // before any range/search has started. When populated, the UI timer
    // derives a real progress fraction from these plus the matches file's
    // own last line (see matchProgressFraction/updateUiFromSharedState)
    // instead of guessing from elapsed time: candidates are enumerated in a
    // fixed order (candidate.h's stringToIndex), so the most recent Hash-A-
    // only match's position in that order is a true measure of how far a
    // search has gotten - the same signal the coordinator server itself
    // already uses for stall detection (see coordinator/README.md). When
    // empty, the UI shows the indeterminate marquee style instead.
    std::string alphabet;
    std::string prefix;
    std::string suffix;
    Insertion insertFromStart;
    Insertion insertFromEnd;
    std::string lowerBound; // candidate-only (no prefix/suffix)
    std::string upperBound; // candidate-only
    // How far a local dictionary search has got, from 0 to 1 - it counts its
    // own candidates (DictionarySearchHooks::onCount), so this is used as it
    // is, rather than worked out from the matches file. -1 for any other
    // search, and a coordinator's dictionary range, which has neither (see
    // runDictionaryRange) and so shows the marquee style.
    double progressFraction = -1;
    bool hasActiveRange = false;
    // Set true by onRangeFinished, consumed (cleared back to false) the next
    // time the UI timer reads this struct - a one-shot "a range just ended"
    // edge, not a level, so the progress bar only snaps to 100% once.
    bool rangeJustFinished = false;
    // Basenames a local dictionary search has found since the UI timer last
    // took them (and cleared this) - for the matches box.
    std::vector<std::string> newBasenames;
    // A local search ran to its end - found the file, or searched every
    // candidate - rather than stopping on an error.
    bool searchEnded = false;
    std::string statusText = "Starting...";
};

extern SharedStatus g_status;

// Set by the UI thread, polled by the search/coordinator loop.
extern std::atomic<bool> g_pauseRequested;
// Coordinator mode only: pause once the current range is finished - see
// runCoordinator's finishRangeThenPause, which also clears it again.
extern std::atomic<bool> g_finishRangeThenPause;
extern std::atomic<bool> g_quitRequested;

// The worker thread's body: runs whichever mode was configured (see
// AppConfig) - on a background thread, since runCoordinator(), runSearch()
// and runDictionarySearch() all block for the life of the run. Always ends with a
// WM_APP_WORKER_STOPPED post to `notifyWindow` (see MainWndProc) - never any
// other Win32 call from here, same rule SharedStatus's own comment already
// states for the coordinator callbacks (worker.cpp).
void workerThreadMain(AppConfig config, HWND notifyWindow);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_WORKER_H
