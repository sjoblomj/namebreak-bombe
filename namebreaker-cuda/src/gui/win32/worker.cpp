#include "gui/win32/worker.h"

#include <memory>

#include "backends/backends.h"
#include "net/coordinator_runner.h"
#include "net/protocol.h"

namespace gui {

SharedStatus g_status;
std::atomic<bool> g_pauseRequested{false};
std::atomic<bool> g_quitRequested{false};

} // namespace gui

namespace {

using namespace gui;

void onRangeClaimed(const ClaimResponse& claim, const SearchRequest& req) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.targetName = claim.targetName;
    g_status.outputFilePath = req.outputFilePath;
    // A coordinator range is always bounded (a fixed candidate length - see
    // toSearchRequest, coordinator_runner.cpp), so it always has a real
    // finish line to measure progress against - see SharedStatus's comment.
    g_status.alphabet = req.alphabet;
    g_status.prefix = req.prefix;
    g_status.suffix = req.suffix;
    g_status.lowerBound = req.lowerBound;
    g_status.upperBound = req.upperBound;
    g_status.hasActiveRange = true;
    g_status.rangeJustFinished = false;
    g_status.statusText = "Searching target: " + claim.targetName;
}

void onRangeFinished(bool found) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.hasActiveRange = false;
    g_status.rangeJustFinished = true;
    g_status.statusText = found ? ("Match found in target: " + g_status.targetName) : "Range finished, no match";
}

void onStatus(const std::string& text) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.statusText = text;
}

// Runs a single bounded/continuous search directly (no coordinator server
// involved - see src/cli/main.cpp's main() for the CLI's equivalent one-shot
// call). Bounded mode searches one fixed candidate length - a real finish
// line, same as a coordinator range - so alphabet/lowerBound/upperBound are
// populated for it too. Continuous mode has no such line (it moves on to
// ever-longer candidate lengths indefinitely - see README.md's "Modes"), so
// those are left empty and updateUiFromSharedState shows the indeterminate
// marquee style for its whole run instead of a fraction that would stop
// meaning anything once it moves past its first length.
void runLocalSearch(const SearchRequest& req, bool continuous) {
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.targetName = req.prefix + "..." + req.suffix;
        g_status.outputFilePath = req.outputFilePath;
        if (continuous) {
            g_status.alphabet.clear();
            g_status.prefix.clear();
            g_status.suffix.clear();
            g_status.lowerBound.clear();
            g_status.upperBound.clear();
        } else {
            g_status.alphabet = req.alphabet;
            g_status.prefix = req.prefix;
            g_status.suffix = req.suffix;
            g_status.lowerBound = req.lowerBound;
            g_status.upperBound = req.upperBound;
        }
        g_status.hasActiveRange = true;
        g_status.rangeJustFinished = false;
        g_status.statusText = std::string("Running ") + (continuous ? "continuous" : "bounded") + " search...";
    }
    std::unique_ptr<SearchBackend> backend = createDefaultBackend();
    SearchResult result = runSearch(*backend, req, &g_quitRequested, nullptr, &g_pauseRequested);
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.hasActiveRange = false;
    g_status.rangeJustFinished = true;
    if (!result.ok)
        g_status.statusText = "Search failed: " + result.error;
    else if (result.aborted)
        g_status.statusText = "Aborted";
    else if (result.found)
        g_status.statusText = "Match found: " + result.filename;
    else
        g_status.statusText = "Search exhausted, no match";
}

} // namespace

namespace gui {

void workerThreadMain(AppConfig config, HWND notifyWindow) {
    if (config.mode == "coordinator") {
        CoordinatorCallbacks callbacks;
        callbacks.onRangeClaimed = onRangeClaimed;
        callbacks.onRangeFinished = onRangeFinished;
        callbacks.onStatus = onStatus;
        runCoordinator(config.coordinatorArgs, &g_pauseRequested, &g_quitRequested, &callbacks);
    } else {
        runLocalSearch(config.searchRequest, config.mode == "continuous");
    }
    // Marshal back onto the UI thread rather than touching any Win32 API
    // from here - see WM_APP_WORKER_STOPPED in MainWndProc.
    PostMessage(notifyWindow, WM_APP_WORKER_STOPPED, 0, 0);
}

} // namespace gui
