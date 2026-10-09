#include "gui/win32/worker.h"

#include <chrono>
#include <cstdio>
#include <memory>

#include "backends/backends.h"
#include "net/coordinator_runner.h"
#include "net/protocol.h"

namespace gui {

SharedStatus g_status;
std::atomic<bool> g_pauseRequested{false};
std::atomic<bool> g_finishRangeThenPause{false};
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
    g_status.insertFromStart = req.insertFromStart;
    g_status.insertFromEnd = req.insertFromEnd;
    g_status.lowerBound = req.lowerBound;
    g_status.upperBound = req.upperBound;
    g_status.progressFraction = -1;
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
void runLocalSearch(const SearchRequest& req, bool continuous, const std::string& backendName) {
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
            g_status.insertFromStart = req.insertFromStart;
            g_status.insertFromEnd = req.insertFromEnd;
            g_status.lowerBound = req.lowerBound;
            g_status.upperBound = req.upperBound;
        }
        g_status.progressFraction = -1;
        g_status.hasActiveRange = true;
        g_status.rangeJustFinished = false;
        g_status.statusText = std::string("Running ") + (continuous ? "continuous" : "bounded") + " search...";
    }
    std::string backendError;
    std::unique_ptr<SearchBackend> backend = createBackend(backendName, backendError);
    if (!backend) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.hasActiveRange = false;
        g_status.statusText = backendError;
        return;
    }
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
    g_status.searchEnded = result.ok && !result.aborted;
}

// `n` with thousands separators: 4,080,040,000.
std::string withCommas(uint64_t n) {
    std::string digits = std::to_string(n), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0)
            out += ',';
        out += digits[i];
    }
    return out;
}

// `n` short enough for the status line: 99,999, then 3 significant digits
// and a unit - 4.08 G, 261 T.
std::string compactCount(uint64_t n) {
    if (n < 100000)
        return withCommas(n);
    const char* units[] = {"K", "M", "G", "T", "P", "E"};
    double value = (double) n / 1000;
    int unit = 0;
    while (value >= 999.5 && unit < 5) {
        value /= 1000;
        ++unit;
    }
    char buf[32];
    snprintf(buf, sizeof buf, "%.3g %s", value, units[unit]);
    return buf;
}

// Runs a dictionary search directly (see src/cli/main.cpp's main() for the
// CLI's equivalent). It counts its own candidates, so the progress bar is
// that fraction (see SharedStatus::progressFraction), and the status line
// the CLI's progress line, shortened.
void runLocalDictionarySearch(const DictionaryRequest& req, const std::string& backendName) {
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.targetName = req.prefix + "<words>" + req.suffix;
        g_status.outputFilePath = req.outputFilePath;
        g_status.alphabet.clear();
        g_status.lowerBound.clear();
        g_status.upperBound.clear();
        g_status.progressFraction = 0;
        g_status.hasActiveRange = true;
        g_status.rangeJustFinished = false;
        g_status.statusText = "Testing the backend before the dictionary search...";
    }
    std::string backendError;
    std::unique_ptr<SearchBackend> backend = createDictionaryBackend(backendName, backendError);
    if (!backend) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.hasActiveRange = false;
        g_status.statusText = backendError;
        return;
    }

    // Only this run's candidates count towards the rate - not those a
    // resumed search had searched before.
    const auto startTime = std::chrono::steady_clock::now();
    uint64_t basenameHits = 0;
    DictionarySearchHooks hooks;
    hooks.onBasenameMatch = [&](const std::string& basename, const std::string&) {
        ++basenameHits;
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.newBasenames.push_back(basename);
    };
    hooks.onCount = [&](uint64_t searched, uint64_t toSearch) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        const double rate = elapsed > 0 ? (double) searched / elapsed : 0;
        const double fraction = toSearch ? (double) searched / (double) toSearch : 1.0;
        char percent[16];
        snprintf(percent, sizeof percent, "%.2f%%", 100.0 * fraction);
        std::string text = std::string(percent) + " of " + compactCount(toSearch) + " candidates, about " +
                           (rate > 0 ? formatDuration((double) (toSearch - searched) / rate) : "?") + " left";
        if (basenameHits)
            text += " - " + withCommas(basenameHits) + (basenameHits == 1 ? " basename" : " basenames");
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.progressFraction = fraction;
        g_status.statusText = text;
    };
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.statusText = "Running dictionary search on " + std::string(backend->name()) + "...";
    }
    DictionaryResult result = runDictionarySearch(*backend, req, &g_quitRequested, hooks, &g_pauseRequested);
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.hasActiveRange = false;
    g_status.rangeJustFinished = true;
    const std::string basenames = result.basenameHits ? " - " + withCommas(result.basenameHits) + " basenames" : "";
    g_status.searchEnded = result.ok && !result.aborted;
    if (!result.ok)
        g_status.statusText = "Search failed: " + result.error;
    else if (result.aborted)
        g_status.statusText = "Aborted" + basenames;
    else if (result.found)
        g_status.statusText = "Match found: " + result.filename;
    else
        g_status.statusText = "Searched every candidate, no match" + basenames;
}

} // namespace

namespace gui {

void workerThreadMain(AppConfig config, HWND notifyWindow) {
    if (config.mode == "coordinator") {
        CoordinatorCallbacks callbacks;
        callbacks.onRangeClaimed = onRangeClaimed;
        callbacks.onRangeFinished = onRangeFinished;
        callbacks.onStatus = onStatus;
        runCoordinator(config.coordinatorArgs, &g_pauseRequested, &g_quitRequested, &callbacks, &g_finishRangeThenPause);
    } else if (config.mode == "dictionary") {
        runLocalDictionarySearch(config.dictionaryRequest, config.backend);
    } else {
        runLocalSearch(config.searchRequest, config.mode == "continuous", config.backend);
    }
    // Marshal back onto the UI thread rather than touching any Win32 API
    // from here - see WM_APP_WORKER_STOPPED in MainWndProc.
    PostMessage(notifyWindow, WM_APP_WORKER_STOPPED, 0, 0);
}

} // namespace gui
