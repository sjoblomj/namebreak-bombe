#include "gui/win32/worker.h"

#include <algorithm>
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

// The search the calling thread's progress is about, in g_status: the
// coordinator worker's the thread is (coordinatorWorker), or the only one.
// g_status.mutex must be held.
SearchStatus& searchOfThisThread() {
    const size_t worker = (size_t) std::max(0, coordinatorWorker());
    if (worker >= g_status.searches.size())
        g_status.searches.resize(worker + 1);
    return g_status.searches[worker];
}

// Sets the calling thread's search's status line. g_status.mutex must be
// held.
void setStatusText(SearchStatus& search, const std::string& text) {
    search.statusText = text;
    g_status.lastStatus = text;
}

// The progress of a search that counts its own work, as the status line and
// the matches box show it: a dictionary search (a local one, or each of a
// coordinator's dictionary ranges) counting candidates, or a
// meet-in-the-middle one counting tails. The status line is the CLI's
// progress line, shortened. Called on the thread of the search it's about -
// a coordinator run's workers each have their own (see searchOfThisThread).
class CountedProgressText {
public:
    // `unit` is what this kind of search counts and `hit`/`hits` what it
    // finds along the way, for the status line; `found` is the list in a
    // SearchStatus those go to, for the matches box.
    CountedProgressText(const char* unit, const char* hit, const char* hits, std::vector<std::string> SearchStatus::* found)
        : unit_(unit), hit_(hit), hits_(hits), found_(found) {}

    // Before every search, for only its own work to count towards the rate -
    // not what a resumed search had searched before.
    void start() {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        counter() = Counter();
    }

    void hit(const std::string& text) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        ++counter().hits;
        (searchOfThisThread().*found_).push_back(text);
    }

    void count(uint64_t searched, uint64_t toSearch) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        const Counter& own = counter();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - own.started).count();
        const double rate = elapsed > 0 ? (double) searched / elapsed : 0;
        const double fraction = toSearch ? (double) searched / (double) toSearch : 1.0;
        char percent[16];
        snprintf(percent, sizeof percent, "%.2f%%", 100.0 * fraction);
        std::string text = std::string(percent) + " of " + compactCount(toSearch) + " " + unit_ + ", about " +
                           (rate > 0 ? formatDuration((double) (toSearch - searched) / rate) : "?") + " left";
        if (own.hits)
            text += " - " + withCommas(own.hits) + " " + (own.hits == 1 ? hit_ : hits_);
        SearchStatus& search = searchOfThisThread();
        search.progressFraction = fraction;
        setStatusText(search, text);
    }

    void position(const std::string& filename) {
        if (filename.empty())
            return;
        std::lock_guard<std::mutex> lock(g_status.mutex);
        searchOfThisThread().newSearchPosition = filename;
    }

private:
    struct Counter {
        std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        uint64_t hits = 0;
    };
    // The calling thread's search's. g_status.mutex must be held.
    Counter& counter() {
        const size_t worker = (size_t) std::max(0, coordinatorWorker());
        if (worker >= counters_.size())
            counters_.resize(worker + 1);
        return counters_[worker];
    }

    const char* unit_;
    const char* hit_;
    const char* hits_;
    std::vector<std::string> SearchStatus::* found_;
    std::vector<Counter> counters_;
};

CountedProgressText g_dictionaryProgress{"candidates", "basename", "basenames", &SearchStatus::newBasenames};
CountedProgressText g_meetInTheMiddleProgress{"tails", "directory", "directories", &SearchStatus::newMeetingPointMatches};

void onWorkers(const std::vector<std::string>& gpus) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.gpus = gpus;
    g_status.searches.resize(gpus.size());
}

void onRangeClaimed(const ClaimResponse& claim, const SearchRequest& req) {
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        SearchStatus& search = searchOfThisThread();
        search.targetName = claim.targetName + " (range " + std::to_string(claim.rangeId) + ")";
        search.outputFilePath = req.outputFilePath;
        // A coordinator range is always bounded (a fixed candidate length -
        // see toSearchRequest, coordinator_runner.cpp), so it always has a
        // real finish line to measure progress against - see SearchStatus's
        // comment.
        search.alphabet = req.alphabet;
        search.prefix = req.prefix;
        search.suffix = req.suffix;
        search.insertFromStart = req.insertFromStart;
        search.insertFromEnd = req.insertFromEnd;
        search.lowerBound = req.lowerBound;
        search.upperBound = req.upperBound;
        // A dictionary range counts its own candidates, and a
        // meet-in-the-middle one its own tails - from 0, until its first
        // count (see CountedProgressText::count) replaces the status line
        // too. Neither has a candidate order for the matches file to be
        // measured against, so the count is all there is to go by.
        search.progressFraction = claim.dictionary || claim.meetInTheMiddle ? 0 : -1;
        search.hasActiveRange = true;
        search.rangeJustFinished = false;
        setStatusText(search, "Searching '" + claim.lowerBoundFilename + "' to '" + claim.upperBoundFilename + "'");
    }
    if (claim.dictionary)
        g_dictionaryProgress.start();
    else if (claim.meetInTheMiddle)
        g_meetInTheMiddleProgress.start();
}

void onRangeFinished(bool found) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    SearchStatus& search = searchOfThisThread();
    search.hasActiveRange = false;
    search.rangeJustFinished = true;
    setStatusText(search, found ? ("Match found in target: " + search.targetName) : "Range finished, no match");
}

void onStatus(const std::string& text) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    setStatusText(searchOfThisThread(), text);
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
        g_status.searches[0].targetName = req.prefix + "..." + req.suffix;
        g_status.searches[0].outputFilePath = req.outputFilePath;
        if (continuous) {
            g_status.searches[0].alphabet.clear();
            g_status.searches[0].prefix.clear();
            g_status.searches[0].suffix.clear();
            g_status.searches[0].lowerBound.clear();
            g_status.searches[0].upperBound.clear();
        } else {
            g_status.searches[0].alphabet = req.alphabet;
            g_status.searches[0].prefix = req.prefix;
            g_status.searches[0].suffix = req.suffix;
            g_status.searches[0].insertFromStart = req.insertFromStart;
            g_status.searches[0].insertFromEnd = req.insertFromEnd;
            g_status.searches[0].lowerBound = req.lowerBound;
            g_status.searches[0].upperBound = req.upperBound;
        }
        g_status.searches[0].progressFraction = -1;
        g_status.searches[0].hasActiveRange = true;
        g_status.searches[0].rangeJustFinished = false;
        setStatusText(g_status.searches[0], std::string("Running ") + (continuous ? "continuous" : "bounded") + " search...");
    }
    std::string backendError;
    std::unique_ptr<SearchBackend> backend = createBackend(backendName, backendError);
    if (!backend) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.searches[0].hasActiveRange = false;
        setStatusText(g_status.searches[0], backendError);
        return;
    }
    SearchResult result = runSearch(*backend, req, &g_quitRequested, nullptr, &g_pauseRequested);
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.searches[0].hasActiveRange = false;
    g_status.searches[0].rangeJustFinished = true;
    if (!result.ok)
        setStatusText(g_status.searches[0], "Search failed: " + result.error);
    else if (result.aborted)
        setStatusText(g_status.searches[0], "Aborted");
    else if (result.found)
        setStatusText(g_status.searches[0], "Match found: " + result.filename);
    else
        setStatusText(g_status.searches[0], "Search exhausted, no match");
    g_status.searchEnded = result.ok && !result.aborted;
}

// Runs a dictionary search directly (see src/cli/main.cpp's main() for the
// CLI's equivalent). It counts its own candidates, so the progress bar is
// that fraction (see SearchStatus::progressFraction) - see
// CountedProgressText.
void runLocalDictionarySearch(const DictionaryRequest& req, const std::string& backendName) {
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.searches[0].targetName = req.prefix + "<words>" + req.suffix;
        g_status.searches[0].outputFilePath = req.outputFilePath;
        g_status.searches[0].alphabet.clear();
        g_status.searches[0].lowerBound.clear();
        g_status.searches[0].upperBound.clear();
        g_status.searches[0].progressFraction = 0;
        g_status.searches[0].hasActiveRange = true;
        g_status.searches[0].rangeJustFinished = false;
        setStatusText(g_status.searches[0], "Testing the backend before the dictionary search...");
    }
    std::string backendError;
    std::unique_ptr<SearchBackend> backend = createDictionaryBackend(backendName, backendError);
    if (!backend) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.searches[0].hasActiveRange = false;
        setStatusText(g_status.searches[0], backendError);
        return;
    }

    g_dictionaryProgress.start();
    DictionarySearchHooks hooks;
    hooks.onBasenameMatch = [](const std::string& basename, const std::string&) { g_dictionaryProgress.hit(basename); };
    hooks.onCount = [](uint64_t searched, uint64_t toSearch) { g_dictionaryProgress.count(searched, toSearch); };
    hooks.onStatus = [](const std::string& filename) { g_dictionaryProgress.position(filename); };
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        setStatusText(g_status.searches[0], "Running dictionary search on " + std::string(backend->name()) + "...");
    }
    DictionaryResult result = runDictionarySearch(*backend, req, &g_quitRequested, hooks, &g_pauseRequested);
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.searches[0].hasActiveRange = false;
    g_status.searches[0].rangeJustFinished = true;
    const std::string basenames = result.basenameHits ? " - " + withCommas(result.basenameHits) + " basenames" : "";
    g_status.searchEnded = result.ok && !result.aborted;
    if (!result.ok)
        setStatusText(g_status.searches[0], "Search failed: " + result.error);
    else if (result.aborted)
        setStatusText(g_status.searches[0], "Aborted" + basenames);
    else if (result.found)
        setStatusText(g_status.searches[0], "Match found: " + result.filename);
    else
        setStatusText(g_status.searches[0], "Searched every candidate, no match" + basenames);
}

// Runs a meet-in-the-middle search directly (see src/cli/main.cpp's main()
// for the CLI's equivalent). Like a dictionary search it counts its own work
// - tails rather than candidates - so the progress bar is that fraction.
void runLocalMeetInTheMiddleSearch(const MeetInTheMiddleRequest& req, const std::string& backendName) {
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.searches[0].targetName = req.prefix + "<directory>" + req.suffix;
        g_status.searches[0].outputFilePath = req.outputFilePath;
        g_status.searches[0].alphabet.clear();
        g_status.searches[0].lowerBound.clear();
        g_status.searches[0].upperBound.clear();
        g_status.searches[0].progressFraction = 0;
        g_status.searches[0].hasActiveRange = true;
        g_status.searches[0].rangeJustFinished = false;
        setStatusText(g_status.searches[0], "Testing the backend before the meet-in-the-middle search...");
    }
    std::string backendError;
    std::unique_ptr<SearchBackend> backend = createMeetInTheMiddleBackend(backendName, backendError);
    if (!backend) {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.searches[0].hasActiveRange = false;
        setStatusText(g_status.searches[0], backendError);
        return;
    }

    g_meetInTheMiddleProgress.start();
    MeetInTheMiddleSearchHooks hooks;
    hooks.onPartialMatch = [](const std::string& filename) { g_meetInTheMiddleProgress.hit(filename); };
    hooks.onCount = [](uint64_t searched, uint64_t toSearch) { g_meetInTheMiddleProgress.count(searched, toSearch); };
    hooks.onStatus = [](const std::string& tail) { g_meetInTheMiddleProgress.position(tail); };
    {
        // The table is built first, which for a big one takes a while and
        // says nothing until it's done - hence a status line of its own,
        // replaced by the first count once the walk starts.
        std::lock_guard<std::mutex> lock(g_status.mutex);
        setStatusText(g_status.searches[0], "Building the table, then searching on " + std::string(backend->name()) + "...");
    }
    MeetInTheMiddleResult result = runMeetInTheMiddleSearch(*backend, req, &g_quitRequested, hooks, &g_pauseRequested);
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.searches[0].hasActiveRange = false;
    g_status.searches[0].rangeJustFinished = true;
    const std::string hits = result.hits ? " - " + withCommas(result.hits) + (result.hits == 1 ? " directory" : " directories") +
                                               " matched the meeting point"
                                         : "";
    g_status.searchEnded = result.ok && !result.aborted;
    if (!result.ok)
        setStatusText(g_status.searches[0], "Search failed: " + result.error);
    else if (result.aborted)
        setStatusText(g_status.searches[0], "Aborted" + hits);
    else if (result.found)
        setStatusText(g_status.searches[0], "Match found: " + result.filename);
    else
        setStatusText(g_status.searches[0], "Searched every directory, no match" + hits);
}

} // namespace

namespace gui {

void workerThreadMain(AppConfig config, HWND notifyWindow) {
    if (config.mode == "coordinator") {
        CoordinatorCallbacks callbacks;
        callbacks.onWorkers = onWorkers;
        callbacks.onRangeClaimed = onRangeClaimed;
        callbacks.onRangeFinished = onRangeFinished;
        callbacks.onStatus = onStatus;
        callbacks.onDictionaryCount = [](uint64_t searched, uint64_t toSearch) { g_dictionaryProgress.count(searched, toSearch); };
        callbacks.onDictionaryStatus = [](const std::string& filename) { g_dictionaryProgress.position(filename); };
        callbacks.onDictionaryBasename = [](const std::string& basename) { g_dictionaryProgress.hit(basename); };
        callbacks.onMeetInTheMiddleCount = [](uint64_t searched, uint64_t toSearch) { g_meetInTheMiddleProgress.count(searched, toSearch); };
        callbacks.onMeetInTheMiddleStatus = [](const std::string& tail) { g_meetInTheMiddleProgress.position(tail); };
        callbacks.onMeetInTheMiddleMatch = [](const std::string& filename) { g_meetInTheMiddleProgress.hit(filename); };
        runCoordinator(config.coordinatorArgs, &g_pauseRequested, &g_quitRequested, &callbacks, &g_finishRangeThenPause);
    } else if (config.mode == "dictionary") {
        runLocalDictionarySearch(config.dictionaryRequest, config.backend);
    } else if (config.mode == "meet-in-the-middle") {
        runLocalMeetInTheMiddleSearch(config.meetInTheMiddleRequest, config.backend);
    } else {
        runLocalSearch(config.searchRequest, config.mode == "continuous", config.backend);
    }
    // Marshal back onto the UI thread rather than touching any Win32 API
    // from here - see WM_APP_WORKER_STOPPED in MainWndProc.
    PostMessage(notifyWindow, WM_APP_WORKER_STOPPED, 0, 0);
}

} // namespace gui
