#ifndef NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
#define NAMEBREAK_CUDA_COORDINATOR_RUNNER_H

#include <atomic>
#include <functional>
#include <map>
#include <string>

#include "config.h"
#include "protocol.h"
#include "search.h"

struct CoordinatorArgs {
    std::string serverUrl;
    // Empty means "not set in config.conf" - runCoordinator() auto-detects
    // and interactively confirms/persists a value for either of these two
    // before using them (see resolveMissingIdentity in coordinator_runner.cpp).
    std::string username;
    std::string hostname;
    int pollIntervalSecs = 30;
    // Path the config was actually loaded from (kDefaultConfigPath, or
    // whatever --config <file> gave main()) - not itself a config.conf key,
    // main() fills this in after buildCoordinatorArgs(). resolveMissingIdentity
    // writes any interactively-confirmed username/hostname back here, so a
    // --config'd run persists to the file it was actually read from rather
    // than always writing back to kDefaultConfigPath.
    std::string configPath = kDefaultConfigPath;
};

// Builds a CoordinatorArgs from a config.conf [coordinator] section (see
// config.h). Returns false with `error` set if a required key is missing or
// a value doesn't parse. username/hostname are left "" if absent from the
// section - see CoordinatorArgs above.
bool buildCoordinatorArgs(const std::map<std::string, std::string>& section, CoordinatorArgs& out, std::string& error);

// Optional hooks a caller (e.g. the Windows GUI, gui_win32.cpp) can pass into
// runCoordinator() to observe its progress without a console - the CLI's
// main() passes none of these (all fields default-null), so its behavior is
// completely unaffected. Every callback may be called from a background
// thread (the same thread running runCoordinator()/runOneRange()), never
// the caller's own thread - a GUI must not touch its widgets directly from
// inside one of these; write into caller-owned, mutex-protected state
// instead and let the GUI's own thread read that on its own schedule.
struct CoordinatorCallbacks {
    // Fired once a range is claimed, before it's searched - gives the target
    // name and the exact SearchRequest that range is about to run (including
    // outputFilePath - where that range's Hash-A matches will be appended -
    // and alphabet/lowerBound/upperBound, which a caller can use together
    // with the matches file's own last line to derive real progress through
    // the range: candidates are enumerated in a fixed order (see cpu-utils.h's
    // stringToIndex), so the most recent Hash-A-only match's position in that
    // order is a truer measure of how far a search has gotten than any
    // time-based guess - there's no live "candidates processed so far"
    // counter available anywhere, client or server, so this is the best
    // signal there is).
    std::function<void(const ClaimResponse&, const SearchRequest&)> onRangeClaimed;
    // Fired when the current range's search ends, however it ended (found,
    // exhausted, or aborted) - lets a progress bar be reset/completed.
    std::function<void(bool found)> onRangeFinished;
    // Outer-loop lifecycle text only (registered, registration failed,
    // claim retry/backoff, no work available) - mirrors the printf/fprintf
    // already at those call sites in runCoordinator(). Does NOT cover
    // runOneRange()'s per-range diagnostic prints (heartbeat conflicts,
    // lost-ownership wording, etc.) - those stay console-only; a caller's
    // per-range state should come from onRangeClaimed/onRangeFinished
    // instead.
    std::function<void(const std::string&)> onStatus;
};

// Runs the claim/search/heartbeat/complete loop against the coordinator
// server, until `quitRequested` is set or the process is killed. Returns a
// process exit code only on a startup error (e.g. registration failure) or
// once `quitRequested` is observed (0); otherwise this never returns under
// normal operation. Takes `args` by value since it fills in (and may prompt
// to confirm/persist) any of username/hostname left empty.
// `pauseRequested`, if given, is forwarded to every claimed range's
// runSearch() call - the heartbeat thread keeps running (and keeps
// heartbeating) regardless of it, since it's independent of the search
// itself; see runSearch's own doc comment (search.h) for what pausing does.
// `quitRequested`, if given, is polled (at intervals no coarser than the
// existing pause-wait spin's ~500ms) everywhere this loop would otherwise
// block indefinitely or for a long backoff, and is also forwarded to each
// claimed range's search so it can be interrupted almost immediately rather
// than run to completion. `callbacks`, if given, is notified per
// CoordinatorCallbacks above.
int runCoordinator(CoordinatorArgs args, const std::atomic<bool>* pauseRequested = nullptr,
                    std::atomic<bool>* quitRequested = nullptr, const CoordinatorCallbacks* callbacks = nullptr);

#endif // NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
