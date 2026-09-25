#ifndef NAMEBREAK_NET_COORDINATOR_RUNNER_H
#define NAMEBREAK_NET_COORDINATOR_RUNNER_H

#include <atomic>
#include <functional>
#include <map>
#include <string>

#include "common/config.h"
#include "net/protocol.h"
#include "engine/search.h"

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
    // Where each target's matches file goes - config.conf's matches_dir (see
    // matches_file.h).
    std::string matchesDir = kDefaultMatchesDir;
    // The backend to search with - config.conf's `backend`, or --backend;
    // empty for the first one that can run here (see backends/backends.h).
    std::string backend;
};

// Builds a CoordinatorArgs from `config`'s [coordinator] section (and its
// matches_dir and backend - see config.h). Returns false with `error` set if a required key is missing or
// a value doesn't parse. username/hostname are left "" if absent from the
// section - see CoordinatorArgs above.
bool buildCoordinatorArgs(const ConfigFile& config, CoordinatorArgs& out, std::string& error);

// Fills in whichever of args.username/args.hostname are empty: auto-detects
// them, and on an interactive terminal asks the user to confirm or override
// them and saves the result to args.configPath. A no-op when both are set.
// runCoordinator() calls this itself; the CLI's main() calls it earlier as
// well, before its pause-key listener puts the terminal into single-key,
// no-echo mode and starts reading stdin - either of which would otherwise
// garble the prompt's input.
void resolveMissingIdentity(CoordinatorArgs& args);

// Optional hooks a caller (e.g. the Windows GUI, src/gui/win32/worker.cpp)
// can pass into runCoordinator() to observe its progress without a console -
// the CLI's main() passes none of these (all fields default-null), so its
// behavior is completely unaffected. Every callback may be called from a background
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
    // the range: candidates are enumerated in a fixed order (see candidate.h's
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
// `finishRangeThenPause`, if given (and `pauseRequested` too), asks for a
// pause once the range in hand is finished, before any new work is claimed:
// whenever this loop is about to claim and finds it set, it clears it, sets
// `*pauseRequested` and waits there like any other pause. Set while no
// range is in hand (paused between ranges, or waiting for work), that means
// pausing right away.
int runCoordinator(CoordinatorArgs args, std::atomic<bool>* pauseRequested = nullptr, std::atomic<bool>* quitRequested = nullptr,
                    const CoordinatorCallbacks* callbacks = nullptr, std::atomic<bool>* finishRangeThenPause = nullptr);

#endif // NAMEBREAK_NET_COORDINATOR_RUNNER_H
