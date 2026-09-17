#ifndef NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
#define NAMEBREAK_CUDA_COORDINATOR_RUNNER_H

#include <atomic>
#include <map>
#include <string>

struct CoordinatorArgs {
    std::string serverUrl;
    // Empty means "not set in config.conf" - runCoordinator() auto-detects
    // and interactively confirms/persists a value for either of these two
    // before using them (see resolveMissingIdentity in coordinator_runner.cpp).
    std::string username;
    std::string hostname;
    int pollIntervalSecs = 30;
};

// Builds a CoordinatorArgs from a config.conf [coordinator] section (see
// config.h). Returns false with `error` set if a required key is missing or
// a value doesn't parse. username/hostname are left "" if absent from the
// section - see CoordinatorArgs above.
bool buildCoordinatorArgs(const std::map<std::string, std::string>& section, CoordinatorArgs& out, std::string& error);

// Runs the claim/search/heartbeat/complete loop against the coordinator
// server, forever, until the process is killed. Returns a process exit code
// only on a startup error (e.g. registration failure); otherwise this never
// returns under normal operation. Takes `args` by value since it fills in
// (and may prompt to confirm/persist) any of username/hostname left empty.
// `pauseRequested`, if given, is forwarded to every claimed range's
// runSearch() call - the heartbeat thread keeps running (and keeps
// heartbeating) regardless of it, since it's independent of the search
// itself; see runSearch's own doc comment (search.h) for what pausing does.
int runCoordinator(CoordinatorArgs args, const std::atomic<bool>* pauseRequested = nullptr);

#endif // NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
