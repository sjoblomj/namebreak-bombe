#ifndef NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
#define NAMEBREAK_CUDA_COORDINATOR_RUNNER_H

#include <atomic>
#include <map>
#include <string>

#include "config.h"

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
