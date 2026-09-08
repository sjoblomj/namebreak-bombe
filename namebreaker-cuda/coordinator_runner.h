#ifndef NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
#define NAMEBREAK_CUDA_COORDINATOR_RUNNER_H

#include <map>
#include <string>

struct CoordinatorArgs {
    std::string serverUrl;
    std::string username;
    std::string hostname; // empty => resolved via gethostname(2)
    int pollIntervalSecs = 30;
};

// Builds a CoordinatorArgs from a config.conf [coordinator] section (see
// config.h). Returns false with `error` set if a required key is missing or
// a value doesn't parse.
bool buildCoordinatorArgs(const std::map<std::string, std::string>& section, CoordinatorArgs& out, std::string& error);

// Runs the claim/search/heartbeat/complete loop against the coordinator
// server, forever, until the process is killed. Returns a process exit code
// only on a startup error (e.g. registration failure); otherwise this never
// returns under normal operation.
int runCoordinator(const CoordinatorArgs& args);

#endif // NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
