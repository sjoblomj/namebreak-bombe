#ifndef NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
#define NAMEBREAK_CUDA_COORDINATOR_RUNNER_H

// Entry point for `namebreak coordinator ...` - parses argv itself (flags,
// not the positional style continuous/bounded use: --server-url, --username,
// --hostname, --poll-interval-secs) and runs the claim/search/heartbeat/
// complete loop against the coordinator server, forever, until the process
// is killed. Returns a process exit code (1 on a startup/argument error;
// otherwise this never returns under normal operation).
int runCoordinator(int argc, char** argv);

#endif // NAMEBREAK_CUDA_COORDINATOR_RUNNER_H
