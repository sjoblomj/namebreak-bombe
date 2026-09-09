#ifndef NAMEBREAK_CUDA_PLATFORM_H
#define NAMEBREAK_CUDA_PLATFORM_H

#include <string>

// The handful of OS-specific facts the coordinator client needs - see
// platform.cpp for the Windows/POSIX implementation of each. Nothing else
// in this codebase (the CUDA search, libcurl, JSON/config parsing,
// std::thread-based orchestration) needs any OS-specific code at all.

// The machine's hostname, or "unknown-host" if it can't be determined.
std::string resolveHostname();

// The OS's notion of "the currently logged-in user", or "unknown-user" if
// it can't be determined.
std::string resolveUsername();

// True if stdin is an interactive terminal, as opposed to a pipe, redirect,
// or no controlling terminal at all (e.g. a cron job or systemd service).
bool isInteractiveTerminal();

#endif // NAMEBREAK_CUDA_PLATFORM_H
