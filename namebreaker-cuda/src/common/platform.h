#ifndef NAMEBREAK_CUDA_PLATFORM_H
#define NAMEBREAK_CUDA_PLATFORM_H

#include <cstddef>
#include <string>

// The handful of OS-specific facts this program needs - the hostname/username
// the coordinator client reports, and the terminal/keypress handling behind
// the pause key - see platform.cpp for the Windows/POSIX implementation of
// each. Nothing else in this codebase (the CUDA search, libcurl, JSON/config
// parsing, std::thread-based orchestration) needs any OS-specific code at all.
// Built in every configuration, including NETWORK=0 (see the Makefile).

// The machine's hostname, or "unknown-host" if it can't be determined.
std::string resolveHostname();

// The OS's notion of "the currently logged-in user", or "unknown-user" if
// it can't be determined.
std::string resolveUsername();

// True if stdin is an interactive terminal, as opposed to a pipe, redirect,
// or no controlling terminal at all (e.g. a cron job or systemd service).
bool isInteractiveTerminal();

// Enables raw, unbuffered, no-echo single-keypress reading on stdin (used by
// the pause/resume key listener - see src/cli/main.cpp's main()). POSIX: puts
// the terminal into "cbreak" mode (ICANON/ECHO off) but deliberately leaves
// ISIG on, so Ctrl+C/Ctrl+Z keep working exactly as normal; also installs a
// best-effort SIGINT/SIGTERM handler that restores the terminal before the
// process actually terminates; without it, a plain Ctrl+C during
// coordinator mode's infinite loop would kill the process while stdin is
// still in raw/no-echo mode, leaving the user's shell needing `stty sane` to
// recover. A caller is free to install its own SIGINT handler afterward to
// customize just that signal's behavior further (see src/cli/main.cpp's main(),
// which does this to make a first Ctrl+C pause instead of quit) - SIGTERM's
// handler is left as this function installed it either way, since SIGTERM
// should always just mean "terminate", never "pause". Windows: a no-op that
// always returns true - _getch() (used by readKeypressBlocking below)
// already reads unbuffered and without echo, so there's no mode to change or
// restore. Returns false if stdin isn't a terminal or the underlying OS call
// fails - callers should treat single-keypress reading as simply
// unavailable then, not fall back to it.
bool enableRawKeypressMode();

// Reads one keypress, blocking until one is available. Only meaningful after
// a successful enableRawKeypressMode(). Returns -1 on EOF or a read error
// (e.g. stdin closed/redirected out from under a long-running process).
int readKeypressBlocking();

// Restores whatever enableRawKeypressMode() changed, and is what its
// SIGINT/SIGTERM handler calls before letting the process actually
// terminate. Safe to call even if enableRawKeypressMode() was never called
// or failed - a no-op then.
void restoreKeypressMode();

// Writes `len` bytes straight to stdout via the OS write call, bypassing
// stdio's buffered printf/fputs/etc. entirely - the only safe way to print
// from within a signal handler (a signal can interrupt the process mid-way
// through some other printf call already holding stdio's internal lock;
// printf/fflush aren't guaranteed reentrant against that, but a raw write()
// is). `len` excludes any null terminator. Best-effort: failures are
// silently ignored, since a signal handler has nothing useful to do with an
// error here anyway.
void writeStdoutSignalSafe(const char* msg, std::size_t len);

#endif // NAMEBREAK_CUDA_PLATFORM_H
