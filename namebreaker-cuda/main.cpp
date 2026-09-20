// The console entry point: parses --mode/--config, reads the config file
// and runs whichever mode it asks for, with a pause key listener on an
// interactive terminal. Everything it runs lives elsewhere - runSearch()
// (search.h) for bounded/continuous mode, runCoordinator()
// (coordinator_runner.h) for coordinator mode - so the tests and the Windows
// GUI link the same code without this file.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>
#include "config.h"
#include "platform.h"
#include "search.h"
#ifdef NAMEBREAK_WITH_NETWORK
#include "coordinator_runner.h"
#endif

// Toggled by pauseKeyListener below, polled by runCudaBatch (via runSearch's
// pauseRequested parameter) so a pause takes effect between batches rather
// than needing to interrupt one mid-flight.
std::atomic<bool> g_paused{false};

// Runs for the life of the process once main() starts it (only when stdin is
// an interactive terminal - see its call site), toggling g_paused on 'p'/
// 'P', the same key cgminer/xmrig and other long-running GPU compute tools
// already use for this. One key toggles both directions (like a media
// player's pause button) rather than separate pause/resume keys, since
// there's only ever one thing to remember.
void pauseKeyListener() {
    for (;;) {
        int key = readKeypressBlocking();
        if (key < 0)
            return; // stdin closed - nothing left to listen for
        if (key != 'p' && key != 'P')
            continue;
        bool nowPaused = !g_paused.load(std::memory_order_relaxed);
        g_paused.store(nowPaused, std::memory_order_relaxed);
        printf(nowPaused ? "\n[paused] finishing the current batch; no new batches will start until resumed (press 'p' to resume)\n"
                          : "\n[resumed]\n");
        fflush(stdout);
    }
}

// Installed as SIGINT's handler (overriding the plain restore-and-terminate
// one enableRawKeypressMode() already installed - see platform.h) so a
// reflexive first Ctrl+C pauses instead of losing the run outright: a pause
// is trivially undone (press 'p', or Ctrl+C once more), a quit isn't. A
// second Ctrl+C while already paused - however it got paused, this handler
// or the 'p' key - actually quits, restoring the terminal first exactly
// like the handler it replaced would have.
//
// Signal-handler context: only touches an atomic and async-signal-safe
// calls (write() via writeStdoutSignalSafe, tcsetattr via
// restoreKeypressMode, signal(), raise()) - no printf/fflush, which aren't
// guaranteed reentrant-safe if this signal interrupts another stdio call
// already in progress on the main or listener thread.
void handleSigintPauseOrQuit(int sig) {
    if (!g_paused.exchange(true, std::memory_order_relaxed)) {
        static constexpr char kMsg[] =
            "\n[paused] (Ctrl+C) finishing the current batch; no new batches will start until resumed "
            "(press 'p' to resume, or Ctrl+C again to quit)\n";
        writeStdoutSignalSafe(kMsg, sizeof(kMsg) - 1);
        return;
    }
    static constexpr char kMsg[] = "\n[quitting] (Ctrl+C again)\n";
    writeStdoutSignalSafe(kMsg, sizeof(kMsg) - 1);
    restoreKeypressMode();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

// Usage/error message for a malformed command line - shared by every exit
// path in the argv-parsing loop below so they all describe the same syntax.
void printUsage(const char* argv0) {
    fprintf(stderr, "Usage: %s [--mode continuous|bounded|coordinator] [--config <file>]\n"
                     "Reads the given --config file (default: %s, in the current directory) for\n"
                     "everything else; --mode, if given, overrides that file's own 'mode = ...'.\n",
            argv0, kDefaultConfigPath);
}

int main(int argc, char* argv[]) {
    // Two optional, order-independent flags: --mode, overriding config.conf's
    // own `mode = ...` (see config.h), and --config <file>, overriding which
    // file that config (and everything else below) is read from. Everything
    // other than these two lives in the config file itself.
    std::string modeOverride;
    std::string configPath = kDefaultConfigPath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--config requires a file path argument\n");
                return 1;
            }
            configPath = argv[++i];
        } else if (arg == "--mode") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--mode requires an argument (continuous, bounded, or coordinator)\n");
                return 1;
            }
            modeOverride = argv[++i];
            if (modeOverride != "continuous" && modeOverride != "bounded" && modeOverride != "coordinator") {
                fprintf(stderr, "--mode must be one of continuous, bounded, or coordinator (got '%s')\n", modeOverride.c_str());
                return 1;
            }
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    ConfigFile config;
    std::string error;
    if (!loadConfigFile(configPath, config, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    std::string mode = !modeOverride.empty() ? modeOverride : config.mode;
    if (mode != "continuous" && mode != "bounded" && mode != "coordinator") {
        fprintf(stderr, "Unknown or missing mode '%s' - expected continuous, bounded, or coordinator "
                         "(set %s's 'mode = ...', or pass --mode <mode>)\n",
                mode.c_str(), configPath.c_str());
        return 1;
    }

    // Only when stdin is actually a terminal - a piped/redirected/absent
    // stdin (cron, systemd, ...) has no keypresses to listen for, and
    // enableRawKeypressMode() would just fail anyway.
    if (isInteractiveTerminal() && enableRawKeypressMode()) {
        printf("Press 'p' to pause/resume the search. Ctrl+C pauses too - press it again to quit.\n");
        std::thread(pauseKeyListener).detach();
        std::signal(SIGINT, handleSigintPauseOrQuit);
    }

#ifdef NAMEBREAK_WITH_NETWORK
    if (mode == "coordinator") {
        CoordinatorArgs cargs;
        if (!buildCoordinatorArgs(config.coordinator, cargs, error)) {
            fprintf(stderr, "%s [coordinator]: %s\n", configPath.c_str(), error.c_str());
            return 1;
        }
        cargs.configPath = configPath;
        return runCoordinator(cargs, &g_paused);
    }
#else
    if (mode == "coordinator") {
        fprintf(stderr, "This build of %s was compiled without networking support (rebuild without NETWORK=0 to enable 'coordinator' mode).\n", argv[0]);
        return 1;
    }
#endif

    SearchRequest req;
    if (!buildSearchRequest(config.search, mode == "continuous", req, error)) {
        fprintf(stderr, "%s [search]: %s\n", configPath.c_str(), error.c_str());
        return 1;
    }

    SearchResult result = runSearch(req, nullptr, nullptr, &g_paused);

    if (!result.ok) {
        fprintf(stderr, "%s\n", result.error.c_str());
        return 1;
    }
    return result.found ? 0 : 2;
}
