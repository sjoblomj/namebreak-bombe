// The console entry point: parses --mode/--config, reads the config file
// and runs whichever mode it asks for, with a pause key listener on an
// interactive terminal. Everything it runs lives elsewhere - runSearch()
// (search.h) for bounded/continuous mode, runCoordinator()
// (coordinator_runner.h) for coordinator mode - so the tests and the Windows
// GUI link the same code without this file.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include "backends/backends.h"
#include "common/config.h"
#include "common/platform.h"
#include "common/version.h"
#include "engine/search.h"
#ifdef NAMEBREAK_WITH_NETWORK
#include "net/coordinator_runner.h"
#include "net/update_check.h"
#endif

// Toggled by pauseKeyListener below, polled by runSearch (via its
// pauseRequested parameter) so a pause takes effect between batches rather
// than needing to interrupt one mid-flight.
std::atomic<bool> g_paused{false};

// Coordinator mode only: "finish the current range, then pause" - toggled by
// the 'f' key, and cleared again by runCoordinator() once it has paused
// (see its finishRangeThenPause parameter). Only honoured in coordinator
// mode (g_coordinatorMode); local searches have no ranges to finish.
std::atomic<bool> g_finishRangeThenPause{false};
bool g_coordinatorMode = false;

// Coordinator mode only: set by requestQuitOrQuitNow to have runCoordinator()
// stop and return, telling the server how far the range in hand got (see
// its quitRequested parameter), instead of the process just dying.
std::atomic<bool> g_quitRequested{false};

// The 'f' key: see g_finishRangeThenPause. Turning it on while paused
// resumes the search too - that's the point of it: carry on, but only to the
// end of the range in hand.
void toggleFinishRangeThenPause() {
    if (!g_coordinatorMode) {
        printf("\n['f' only applies in coordinator mode - there are no ranges to finish in a local search]\n");
        return;
    }
    bool nowOn = !g_finishRangeThenPause.load(std::memory_order_relaxed);
    g_finishRangeThenPause.store(nowOn, std::memory_order_relaxed);
    if (!nowOn) {
        printf("\n[finish-then-pause off] new work will be claimed after the current range as usual (press 'f' to turn it back on)\n");
    } else if (g_paused.exchange(false, std::memory_order_relaxed)) {
        printf("\n[resumed until the current range is finished] then pausing before claiming new work "
               "(press 'f' to keep going afterwards, 'p' to pause now)\n");
    } else {
        printf("\n[finish-then-pause on] will pause once the current range is finished, before claiming new work "
               "(press 'f' again to cancel)\n");
    }
}

// Runs for the life of the process once main() starts it (only when stdin is
// an interactive terminal - see its call site), toggling g_paused on 'p'/
// 'P', the same key cgminer/xmrig and other long-running GPU compute tools
// already use for this. One key toggles both directions (like a media
// player's pause button) rather than separate pause/resume keys, since
// there's only ever one thing to remember. 'f'/'F' toggles
// g_finishRangeThenPause the same way - see toggleFinishRangeThenPause.
void pauseKeyListener() {
    for (;;) {
        int key = readKeypressBlocking();
        if (key < 0)
            return; // stdin closed - nothing left to listen for
        if (key == 'f' || key == 'F') {
            toggleFinishRangeThenPause();
            fflush(stdout);
            continue;
        }
        if (key != 'p' && key != 'P')
            continue;
        bool nowPaused = !g_paused.load(std::memory_order_relaxed);
        g_paused.store(nowPaused, std::memory_order_relaxed);
        bool finishing = g_coordinatorMode && g_finishRangeThenPause.load(std::memory_order_relaxed);
        if (nowPaused && finishing)
            printf("\n[paused] finishing the current batch; no new batches will start until resumed (Press 'p' to resume - still "
                   "set to pause again once the current range is finished. Press 'f' to keep going afterwards)\n");
        else if (nowPaused)
            printf("\n[paused] finishing the current batch; no new batches will start until resumed (press 'p' to resume)\n%s",
                   g_coordinatorMode ? "Press 'f' to finish the current range and then pause\n" : "");
        else
            printf(finishing ? "\n[resumed until the current range is finished] (press 'f' to keep going afterwards)\n" : "\n[resumed]\n");
        fflush(stdout);
    }
}

// Coordinator mode's quit: the first time, asks runCoordinator() to stop -
// it aborts the range in hand and tells the server how far it got, which
// takes a moment (one HTTP request) - and the next time quits right away,
// restoring the terminal first like the handler enableRawKeypressMode()
// installed would have. For SIGTERM always, for SIGINT when stdin isn't a
// terminal (no pausing then), and for the Ctrl+C that would otherwise quit
// - see handleSigintPauseOrQuit. Signal-handler context: see there.
void requestQuitOrQuitNow(int sig) {
    if (!g_quitRequested.exchange(true, std::memory_order_relaxed)) {
        static constexpr char kMsg[] =
            "\n[quitting] telling the coordinator how far the current range got (Ctrl+C again to quit right away)\n";
        writeStdoutSignalSafe(kMsg, sizeof(kMsg) - 1);
        return;
    }
    restoreKeypressMode();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

// The Windows CRT resets a signal to SIG_DFL before calling its handler
// (which the C standard allows; glibc doesn't), so each handler puts itself
// back first - otherwise the second Ctrl+C there would end the process
// outright, without telling the coordinator anything.
void handleQuitSignal(int sig) {
    std::signal(sig, handleQuitSignal);
    requestQuitOrQuitNow(sig);
}

// Installed as SIGINT's handler (overriding the plain restore-and-terminate
// one enableRawKeypressMode() already installed - see platform.h) so a
// reflexive first Ctrl+C pauses instead of losing the run outright: a pause
// is trivially undone (press 'p', or Ctrl+C once more), a quit isn't. A
// second Ctrl+C while already paused - however it got paused, this handler
// or the 'p' key - actually quits, restoring the terminal first exactly
// like the handler it replaced would have. In coordinator mode that quit is
// handleQuitSignal's, which tells the server first.
//
// Signal-handler context: only touches an atomic and async-signal-safe
// calls (write() via writeStdoutSignalSafe, tcsetattr via
// restoreKeypressMode, signal(), raise()) - no printf/fflush, which aren't
// guaranteed reentrant-safe if this signal interrupts another stdio call
// already in progress on the main or listener thread.
void handleSigintPauseOrQuit(int sig) {
    std::signal(sig, handleSigintPauseOrQuit); // see handleQuitSignal
    if (!g_paused.exchange(true, std::memory_order_relaxed)) {
        static constexpr char kMsg[] =
            "\n[paused] (Ctrl+C) finishing the current batch; no new batches will start until resumed "
            "(press 'p' to resume, or Ctrl+C again to quit)\n";
        writeStdoutSignalSafe(kMsg, sizeof(kMsg) - 1);
        return;
    }
    if (g_coordinatorMode) {
        requestQuitOrQuitNow(sig);
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
    std::string backends;
    for (const std::string& name : backendNames())
        backends += (backends.empty() ? "" : "|") + name;
    fprintf(stderr, "Usage: %s [--mode continuous|bounded|coordinator] [--config <file>] [--backend %s]\n"
                     "       %s -v|--version\n"
                     "Reads the given --config file (default: %s, in the current directory) for\n"
                     "everything else; --mode and --backend, if given, override that file's own\n"
                     "'mode = ...'/'backend = ...'. Without either, the first backend listed that\n"
                     "can run on this machine is used. -v/--version prints this build's version.\n",
            argv0, backends.c_str(), argv0, kDefaultConfigPath);
}

// For a first run: where the config file was looked for, and the smallest
// one that works.
void printMissingConfigHelp(const std::string& configPath) {
    std::error_code ec;
    std::filesystem::path where = std::filesystem::absolute(configPath, ec);
    fprintf(stderr,
            "No config file found at %s\n"
            "\n"
            "namebreak reads its settings from %s in the current directory, or from\n"
            "the file given with --config <file>. To join a shared search, this is all it\n"
            "needs:\n"
            "\n"
            "    mode = coordinator\n"
            "\n"
            "    [coordinator]\n"
            "    server_url = %s\n"
            "\n"
            "To search on your own instead (mode = bounded or continuous), see the [search]\n"
            "settings in README.md's \"Configuring\" section.\n",
            (ec ? configPath : where.string()).c_str(), kDefaultConfigPath, kDefaultServerUrl);
}

int main(int argc, char* argv[]) {
    // Three optional, order-independent flags: --mode and --backend,
    // overriding config.conf's own `mode = ...`/`backend = ...` (see
    // config.h), and --config <file>, overriding which file that config (and
    // everything else below) is read from. Everything else lives in the
    // config file itself.
    std::string modeOverride;
    std::string backendOverride;
    std::string configPath = kDefaultConfigPath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--version") {
            printf("namebreak %s\n", namebreakVersion());
            return 0;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--config requires a file path argument\n");
                return 1;
            }
            configPath = argv[++i];
        } else if (arg == "--backend") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--backend requires a backend name\n");
                return 1;
            }
            backendOverride = argv[++i];
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

    std::error_code existsError;
    if (!std::filesystem::exists(configPath, existsError)) {
        printMissingConfigHelp(configPath);
        return 1;
    }

    ConfigFile config;
    std::string error;
    if (!loadConfigFile(configPath, config, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    if (!backendOverride.empty())
        config.backend = backendOverride;

    std::string mode = !modeOverride.empty() ? modeOverride : config.mode;
    if (mode != "continuous" && mode != "bounded" && mode != "coordinator") {
        fprintf(stderr, "Unknown or missing mode '%s' - expected continuous, bounded, or coordinator "
                         "(set %s's 'mode = ...', or pass --mode <mode>)\n",
                mode.c_str(), configPath.c_str());
        return 1;
    }

#ifdef NAMEBREAK_WITH_NETWORK
    // Only ever a notice - nothing is downloaded, and nothing is said if
    // the lookup fails. Done before any work starts, so it isn't lost in
    // the search's output.
    if (config.checkForUpdates) {
        if (std::optional<std::string> notice = checkForNewerRelease())
            printf("%s\n", notice->c_str());
    }

    CoordinatorArgs cargs;
    if (mode == "coordinator") {
        if (!buildCoordinatorArgs(config, cargs, error)) {
            fprintf(stderr, "%s [coordinator]: %s\n", configPath.c_str(), error.c_str());
            return 1;
        }
        cargs.configPath = configPath;
        // Any username/hostname prompt has to happen here, before the key
        // listener below turns off echo and starts reading stdin itself -
        // see resolveMissingIdentity's doc comment.
        resolveMissingIdentity(cargs);
    }
#else
    if (mode == "coordinator") {
        fprintf(stderr, "This build of %s was compiled without networking support (rebuild with NAMEBREAK_NETWORK=ON to enable 'coordinator' mode).\n", argv[0]);
        return 1;
    }
#endif

    // Only when stdin is actually a terminal - a piped/redirected/absent
    // stdin (cron, systemd, ...) has no keypresses to listen for, and
    // enableRawKeypressMode() would just fail anyway.
    g_coordinatorMode = mode == "coordinator";
    bool interactive = isInteractiveTerminal() && enableRawKeypressMode();
    if (interactive) {
        printf("Press 'p' to pause/resume the search. Ctrl+C pauses too - press it again to quit.\n");
        if (g_coordinatorMode)
            printf("Press 'f' to finish the current range and then pause, before claiming new work.\n");
        std::thread(pauseKeyListener).detach();
        std::signal(SIGINT, handleSigintPauseOrQuit);
    }

#ifdef NAMEBREAK_WITH_NETWORK
    if (mode == "coordinator") {
        // Quitting tells the server how far the range in hand got - see
        // handleQuitSignal. A SIGTERM (systemd stopping a service, say) is a
        // quit too, and so is Ctrl+C without a terminal to pause from.
        std::signal(SIGTERM, handleQuitSignal);
        if (!interactive)
            std::signal(SIGINT, handleQuitSignal);
        int exitCode = runCoordinator(cargs, &g_paused, &g_quitRequested, nullptr, &g_finishRangeThenPause);
        restoreKeypressMode();
        return exitCode;
    }
#endif

    SearchRequest req;
    if (!buildSearchRequest(config, mode == "continuous", req, error)) {
        fprintf(stderr, "%s [search]: %s\n", configPath.c_str(), error.c_str());
        return 1;
    }

    std::unique_ptr<SearchBackend> backend = createBackend(config.backend, error);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    SearchResult result = runSearch(*backend, req, nullptr, nullptr, &g_paused);

    if (!result.ok) {
        fprintf(stderr, "%s\n", result.error.c_str());
        return 1;
    }
    return result.found ? 0 : 2;
}
