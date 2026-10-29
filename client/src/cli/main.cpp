// The console entry point: parses --mode/--config, reads the config file
// and runs whichever mode it asks for, with a pause key listener on an
// interactive terminal. Everything it runs lives elsewhere - runSearch()
// (search.h) for bounded/continuous mode, runDictionarySearch()
// (dictionary_search.h) for dictionary mode, runCoordinator()
// (coordinator_runner.h) for coordinator mode - so the tests and the Windows
// GUI link the same code without this file.

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include "backends/backends.h"
#include "common/autotune.h"
#include "common/config.h"
#include "common/platform.h"
#include "common/version.h"
#include "engine/dictionary_search.h"
#include "engine/meet_in_the_middle_search.h"
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
// - see handleSigintPauseOrQuit - and for Ctrl+Break on Windows.
// Signal-handler context: see there.
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
    fprintf(stderr, "Usage: %s [--mode <mode>] [--config <file>] [--backend %s]\n"
                     "       %s --autotune [--config <file>] [--backend opencl|metal]\n"
                     "       %s --list-gpus\n"
                     "       %s -v|--version\n"
                     "<mode> is continuous, bounded, coordinator, dictionary or meet-in-the-middle.\n"
                     "Reads the given --config file (default: %s, in the current directory) for\n"
                     "everything else; --mode and --backend, if given, override that file's own\n"
                     "'mode = ...'/'backend = ...'. Without either, the first backend listed that\n"
                     "can run on this machine is used (in dictionary or meet-in-the-middle mode,\n"
                     "that can search those). Coordinator mode searches on every GPU the config\n"
                     "file's [hardware] gpus selects at once; the other modes on the first of\n"
                     "them. --autotune measures the OpenCL or Metal backend's launch settings on\n"
                     "those GPUs and writes the fastest to the config file's [hardware] section.\n"
                     "--list-gpus lists the GPUs each backend can search on, by the numbers gpus\n"
                     "takes. -v/--version prints this build's version.\n",
            argv0, backends.c_str(), argv0, argv0, argv0, kDefaultConfigPath);
}

// --list-gpus: every backend's GPUs, by the numbers [hardware]'s gpus takes,
// and which of them `hardware` selects.
int listGpus(const HardwareConfig& hardware) {
    bool any = false;
    for (const std::string& name : backendNames()) {
        const std::vector<GpuDevice> devices = gpuDevices(name);
        if (devices.empty())
            continue;
        any = true;
        const std::vector<GpuDevice> selected = selectedGpus(devices, hardware);
        printf("%s:\n", name.c_str());
        for (const GpuDevice& device : devices) {
            const bool used = std::any_of(selected.begin(), selected.end(), [&](const GpuDevice& d) { return d.index == device.index; });
            const auto tunedFor = hardware.gpuNames.find(device.index);
            const std::string other = tunedFor != hardware.gpuNames.end() && tunedFor->second != device.name
                                          ? " - [hardware]'s gpu" + std::to_string(device.index) + " settings are for " + tunedFor->second
                                          : "";
            printf("  %d: %s%s%s%s\n", device.index, device.name.c_str(), device.integrated ? " (integrated)" : "", used ? "" : " - not selected",
                   other.c_str());
        }
    }
    if (!any)
        printf("No backend of this build finds a GPU here - it searches with the cpu backend.\n");
    return 0;
}

// A local search runs on one GPU - the first [hardware] selects - and says
// so when it selects more, which only coordinator mode searches on at once.
void noteOneGpuOnly(const SearchBackend& backend) {
    const std::vector<GpuDevice> selected = selectedGpus(gpuDevices(backend.name()), hardwareConfig());
    if (selected.size() > 1)
        printf("Searching on GPU %d (%s) only: a local search runs on one GPU - coordinator mode searches on all %zu that [hardware] "
               "selects.\n",
               selected.front().index, selected.front().name.c_str(), selected.size());
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
            "settings in docs/configuration.md, and for mode = dictionary or\n"
            "mode = meet-in-the-middle docs/dictionary-mode.md and\n"
            "docs/meet-in-the-middle.md.\n",
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
    bool autotune = false;
    bool listGpusRequested = false;
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
        } else if (arg == "--autotune") {
            autotune = true;
        } else if (arg == "--list-gpus") {
            listGpusRequested = true;
        } else if (arg == "--backend") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--backend requires a backend name\n");
                return 1;
            }
            backendOverride = argv[++i];
        } else if (arg == "--mode") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--mode requires an argument (continuous, bounded, coordinator, dictionary, or meet-in-the-middle)\n");
                return 1;
            }
            modeOverride = argv[++i];
            if (modeOverride != "continuous" && modeOverride != "bounded" && modeOverride != "coordinator" &&
                modeOverride != "dictionary" && modeOverride != "meet-in-the-middle") {
                fprintf(stderr, "--mode must be one of continuous, bounded, coordinator, dictionary, or meet-in-the-middle (got '%s')\n",
                        modeOverride.c_str());
                return 1;
            }
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    std::error_code existsError;
    const bool configExists = std::filesystem::exists(configPath, existsError);
    if (!configExists && !autotune && !listGpusRequested) {
        printMissingConfigHelp(configPath);
        return 1;
    }

    ConfigFile config;
    std::string error;
    // --autotune writes its section into a file it creates if need be.
    if (configExists && !loadConfigFile(configPath, config, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    if (!backendOverride.empty())
        config.backend = backendOverride;

    HardwareConfig hardware;
    if (!buildHardwareConfig(config, hardware, error)) {
        fprintf(stderr, "%s %s\n", configPath.c_str(), error.c_str());
        return 1;
    }
    setHardwareConfig(hardware);

    if (listGpusRequested)
        return listGpus(hardware);

    if (autotune) {
        // Ctrl+C stops the tuning between trials, without writing anything,
        // and lets it remove its temporary files.
        std::signal(SIGINT, [](int) { g_quitRequested = true; });
        std::string why;
        const std::string backend = autotuneBackend(config.backend, why);
        if (backend.empty()) {
            const std::vector<std::string> names = backendNames();
            const bool hasOpenCl = std::find(names.begin(), names.end(), "opencl") != names.end();
            fprintf(stderr, "%s\n%s", why.c_str(),
                    why.find(kAutotuneScriptUrl) != std::string::npos && hasOpenCl
                        ? "(--autotune --backend opencl tunes this build's OpenCL backend instead, for searches run with --backend opencl.)\n"
                        : "");
            return 1;
        }
        const std::vector<GpuDevice> devices = autotuneDevices(backend, why);
        if (devices.empty()) {
            fprintf(stderr, "%s\n", why.c_str());
            return 1;
        }
        printf("Ctrl+C stops the tuning.\n");
        return runAutotune(configPath, backend, devices, 2.0, &g_quitRequested);
    }

    std::string mode = !modeOverride.empty() ? modeOverride : config.mode;
    if (mode != "continuous" && mode != "bounded" && mode != "coordinator" && mode != "dictionary" && mode != "meet-in-the-middle") {
        fprintf(stderr, "Unknown or missing mode '%s' - expected continuous, bounded, coordinator, dictionary, or meet-in-the-middle "
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
#ifdef SIGBREAK
        // Windows has no SIGTERM to send: a console program is stopped with
        // Ctrl+Break (GenerateConsoleCtrlEvent), which arrives as SIGBREAK.
        std::signal(SIGBREAK, handleQuitSignal);
#endif
        if (!interactive)
            std::signal(SIGINT, handleQuitSignal);
        int exitCode = runCoordinator(cargs, &g_paused, &g_quitRequested, nullptr, &g_finishRangeThenPause);
        restoreKeypressMode();
        return exitCode;
    }
#endif

    if (mode == "dictionary") {
        DictionaryRequest dictReq;
        std::vector<std::string> warnings;
        bool built = buildDictionaryRequest(config, dictReq, warnings, error);
        // A list full of unusable lines would bury everything else.
        constexpr size_t kMaxWarnings = 20;
        for (size_t i = 0; i < warnings.size() && i < kMaxWarnings; ++i)
            fprintf(stderr, "warning: %s\n", warnings[i].c_str());
        if (warnings.size() > kMaxWarnings)
            fprintf(stderr, "warning: ... and %zu more lines skipped\n", warnings.size() - kMaxWarnings);
        if (!built) {
            fprintf(stderr, "%s [dictionary]: %s\n", configPath.c_str(), error.c_str());
            return 1;
        }
        std::unique_ptr<SearchBackend> backend = createDictionaryBackend(config.backend, error);
        if (!backend) {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        noteOneGpuOnly(*backend);
        DictionaryResult result = runDictionarySearch(*backend, dictReq, nullptr, nullptr, &g_paused);
        if (!result.ok) {
            fprintf(stderr, "%s\n", result.error.c_str());
            return 1;
        }
        return result.found ? 0 : 2;
    }

    if (mode == "meet-in-the-middle") {
        MeetInTheMiddleRequest meetInTheMiddleReq;
        if (!buildMeetInTheMiddleRequest(config, meetInTheMiddleReq, error)) {
            fprintf(stderr, "%s [meet-in-the-middle]: %s\n", configPath.c_str(), error.c_str());
            return 1;
        }
        std::unique_ptr<SearchBackend> backend = createMeetInTheMiddleBackend(config.backend, error);
        if (!backend) {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        noteOneGpuOnly(*backend);
        MeetInTheMiddleResult result = runMeetInTheMiddleSearch(*backend, meetInTheMiddleReq, nullptr, nullptr, &g_paused);
        if (!result.ok) {
            fprintf(stderr, "%s\n", result.error.c_str());
            return 1;
        }
        return result.found ? 0 : 2;
    }

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
    noteOneGpuOnly(*backend);
    SearchResult result = runSearch(*backend, req, nullptr, nullptr, &g_paused);

    if (!result.ok) {
        fprintf(stderr, "%s\n", result.error.c_str());
        return 1;
    }
    return result.found ? 0 : 2;
}
