// A little native Win32 GUI for namebreak's coordinator mode - built only on
// Windows (see the Makefile's `namebreak-gui` target), linking the same
// search/coordinator/config/platform sources the console `namebreak` binary
// does. namebreak.cu is compiled with -DNAMEBREAK_NO_MAIN there (the same
// flag the test executables already use, see namebreak.cu's own comment on
// it) so this file's own main() - not namebreak.cu's console one - is the
// process entry point. Built with /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup
// (see the Makefile) so the process has no console, but the CRT still calls
// this ordinary main() rather than requiring a WinMain() - which also means
// this GUI can honor the same --config <file> the CLI does (kDefaultConfigPath,
// config.h). There's no --mode: this GUI only ever runs coordinator mode.

#ifndef _WIN32
#error "gui_win32.cpp is Windows-only - see the Makefile's namebreak-gui target"
#endif

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Windows 7 floor - guarantees PBS_MARQUEE and other comctl32 v6 behavior used below
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>

// PBS_MARQUEE (the "no work claimed yet" progress-bar state below) needs
// comctl32 v6, which Windows only loads side-by-side instead of the old
// system v5 copy when the process declares that dependency - normally done
// via a separate .manifest/.rc file. Embedding it as a linker directive
// instead keeps the whole GUI to this one .cpp, no resource compiler step.
#pragma comment(linker, \
    "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "coordinator_runner.h"
#include "platform.h"
#include "protocol.h"

namespace {

// ---------------------------------------------------------------------
// Window/control/message ids
// ---------------------------------------------------------------------

constexpr const char* kWindowClassName = "NamebreakGuiWindow";
constexpr const char* kSetupClassName = "NamebreakSetupDialog";
constexpr const char* kWindowTitle = "namebreak";
constexpr int kWindowWidth = 500;
constexpr int kWindowHeight = 520;

constexpr UINT_PTR kUiTimerId = 1;
constexpr UINT kUiTimerIntervalMs = 1000;
constexpr UINT WM_APP_TRAYICON = WM_APP + 1;
constexpr UINT WM_APP_COORDINATOR_STOPPED = WM_APP + 2;

constexpr int kIdPauseButton = 101;
constexpr int kIdQuitButton = 102;
constexpr int kIdTrayShow = 201;
constexpr int kIdTrayPause = 202;
constexpr int kIdTrayQuit = 203;
constexpr int kIdSetupOk = 301;
constexpr int kIdSetupCancel = 302;

constexpr size_t kMaxMatchLines = 100;

// ---------------------------------------------------------------------
// Cross-thread state - written by CoordinatorCallbacks on the worker
// thread, read by the UI thread's once-a-second timer. No Win32 calls ever
// happen inside the callbacks themselves (see coordinator_runner.h's own
// warning on this) - they only ever touch this mutex-protected struct.
// ---------------------------------------------------------------------

struct SharedStatus {
    std::mutex mutex;
    std::string targetName;
    std::string outputFilePath;
    std::chrono::steady_clock::time_point rangeClaimedAt{};
    double leaseSeconds = 0.0;
    bool hasActiveRange = false;
    // Set true by onRangeFinished, consumed (cleared back to false) the next
    // time the UI timer reads this struct - a one-shot "a range just ended"
    // edge, not a level, so the progress bar only snaps to 100% once.
    bool rangeJustFinished = false;
    std::string statusText = "Starting...";
};

SharedStatus g_status;

HWND g_hwndMain = nullptr;
HWND g_hwndTargetLabel = nullptr;
HWND g_hwndStatusLabel = nullptr;
HWND g_hwndProgress = nullptr;
HWND g_hwndMatches = nullptr;
HWND g_hwndPauseButton = nullptr;
HWND g_hwndQuitButton = nullptr;

NOTIFYICONDATAA g_trayIcon{};
bool g_trayIconAdded = false;

std::atomic<bool> g_pauseRequested{false};
std::atomic<bool> g_quitRequested{false};
std::thread g_workerThread;
// UI-thread-only (never touched from the worker thread): true once the user
// has confirmed Quit, so WM_APP_COORDINATOR_STOPPED can tell "the loop
// stopped because we asked it to" apart from "the loop stopped on its own"
// (e.g. registration failed at startup) and only alarm the user for the
// latter.
bool g_quitting = false;

// UI-thread-only pause-duration bookkeeping for the progress bar (see
// updateUiFromSharedState/togglePause) - elapsed-since-claim is otherwise
// pure wall-clock time, so without this the bar would jump forward the
// instant a pause ends to "catch up" on however long the pause lasted,
// instead of resuming from wherever it actually left off.
std::chrono::steady_clock::duration g_pausedDurationThisRange{};
std::chrono::steady_clock::time_point g_pauseStartedAt{};
bool g_currentlyPaused = false;
std::chrono::steady_clock::time_point g_lastRangeClaimedAt{};

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

std::string trimCopy(const char* s) {
    std::string str(s);
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// Reads up to the last `maxLines` lines of `path` (oldest first). This file
// only ever grows a few KB to low MB over a session (one line per Hash-A
// hit), so re-reading it whole on every ~1s tick is cheap enough not to need
// a real seek-from-end tail implementation.
std::vector<std::string> readLastLines(const std::string& path, size_t maxLines) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    if (!in)
        return lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    if (lines.size() > maxLines)
        lines.erase(lines.begin(), lines.begin() + (lines.size() - maxLines));
    return lines;
}

// ---------------------------------------------------------------------
// Config readiness / first-run setup
// ---------------------------------------------------------------------

bool configHasServerUrl(const ConfigFile& config) {
    auto it = config.coordinator.find("server_url");
    return it != config.coordinator.end() && !it->second.empty();
}

// appendKeyToConfigSection (config.h) requires the file AND the [coordinator]
// header to already exist - true for a config.conf a CLI run already
// created, but not for one this GUI is creating from scratch, and not for
// one that only ever had a [search] section (e.g. hand-written for local
// bounded/continuous use). Appends a bare "[coordinator]" section header to
// the end of the file if one isn't already present; a no-op (returns true)
// if it already is.
bool ensureCoordinatorSection(const std::string& path, std::string& error) {
    {
        std::ifstream in(path);
        if (!in) {
            error = "cannot open " + path;
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            size_t start = line.find_first_not_of(" \t\r\n");
            size_t end = line.find_last_not_of(" \t\r\n");
            std::string trimmed = (start == std::string::npos) ? "" : line.substr(start, end - start + 1);
            if (trimmed == "[coordinator]")
                return true;
        }
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        error = "cannot append to " + path;
        return false;
    }
    out << "\n[coordinator]\n";
    return true;
}

// Makes sure `path`'s [coordinator] section has server_url/username/hostname
// set to the given values, creating the file (and/or the section) from
// scratch if needed. Values the caller passes empty (username/hostname,
// which are optional - see CoordinatorArgs) are simply left alone. Mirrors
// what the CLI's resolveMissingIdentity (coordinator_runner.cpp) already
// does for username/hostname, extended to also cover server_url and to work
// even when no config.conf exists yet at all.
bool ensureCoordinatorConfig(const std::string& path, const std::string& username, const std::string& hostname,
                              const std::string& serverUrl, std::string& error) {
    std::ifstream probe(path);
    bool fileExists = probe.good();
    probe.close();

    if (!fileExists) {
        std::ofstream out(path, std::ios::trunc);
        if (!out) {
            error = "cannot create " + path;
            return false;
        }
        out << "mode = coordinator\n\n[coordinator]\n";
        out << "server_url = " << serverUrl << "\n";
        if (!username.empty())
            out << "username = " << username << "\n";
        if (!hostname.empty())
            out << "hostname = " << hostname << "\n";
        return true;
    }

    if (!ensureCoordinatorSection(path, error))
        return false;

    ConfigFile existing;
    std::string loadErr;
    if (!loadConfigFile(path, existing, loadErr)) {
        error = loadErr;
        return false;
    }
    // Only ever adds a key that's missing or empty - never overwrites a
    // value the user already had configured that they simply didn't touch
    // in the setup dialog.
    auto hasNonEmpty = [&](const std::string& key) {
        auto it = existing.coordinator.find(key);
        return it != existing.coordinator.end() && !it->second.empty();
    };
    auto ensureKey = [&](const std::string& key, const std::string& value) {
        if (value.empty() || hasNonEmpty(key))
            return true;
        return appendKeyToConfigSection(path, "coordinator", key, value);
    };
    if (!ensureKey("server_url", serverUrl)) {
        error = "failed to write server_url to " + path;
        return false;
    }
    if (!ensureKey("username", username)) {
        error = "failed to write username to " + path;
        return false;
    }
    if (!ensureKey("hostname", hostname)) {
        error = "failed to write hostname to " + path;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// First-run setup dialog - a hand-rolled modal window (CreateWindowEx +
// its own nested message loop) rather than a .rc dialog template, so the
// whole GUI stays a single .cpp with no resource-compiler build step. Only
// ever shown once, before the main window/worker thread exist, so there's
// no other message loop running concurrently to conflict with.
// ---------------------------------------------------------------------

struct SetupDialogState {
    HWND hwndUsername = nullptr;
    HWND hwndHostname = nullptr;
    HWND hwndServerUrl = nullptr;
    std::string username;
    std::string hostname;
    std::string serverUrl;
    bool accepted = false;
    bool done = false;
};

LRESULT CALLBACK SetupDialogWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<SetupDialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_COMMAND:
            if (!state)
                break;
            switch (LOWORD(wParam)) {
                case kIdSetupOk: {
                    char buf[512];
                    GetWindowTextA(state->hwndUsername, buf, sizeof(buf));
                    state->username = trimCopy(buf);
                    GetWindowTextA(state->hwndHostname, buf, sizeof(buf));
                    state->hostname = trimCopy(buf);
                    GetWindowTextA(state->hwndServerUrl, buf, sizeof(buf));
                    state->serverUrl = trimCopy(buf);
                    if (state->serverUrl.empty()) {
                        MessageBoxA(hwnd, "Server URL is required.", "namebreak setup", MB_OK | MB_ICONWARNING);
                        return 0;
                    }
                    state->accepted = true;
                    state->done = true;
                    DestroyWindow(hwnd);
                    return 0;
                }
                case kIdSetupCancel:
                    state->accepted = false;
                    state->done = true;
                    DestroyWindow(hwnd);
                    return 0;
            }
            return 0;
        case WM_CLOSE:
            if (state) {
                state->accepted = false;
                state->done = true;
            }
            DestroyWindow(hwnd);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

// Shows the setup form pre-filled with `username`/`hostname`/`serverUrl`;
// on OK, overwrites all three with what the user confirmed/entered and
// returns true. Returns false (leaving the three untouched) on Cancel/close.
bool showSetupDialog(HINSTANCE hInstance, std::string& username, std::string& hostname, std::string& serverUrl) {
    static bool classRegistered = false;
    if (!classRegistered) {
        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = SetupDialogWndProc;
        wc.hInstance = hInstance;
        wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
        wc.lpszClassName = kSetupClassName;
        RegisterClassExA(&wc);
        classRegistered = true;
    }

    SetupDialogState state;
    state.username = username;
    state.hostname = hostname;
    state.serverUrl = serverUrl;

    HWND hwndDialog = CreateWindowExA(WS_EX_DLGMODALFRAME, kSetupClassName, "namebreak - first-run setup",
                                       WS_POPUP | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT, 440, 220, nullptr, nullptr,
                                       hInstance, nullptr);
    if (!hwndDialog)
        return false;
    SetWindowLongPtr(hwndDialog, GWLP_USERDATA, (LONG_PTR) &state);

    auto makeLabel = [&](const char* text, int y) {
        CreateWindowExA(0, "STATIC", text, WS_CHILD | WS_VISIBLE, 10, y, 140, 18, hwndDialog, nullptr, hInstance, nullptr);
    };
    // ES_AUTOHSCROLL matters here, not just cosmetically: a single-line EDIT
    // control without it refuses to accept any more typed/pasted characters
    // than fit in its visible width at once (no scrolling, no overflow) -
    // it's not a length *limit* so much as a hard stop once the box looks
    // full. Long values (e.g. a real server URL) would otherwise silently
    // get truncated right there, with no error and no obvious reason why.
    constexpr DWORD kEditStyle = WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL;
    makeLabel("Username:", 15);
    state.hwndUsername =
        CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", state.username.c_str(), kEditStyle, 160, 12, 250, 22, hwndDialog, nullptr, hInstance,
                         nullptr);
    makeLabel("Hostname:", 45);
    state.hwndHostname =
        CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", state.hostname.c_str(), kEditStyle, 160, 42, 250, 22, hwndDialog, nullptr, hInstance,
                         nullptr);
    makeLabel("Server URL:", 75);
    state.hwndServerUrl =
        CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", state.serverUrl.c_str(), kEditStyle, 160, 72, 250, 22, hwndDialog, nullptr, hInstance,
                         nullptr);
    CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 150, 120, 80, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupOk,
                     hInstance, nullptr);
    CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE, 240, 120, 80, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupCancel, hInstance,
                     nullptr);

    ShowWindow(hwndDialog, SW_SHOW);
    UpdateWindow(hwndDialog);
    SetForegroundWindow(hwndDialog);

    MSG msg;
    while (!state.done && GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (state.accepted) {
        username = state.username;
        hostname = state.hostname;
        serverUrl = state.serverUrl;
    }
    return state.accepted;
}

// Loads `configPath`, running the setup dialog first if it doesn't exist yet
// or is missing server_url (the one truly required [coordinator] key - see
// buildCoordinatorArgs, coordinator_runner.cpp). Returns false (having
// already reported why, via MessageBox) if the user cancelled setup or a
// config error couldn't be resolved.
bool prepareConfig(HINSTANCE hInstance, const std::string& configPath, CoordinatorArgs& outArgs) {
    ConfigFile config;
    std::string error;
    bool ready = loadConfigFile(configPath, config, error) && configHasServerUrl(config);

    if (!ready) {
        std::string username = resolveUsername();
        std::string hostname = resolveHostname();
        std::string serverUrl;
        if (!showSetupDialog(hInstance, username, hostname, serverUrl))
            return false; // user cancelled - nothing to report

        std::string writeError;
        if (!ensureCoordinatorConfig(configPath, username, hostname, serverUrl, writeError)) {
            MessageBoxA(nullptr, ("Failed to write " + configPath + ": " + writeError).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
        if (!loadConfigFile(configPath, config, error)) {
            MessageBoxA(nullptr, ("Failed to reload " + configPath + " after setup: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
    }

    if (!buildCoordinatorArgs(config.coordinator, outArgs, error)) {
        MessageBoxA(nullptr, (configPath + " [coordinator]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
        return false;
    }
    outArgs.configPath = configPath;
    return true;
}

// ---------------------------------------------------------------------
// Coordinator worker thread
// ---------------------------------------------------------------------

void onRangeClaimed(const ClaimResponse& claim, const std::string& outputFilePath) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.targetName = claim.targetName;
    g_status.outputFilePath = outputFilePath;
    g_status.rangeClaimedAt = std::chrono::steady_clock::now();
    g_status.leaseSeconds = (double) claim.leaseSeconds;
    g_status.hasActiveRange = true;
    g_status.rangeJustFinished = false;
    g_status.statusText = "Searching target: " + claim.targetName;
}

void onRangeFinished(bool found) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.hasActiveRange = false;
    g_status.rangeJustFinished = true;
    g_status.statusText = found ? ("Match found in target: " + g_status.targetName) : "Range finished, no match";
}

void onStatus(const std::string& text) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.statusText = text;
}

void workerThreadMain(CoordinatorArgs args) {
    CoordinatorCallbacks callbacks;
    callbacks.onRangeClaimed = onRangeClaimed;
    callbacks.onRangeFinished = onRangeFinished;
    callbacks.onStatus = onStatus;
    runCoordinator(args, &g_pauseRequested, &g_quitRequested, &callbacks);
    // Marshal back onto the UI thread rather than touching any Win32 API
    // from here - see WM_APP_COORDINATOR_STOPPED in MainWndProc.
    PostMessage(g_hwndMain, WM_APP_COORDINATOR_STOPPED, 0, 0);
}

// ---------------------------------------------------------------------
// Tray icon
// ---------------------------------------------------------------------

void addTrayIcon(HWND hwnd) {
    ZeroMemory(&g_trayIcon, sizeof(g_trayIcon));
    g_trayIcon.cbSize = sizeof(g_trayIcon);
    g_trayIcon.hWnd = hwnd;
    g_trayIcon.uID = 1;
    g_trayIcon.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_trayIcon.uCallbackMessage = WM_APP_TRAYICON;
    g_trayIcon.hIcon = LoadIconA(nullptr, IDI_APPLICATION);
    lstrcpynA(g_trayIcon.szTip, kWindowTitle, sizeof(g_trayIcon.szTip));
    Shell_NotifyIconA(NIM_ADD, &g_trayIcon);
    g_trayIconAdded = true;
}

void removeTrayIcon() {
    if (g_trayIconAdded) {
        Shell_NotifyIconA(NIM_DELETE, &g_trayIcon);
        g_trayIconAdded = false;
    }
}

void showMainWindow(HWND hwnd) {
    ShowWindow(hwnd, SW_SHOW);
    ShowWindow(hwnd, SW_RESTORE);
    SetForegroundWindow(hwnd);
}

void showTrayContextMenu(HWND hwnd) {
    POINT pt;
    GetCursorPos(&pt);
    HMENU menu = CreatePopupMenu();
    AppendMenuA(menu, MF_STRING, kIdTrayShow, "Show GUI");
    AppendMenuA(menu, MF_STRING, kIdTrayPause, g_pauseRequested.load(std::memory_order_relaxed) ? "Resume" : "Pause");
    AppendMenuA(menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(menu, MF_STRING, kIdTrayQuit, "Quit");
    // Standard idiom (see Shell_NotifyIcon's docs) so the popup dismisses
    // correctly if the user clicks away from it instead of choosing an item.
    SetForegroundWindow(hwnd);
    TrackPopupMenu(menu, TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, nullptr);
    PostMessage(hwnd, WM_NULL, 0, 0);
    DestroyMenu(menu);
}

// ---------------------------------------------------------------------
// Main window
// ---------------------------------------------------------------------

void togglePause() {
    bool newState = !g_pauseRequested.load(std::memory_order_relaxed);
    g_pauseRequested.store(newState, std::memory_order_relaxed);
    SetWindowTextA(g_hwndPauseButton, newState ? "Resume" : "Pause");

    auto now = std::chrono::steady_clock::now();
    if (newState) {
        g_pauseStartedAt = now;
        g_currentlyPaused = true;
    } else if (g_currentlyPaused) {
        // Folds the segment that just ended into the running total, so the
        // elapsed-time calculation in updateUiFromSharedState continues to
        // exclude it from here on.
        g_pausedDurationThisRange += now - g_pauseStartedAt;
        g_currentlyPaused = false;
    }
}

void requestQuit(HWND hwnd) {
    if (g_quitting)
        return; // already shutting down - ignore a repeat click
    int result = MessageBoxA(hwnd, "Quit namebreak? This stops searching now; the current range's lease will simply expire and get reassigned.",
                              "Confirm Quit", MB_YESNO | MB_ICONQUESTION);
    if (result != IDYES)
        return;
    g_quitting = true;
    EnableWindow(g_hwndPauseButton, FALSE);
    EnableWindow(g_hwndQuitButton, FALSE);
    SetWindowTextA(g_hwndStatusLabel, "Status: shutting down...");
    // Clearing pause too (rather than leaving runCoordinator's pause-wait
    // spin to notice quitRequested on its own, which it does - see
    // coordinator_runner.cpp) just keeps this state visually consistent
    // regardless of whether the user had paused first.
    g_pauseRequested.store(false, std::memory_order_relaxed);
    g_quitRequested.store(true, std::memory_order_relaxed);
}

// Re-reads the shared coordinator-status struct and the current matches
// file, and refreshes every control accordingly - the only place any of
// this GUI's controls are actually touched, always from the UI thread's own
// WM_TIMER tick (see SharedStatus's own comment on why).
void updateUiFromSharedState() {
    std::string targetName, outputFilePath, statusText;
    bool hasActiveRange, rangeJustFinished;
    std::chrono::steady_clock::time_point rangeClaimedAt;
    double leaseSeconds;
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        targetName = g_status.targetName;
        outputFilePath = g_status.outputFilePath;
        statusText = g_status.statusText;
        hasActiveRange = g_status.hasActiveRange;
        rangeJustFinished = g_status.rangeJustFinished;
        rangeClaimedAt = g_status.rangeClaimedAt;
        leaseSeconds = g_status.leaseSeconds;
        g_status.rangeJustFinished = false; // one-shot edge, consumed here
    }

    bool paused = g_pauseRequested.load(std::memory_order_relaxed);

    if (rangeClaimedAt != g_lastRangeClaimedAt) {
        // A new range (or the first one) - start this range's pause total
        // fresh rather than carrying over whatever a previous range (or
        // idle time before any range was claimed) accumulated.
        g_lastRangeClaimedAt = rangeClaimedAt;
        g_pausedDurationThisRange = std::chrono::steady_clock::duration::zero();
        if (g_currentlyPaused)
            g_pauseStartedAt = std::chrono::steady_clock::now();
    }

    SetWindowTextA(g_hwndTargetLabel, (targetName.empty() ? "Target: (none yet)" : "Target: " + targetName).c_str());
    SetWindowTextA(g_hwndStatusLabel, ("Status: " + statusText + (paused ? " (paused)" : "")).c_str());

    if (hasActiveRange && leaseSeconds > 0) {
        LONG_PTR style = GetWindowLongPtr(g_hwndProgress, GWL_STYLE);
        if (style & PBS_MARQUEE) {
            SendMessage(g_hwndProgress, PBM_SETMARQUEE, FALSE, 0);
            SetWindowLongPtr(g_hwndProgress, GWL_STYLE, style & ~PBS_MARQUEE);
        }
        // Frozen (left wherever it last was) rather than recomputed while
        // paused - elapsed-since-claim is otherwise pure wall-clock time
        // (see below), which keeps ticking whether or not the search is
        // actually allowed to run, so leaving it live during a pause would
        // make Pause look like it does nothing at all.
        if (!paused) {
            // No live "candidates processed so far" counter exists anywhere,
            // client or server (see coordinator_runner.h's CoordinatorCallbacks
            // doc comment) - leaseSeconds (already computed server-side from
            // this user's own historical rate) stands in as a rough time
            // budget for the claimed range instead. Capped short of 100% since
            // a range can legitimately finish faster or slower than its lease.
            // g_pausedDurationThisRange is subtracted out so resuming from a
            // pause picks up from wherever it left off, instead of jumping
            // forward to "catch up" on wall-clock time that passed while
            // paused (see togglePause).
            double wallClockElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - rangeClaimedAt).count();
            double pausedSecs = std::chrono::duration<double>(g_pausedDurationThisRange).count();
            double elapsed = std::max(wallClockElapsed - pausedSecs, 0.0);
            double fraction = std::min(elapsed / leaseSeconds, 0.99);
            SendMessage(g_hwndProgress, PBM_SETPOS, (WPARAM) (fraction * 100.0), 0);
        }
    } else if (rangeJustFinished) {
        SendMessage(g_hwndProgress, PBM_SETPOS, 100, 0);
    } else {
        LONG_PTR style = GetWindowLongPtr(g_hwndProgress, GWL_STYLE);
        if (!(style & PBS_MARQUEE)) {
            SetWindowLongPtr(g_hwndProgress, GWL_STYLE, style | PBS_MARQUEE);
            SendMessage(g_hwndProgress, PBM_SETMARQUEE, TRUE, 100);
        }
    }

    if (!outputFilePath.empty()) {
        std::vector<std::string> lines = readLastLines(outputFilePath, kMaxMatchLines);
        std::string joined;
        for (size_t i = 0; i < lines.size(); ++i) {
            joined += lines[i];
            if (i + 1 < lines.size())
                joined += "\r\n";
        }
        SetWindowTextA(g_hwndMatches, joined.c_str());
        SendMessage(g_hwndMatches, EM_SETSEL, (WPARAM) -1, (LPARAM) -1);
        SendMessage(g_hwndMatches, EM_SCROLLCARET, 0, 0);
    }
}

void createChildControls(HWND hwnd, HINSTANCE hInstance) {
    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    auto setFont = [&](HWND child) { SendMessage(child, WM_SETFONT, (WPARAM) font, TRUE); };

    g_hwndTargetLabel = CreateWindowExA(0, "STATIC", "Target: (none yet)", WS_CHILD | WS_VISIBLE, 10, 10, 470, 20, hwnd, nullptr,
                                         hInstance, nullptr);
    g_hwndStatusLabel =
        CreateWindowExA(0, "STATIC", "Status: starting...", WS_CHILD | WS_VISIBLE, 10, 35, 470, 20, hwnd, nullptr, hInstance, nullptr);
    g_hwndProgress = CreateWindowExA(0, PROGRESS_CLASSA, nullptr, WS_CHILD | WS_VISIBLE | PBS_MARQUEE, 10, 60, 470, 22, hwnd, nullptr,
                                      hInstance, nullptr);
    SendMessage(g_hwndProgress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    SendMessage(g_hwndProgress, PBM_SETMARQUEE, TRUE, 100);

    CreateWindowExA(0, "STATIC", "Hash A matches (most recent last):", WS_CHILD | WS_VISIBLE, 10, 95, 300, 18, hwnd, nullptr, hInstance,
                     nullptr);
    g_hwndMatches = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
                                     WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL, 10, 115, 470, 280,
                                     hwnd, nullptr, hInstance, nullptr);

    g_hwndPauseButton = CreateWindowExA(0, "BUTTON", "Pause", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 10, 405, 150, 30, hwnd,
                                         (HMENU) (INT_PTR) kIdPauseButton, hInstance, nullptr);
    g_hwndQuitButton = CreateWindowExA(0, "BUTTON", "Quit", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 330, 405, 150, 30, hwnd,
                                        (HMENU) (INT_PTR) kIdQuitButton, hInstance, nullptr);

    for (HWND child : {g_hwndTargetLabel, g_hwndStatusLabel, g_hwndMatches, g_hwndPauseButton, g_hwndQuitButton})
        setFont(child);
}

LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_TIMER:
            if (wParam == kUiTimerId)
                updateUiFromSharedState();
            return 0;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case kIdPauseButton:
                case kIdTrayPause:
                    togglePause();
                    return 0;
                case kIdQuitButton:
                case kIdTrayQuit:
                    requestQuit(hwnd);
                    return 0;
                case kIdTrayShow:
                    showMainWindow(hwnd);
                    return 0;
            }
            return 0;

        case WM_APP_TRAYICON:
            switch (lParam) {
                case WM_LBUTTONUP:
                case WM_LBUTTONDBLCLK:
                    showMainWindow(hwnd);
                    break;
                case WM_RBUTTONUP:
                    showTrayContextMenu(hwnd);
                    break;
            }
            return 0;

        // Minimizing hides to the tray instead - the tray icon's own menu
        // (and this window's Quit button) is the only way to actually exit;
        // see the module comment / this file's plan for why.
        case WM_SYSCOMMAND:
            if ((wParam & 0xFFF0) == SC_MINIMIZE) {
                ShowWindow(hwnd, SW_HIDE);
                return 0;
            }
            return DefWindowProcA(hwnd, msg, wParam, lParam);

        // The window's own close button (X) - same reasoning as minimize
        // above: hide to tray, don't quit.
        case WM_CLOSE:
            ShowWindow(hwnd, SW_HIDE);
            return 0;

        // The worker thread's runCoordinator() call has returned - either
        // because the user confirmed Quit (g_quitting already true - see
        // requestQuit) or because it stopped on its own (e.g. registration
        // failed at startup), which is worth surfacing since there's no
        // console for the fprintf/onStatus text it would otherwise be
        // visible in.
        case WM_APP_COORDINATOR_STOPPED: {
            if (g_workerThread.joinable())
                g_workerThread.join();
            removeTrayIcon();
            if (!g_quitting) {
                std::string lastStatus;
                {
                    std::lock_guard<std::mutex> lock(g_status.mutex);
                    lastStatus = g_status.statusText;
                }
                MessageBoxA(hwnd, ("namebreak stopped unexpectedly: " + lastStatus).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            }
            DestroyWindow(hwnd);
            return 0;
        }

        case WM_DESTROY:
            KillTimer(hwnd, kUiTimerId);
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

} // namespace

int main(int argc, char* argv[]) {
    std::string configPath = kDefaultConfigPath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 >= argc) {
                MessageBoxA(nullptr, "--config requires a file path argument", kWindowTitle, MB_OK | MB_ICONERROR);
                return 1;
            }
            configPath = argv[++i];
        } else {
            MessageBoxA(nullptr, "Usage: namebreak-gui [--config <file>]", kWindowTitle, MB_OK | MB_ICONERROR);
            return 1;
        }
    }

    HINSTANCE hInstance = GetModuleHandleA(nullptr);

    INITCOMMONCONTROLSEX icex{};
    icex.dwSize = sizeof(icex);
    icex.dwICC = ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icex);

    CoordinatorArgs args;
    if (!prepareConfig(hInstance, configPath, args))
        return 1; // setup cancelled, or an unrecoverable config error already reported via MessageBox

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIconA(nullptr, IDI_APPLICATION);
    wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
    wc.lpszClassName = kWindowClassName;
    RegisterClassExA(&wc);

    g_hwndMain = CreateWindowExA(0, kWindowClassName, kWindowTitle, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                  CW_USEDEFAULT, CW_USEDEFAULT, kWindowWidth, kWindowHeight, nullptr, nullptr, hInstance, nullptr);
    if (!g_hwndMain) {
        MessageBoxA(nullptr, "Failed to create the main window", kWindowTitle, MB_OK | MB_ICONERROR);
        return 1;
    }
    createChildControls(g_hwndMain, hInstance);
    addTrayIcon(g_hwndMain);
    ShowWindow(g_hwndMain, SW_SHOW);
    UpdateWindow(g_hwndMain);
    SetTimer(g_hwndMain, kUiTimerId, kUiTimerIntervalMs, nullptr);

    g_workerThread = std::thread(workerThreadMain, args);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int) msg.wParam;
}
