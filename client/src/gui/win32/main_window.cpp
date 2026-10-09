// A little native Win32 GUI for namebreak - built only on Windows (see the
// namebreak-gui target in CMakeLists.txt), linking the same search/
// coordinator/config/platform sources the console `namebreak` binary does -
// everything but its src/cli/main.cpp, so this file's own main() is the
// process entry point.
// Built with /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup (see CMakeLists.txt) so
// the process has no console, but the CRT still calls this ordinary main()
// rather than requiring a WinMain() - which also means this GUI can honor the
// same --config <file> the CLI does (kDefaultConfigPath, config.h).
//
// Supports all four of the CLI's modes (coordinator/bounded/continuous/
// dictionary) - which one actually runs is read from config.conf's own
// `mode = ...` exactly like the CLI, with no --mode override here (there'd
// be nowhere to type it - see the first-run setup dialog, which has a
// Coordinator tab, a Local Search tab covering bounded/continuous together,
// since they share the same [search] config keys and only differ in that
// one value, and a Local Dictionary tab).
//
// This file is the main window and tray icon; the rest of the GUI is:
//   app_config    loading the config, running the setup dialog if needed
//   setup_dialog  the first-run setup dialog
//   worker        the background thread running the search/coordinator loop
//   about_dialog, app_icons

#include "gui/win32/win32.h"

// PBS_MARQUEE (the "no work claimed yet" progress-bar state below) needs
// comctl32 v6, which Windows only loads side-by-side instead of the old
// system v5 copy when the process declares that dependency - normally done
// via a separate .manifest/.rc file. Embedding it as a linker directive
// instead means no resource compiler step.
#pragma comment(linker, \
    "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")

#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/config.h"
#include "common/matches_file.h"
#include "gui/win32/about_dialog.h"
#include "gui/win32/app_config.h"
#include "gui/win32/app_icons.h"
#include "gui/win32/worker.h"

namespace gui {
namespace {

constexpr const char* kWindowClassName = "NamebreakGuiWindow";
constexpr const char* kWindowTitle = "namebreak";
constexpr int kWindowWidth = 500;
constexpr int kWindowHeight = 520;

constexpr UINT_PTR kUiTimerId = 1;
constexpr UINT kUiTimerIntervalMs = 1000;
constexpr UINT WM_APP_TRAYICON = WM_APP + 1;

constexpr int kIdPauseButton = 101;
constexpr int kIdQuitButton = 102;
constexpr int kIdAboutButton = 103;
constexpr int kIdFinishCheckbox = 104;
constexpr int kIdTrayShow = 201;
constexpr int kIdTrayPause = 202;
constexpr int kIdTrayQuit = 203;
constexpr int kIdTrayAbout = 204;
constexpr int kIdTrayFinish = 205;

constexpr const char* kFinishRangeText = "Finish current range, then pause";

constexpr size_t kMaxMatchLines = 100;

HWND g_hwndMain = nullptr;
HWND g_hwndTargetLabel = nullptr;
HWND g_hwndStatusLabel = nullptr;
HWND g_hwndProgress = nullptr;
HWND g_hwndMatches = nullptr;
HWND g_hwndPauseButton = nullptr;
HWND g_hwndQuitButton = nullptr;
HWND g_hwndAboutButton = nullptr;
// Coordinator mode only (null otherwise) - see g_finishRangeThenPause.
HWND g_hwndFinishCheckbox = nullptr;

NOTIFYICONDATAA g_trayIcon{};
bool g_trayIconAdded = false;

std::thread g_workerThread;
// UI-thread-only (never touched from the worker thread): true once the user
// has confirmed Quit, so WM_APP_WORKER_STOPPED can tell "the loop stopped
// because we asked it to" apart from "the loop stopped on its own" (e.g.
// registration failed at startup) and only alarm the user for the latter.
bool g_quitting = false;

// UI-thread-only: what the matches box shows. A matches file holds just its
// latest Hash-A match (see MatchWriter, engine/match_writer.h), so each
// tick's is added here whenever it changes, and only the last
// kMaxMatchLines are kept. Started afresh for another matches file (another
// target), as the box shows only the current one's.
std::deque<std::string> g_recentMatches;
std::string g_recentMatchesPath;

// Set once in main() after prepareConfig() succeeds, never touched again -
// "coordinator", "bounded", "continuous" or "dictionary" (window title,
// Quit's confirmation wording).
std::string g_activeMode;

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
    g_trayIcon.hIcon = appIconSmall();
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
    if (g_activeMode == "coordinator")
        AppendMenuA(menu, MF_STRING | (g_finishRangeThenPause.load(std::memory_order_relaxed) ? MF_CHECKED : MF_UNCHECKED), kIdTrayFinish,
                    kFinishRangeText);
    AppendMenuA(menu, MF_STRING, kIdTrayAbout, "About...");
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

// Brings the Pause button's label and the finish-range checkbox in line with
// g_pauseRequested/g_finishRangeThenPause - which the worker thread changes
// too (runCoordinator pauses, and clears the finish flag, once a range it
// was told to finish is done), so this also runs on every UI timer tick.
void syncPauseControls() {
    bool paused = g_pauseRequested.load(std::memory_order_relaxed);
    char current[16];
    GetWindowTextA(g_hwndPauseButton, current, sizeof(current));
    const char* wanted = paused ? "Resume" : "Pause";
    if (strcmp(current, wanted) != 0)
        SetWindowTextA(g_hwndPauseButton, wanted);
    if (g_hwndFinishCheckbox) {
        bool finishing = g_finishRangeThenPause.load(std::memory_order_relaxed);
        if ((SendMessage(g_hwndFinishCheckbox, BM_GETCHECK, 0, 0) == BST_CHECKED) != finishing)
            SendMessage(g_hwndFinishCheckbox, BM_SETCHECK, finishing ? BST_CHECKED : BST_UNCHECKED, 0);
    }
}

// Turning it on while paused resumes too - that's the point of it: carry
// on, but only to the end of the range in hand.
void setFinishRangeThenPause(bool on) {
    g_finishRangeThenPause.store(on, std::memory_order_relaxed);
    if (on)
        g_pauseRequested.store(false, std::memory_order_relaxed);
    syncPauseControls();
}

void togglePause() {
    bool newState = !g_pauseRequested.load(std::memory_order_relaxed);
    g_pauseRequested.store(newState, std::memory_order_relaxed);
    syncPauseControls();
    // No pause-duration bookkeeping needed here (unlike an earlier,
    // time-based version of this progress bar): the matches file only ever
    // changes while actually searching, so the match-based progress
    // fraction (see matchProgressFraction/updateUiFromSharedState) simply
    // stops advancing on its own while paused, with nothing to unwind on
    // resume.
}

void requestQuit(HWND hwnd) {
    if (g_quitting)
        return; // already shutting down - ignore a repeat click
    const char* message = g_activeMode == "coordinator"
                               ? "Quit namebreak? This stops searching now; the coordinator is told how far the current range got, and hands out the rest again."
                           : g_activeMode == "dictionary"
                               ? "Quit namebreak? This stops the search now. How far it got is saved, for a search with Resume from last candidate to carry on from."
                               : "Quit namebreak? This stops the search now.";
    int result = MessageBoxA(hwnd, message, "Confirm Quit", MB_YESNO | MB_ICONQUESTION);
    if (result != IDYES)
        return;
    g_quitting = true;
    EnableWindow(g_hwndPauseButton, FALSE);
    EnableWindow(g_hwndQuitButton, FALSE);
    if (g_hwndFinishCheckbox)
        EnableWindow(g_hwndFinishCheckbox, FALSE);
    SetWindowTextA(g_hwndStatusLabel, "Status: shutting down...");
    // Clearing pause too (rather than leaving runCoordinator's pause-wait
    // spin to notice quitRequested on its own, which it does - see
    // coordinator_runner.cpp) just keeps this state visually consistent
    // regardless of whether the user had paused first.
    g_pauseRequested.store(false, std::memory_order_relaxed);
    g_finishRangeThenPause.store(false, std::memory_order_relaxed);
    g_quitRequested.store(true, std::memory_order_relaxed);
}

// Re-reads the shared coordinator-status struct and the current matches
// file, and refreshes every control accordingly - the only place any of
// this GUI's controls are actually touched, always from the UI thread's own
// WM_TIMER tick (see SharedStatus's own comment on why).
void updateUiFromSharedState() {
    std::string targetName, outputFilePath, statusText;
    std::string alphabet, prefix, suffix, lowerBound, upperBound;
    Insertion insertFromStart, insertFromEnd;
    double progressFraction;
    bool hasActiveRange, rangeJustFinished, searchEnded;
    std::vector<std::string> newBasenames;
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        targetName = g_status.targetName;
        outputFilePath = g_status.outputFilePath;
        statusText = g_status.statusText;
        alphabet = g_status.alphabet;
        prefix = g_status.prefix;
        suffix = g_status.suffix;
        insertFromStart = g_status.insertFromStart;
        insertFromEnd = g_status.insertFromEnd;
        lowerBound = g_status.lowerBound;
        upperBound = g_status.upperBound;
        progressFraction = g_status.progressFraction;
        hasActiveRange = g_status.hasActiveRange;
        searchEnded = g_status.searchEnded;
        rangeJustFinished = g_status.rangeJustFinished;
        g_status.rangeJustFinished = false; // one-shot edge, consumed here
        newBasenames.swap(g_status.newBasenames);
    }

    bool paused = g_pauseRequested.load(std::memory_order_relaxed);
    if (!g_quitting)
        syncPauseControls();

    SetWindowTextA(g_hwndTargetLabel, (targetName.empty() ? "Target: (none yet)" : "Target: " + targetName).c_str());
    SetWindowTextA(g_hwndStatusLabel, ("Status: " + statusText + (paused ? " (paused)" : "")).c_str());

    // Read once, shared below by both the progress calculation and the
    // matches box: the most recent Hash-A match (the file's only line - or
    // its last, in one an older version appended every match to).
    std::string latestMatch;
    if (!outputFilePath.empty()) {
        std::vector<std::string> lines = readLastLines(outputFilePath, 1);
        if (!lines.empty())
            latestMatch = lines.back();
    }

    // Determinate (a real fraction, from matchProgressFraction below) only
    // for bounded-shaped work, which has a fixed finish line to measure
    // against - a coordinator range, or local bounded mode (see
    // onRangeClaimed/runLocalSearch, which leave lowerBound empty exactly
    // when there isn't one: continuous mode, or nothing running yet) - and
    // for a local dictionary search, which says how far it has got itself
    // (progressFraction).
    bool determinate = hasActiveRange && (progressFraction >= 0 || !lowerBound.empty());
    if (determinate) {
        LONG_PTR style = GetWindowLongPtr(g_hwndProgress, GWL_STYLE);
        if (style & PBS_MARQUEE) {
            SendMessage(g_hwndProgress, PBM_SETMARQUEE, FALSE, 0);
            SetWindowLongPtr(g_hwndProgress, GWL_STYLE, style & ~PBS_MARQUEE);
        }
        // 0% until the first Hash-A match of this range/search arrives -
        // there's nothing else to measure progress from (see
        // CoordinatorCallbacks' doc comment on why no other counter
        // exists), and pausing simply stops new matches from arriving, so
        // this naturally freezes/resumes with no extra bookkeeping.
        double fraction = 0.0;
        if (progressFraction >= 0) {
            fraction = progressFraction;
        } else if (!latestMatch.empty()) {
            double f = matchProgressFraction(latestMatch, prefix, suffix, insertFromStart, insertFromEnd, alphabet, lowerBound, upperBound);
            if (f >= 0.0)
                fraction = f;
        }
        SendMessage(g_hwndProgress, PBM_SETPOS, (WPARAM) (fraction * 100.0), 0);
    } else if (rangeJustFinished || searchEnded) {
        // Continuous-mode runs (see runLocalSearch) never set a determinate
        // range, so the bar sits in marquee style for their entire run -
        // marquee has to be turned off explicitly here too, not just above.
        LONG_PTR style = GetWindowLongPtr(g_hwndProgress, GWL_STYLE);
        if (style & PBS_MARQUEE) {
            SendMessage(g_hwndProgress, PBM_SETMARQUEE, FALSE, 0);
            SetWindowLongPtr(g_hwndProgress, GWL_STYLE, style & ~PBS_MARQUEE);
        }
        SendMessage(g_hwndProgress, PBM_SETPOS, 100, 0);
    } else {
        LONG_PTR style = GetWindowLongPtr(g_hwndProgress, GWL_STYLE);
        if (!(style & PBS_MARQUEE)) {
            SetWindowLongPtr(g_hwndProgress, GWL_STYLE, style | PBS_MARQUEE);
            SendMessage(g_hwndProgress, PBM_SETMARQUEE, TRUE, 100);
        }
    }

    bool matchesChanged = false;
    if (outputFilePath != g_recentMatchesPath) {
        g_recentMatches.clear();
        g_recentMatchesPath = outputFilePath;
        matchesChanged = true;
    }
    auto addMatchLine = [&](const std::string& line) {
        g_recentMatches.push_back(line);
        if (g_recentMatches.size() > kMaxMatchLines)
            g_recentMatches.pop_front();
        matchesChanged = true;
    };
    if (!latestMatch.empty() && (g_recentMatches.empty() || g_recentMatches.back() != latestMatch))
        addMatchLine(latestMatch);
    for (const std::string& basename : newBasenames)
        addMatchLine("Basename: " + basename);
    if (matchesChanged) {
        std::string joined;
        for (const std::string& match : g_recentMatches) {
            if (!joined.empty())
                joined += "\r\n";
            joined += match;
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

    // A dictionary search with an encryption key finds basenames that match
    // it too, which go in the same box.
    CreateWindowExA(0, "STATIC",
                     g_activeMode == "dictionary" ? "Hash A matches and basenames (most recent last):" : "Hash A matches (most recent last):",
                     WS_CHILD | WS_VISIBLE, 10, 95, 470, 18, hwnd, nullptr, hInstance, nullptr);
    g_hwndMatches = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
                                     WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL, 10, 115, 470, 280,
                                     hwnd, nullptr, hInstance, nullptr);

    g_hwndPauseButton = CreateWindowExA(0, "BUTTON", "Pause", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 10, 405, 150, 30, hwnd,
                                         (HMENU) (INT_PTR) kIdPauseButton, hInstance, nullptr);
    g_hwndAboutButton = CreateWindowExA(0, "BUTTON", "About...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 170, 405, 150, 30, hwnd,
                                         (HMENU) (INT_PTR) kIdAboutButton, hInstance, nullptr);
    g_hwndQuitButton = CreateWindowExA(0, "BUTTON", "Quit", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 330, 405, 150, 30, hwnd,
                                        (HMENU) (INT_PTR) kIdQuitButton, hInstance, nullptr);

    for (HWND child : {g_hwndTargetLabel, g_hwndStatusLabel, g_hwndMatches, g_hwndPauseButton, g_hwndAboutButton, g_hwndQuitButton})
        setFont(child);

    // Local searches have no ranges to finish.
    if (g_activeMode == "coordinator") {
        g_hwndFinishCheckbox = CreateWindowExA(0, "BUTTON", kFinishRangeText, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 10, 443, 470, 20,
                                                hwnd, (HMENU) (INT_PTR) kIdFinishCheckbox, hInstance, nullptr);
        setFont(g_hwndFinishCheckbox);
    }
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
                case kIdFinishCheckbox:
                    setFinishRangeThenPause(SendMessage(g_hwndFinishCheckbox, BM_GETCHECK, 0, 0) == BST_CHECKED);
                    return 0;
                case kIdTrayFinish:
                    setFinishRangeThenPause(!g_finishRangeThenPause.load(std::memory_order_relaxed));
                    return 0;
                case kIdQuitButton:
                case kIdTrayQuit:
                    requestQuit(hwnd);
                    return 0;
                case kIdAboutButton:
                case kIdTrayAbout:
                    showAboutDialog(hwnd);
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

        // The worker thread's runCoordinator()/runSearch() call has
        // returned - either because the user confirmed Quit (g_quitting
        // already true - see requestQuit) or because it stopped on its own
        // (e.g. coordinator registration failed at startup), which is worth
        // surfacing since there's no console for the fprintf/onStatus text
        // it would otherwise be visible in.
        case WM_APP_WORKER_STOPPED: {
            if (g_workerThread.joinable())
                g_workerThread.join();
            removeTrayIcon();
            if (!g_quitting) {
                std::string lastStatus;
                bool ended;
                {
                    std::lock_guard<std::mutex> lock(g_status.mutex);
                    lastStatus = g_status.statusText;
                    ended = g_status.searchEnded;
                }
                if (ended)
                    MessageBoxA(hwnd, ("namebreak finished: " + lastStatus).c_str(), "namebreak", MB_OK | MB_ICONINFORMATION);
                else
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
} // namespace gui

int main(int argc, char* argv[]) {
    using namespace gui;

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
    icex.dwICC = ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES | ICC_TAB_CLASSES | ICC_WIN95_CLASSES;
    InitCommonControlsEx(&icex);

    AppConfig config;
    if (!prepareConfig(hInstance, configPath, config))
        return 1; // setup cancelled, or an unrecoverable config error already reported via MessageBox
    g_activeMode = config.mode;

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInstance;
    wc.hIcon = appIconLarge();
    wc.hIconSm = appIconSmall();
    wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
    wc.lpszClassName = kWindowClassName;
    RegisterClassExA(&wc);

    std::string windowTitle = std::string(kWindowTitle) + " (" + g_activeMode + ")";
    g_hwndMain = CreateWindowExA(0, kWindowClassName, windowTitle.c_str(), WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
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

    g_workerThread = std::thread(workerThreadMain, config, g_hwndMain);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int) msg.wParam;
}
