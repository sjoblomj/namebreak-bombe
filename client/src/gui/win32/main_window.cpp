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
// Supports every one of the CLI's modes (coordinator/bounded/continuous/
// dictionary/meet-in-the-middle) - which one actually runs is read from
// config.conf's own `mode = ...` exactly like the CLI, with no --mode
// override here (there'd be nowhere to type it - see the setup dialog,
// which has a Coordinator tab, a Local Search tab covering bounded/
// continuous together, since they share the same [search] config keys and
// only differ in that one value, a Local Dictionary tab and a Meet in the
// Middle tab). The same dialog is the tray menu's Settings... (see
// openSettings), which is how any of it is changed later.
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

#include <algorithm>
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
#include "gui/win32/dpi.h"
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
constexpr int kIdGpuList = 105;
constexpr int kIdTrayShow = 201;
constexpr int kIdTrayPause = 202;
constexpr int kIdTrayQuit = 203;
constexpr int kIdTrayAbout = 204;
constexpr int kIdTrayFinish = 205;
constexpr int kIdTraySettings = 206;

constexpr const char* kFinishRangeText = "Finish current range, then pause";

constexpr size_t kMaxMatchLines = 100;
// The matches box's line for the candidate a dictionary search has got to -
// only ever its last line, each replacing the one before (see
// updateUiFromSharedState), so the box is never all of them.
constexpr const char* kSearchingPrefix = "Searching: ";

HWND g_hwndMain = nullptr;
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

// UI-thread-only: a search's controls - its target line, its status line,
// its progress bar and its search log - and what its log shows. The first
// is the window's own; a coordinator run searching on several GPUs gets one
// for each of them (see addGpuCells), each framed and titled with its GPU,
// laid out in a grid - or, with more than fit, one at a time, chosen in a
// list down the left (see layoutWindow).
struct SearchCell {
    HWND frame = nullptr; // the group box - several GPUs only
    HWND target = nullptr;
    HWND status = nullptr;
    HWND progress = nullptr;
    HWND logLabel = nullptr;
    HWND log = nullptr;
    // What the log shows. A matches file holds just its latest Hash-A match
    // (see MatchWriter, engine/match_writer.h), so each tick's is added here
    // whenever it changes, and only the last kMaxMatchLines are kept.
    // Started afresh for another matches file (another target), as the log
    // shows only the current one's.
    std::deque<std::string> recentMatches;
    std::string recentMatchesPath;
    // The line of its matches file last added to the log (not always the
    // log's last line, which may be a basename's or a search position's).
    std::string lastMatch;
};
std::vector<SearchCell> g_cells;
// With more GPUs than fit in a grid: the list of them, and the one whose
// cell is shown. Null otherwise.
HWND g_hwndGpuList = nullptr;
size_t g_selectedCell = 0;

// Layout, in 96-DPI pixels. A cell's contents are as wide as the window's
// were with one search; the log is as tall as fits, between the two.
constexpr int kMargin = 10;
constexpr int kContentWidth = 470;
constexpr int kCellLogHeight = 280; // one search, or the list's cell
constexpr int kGridLogHeight = 200;
constexpr int kMinGridLogHeight = 90;
// A grid of at most this many columns and rows - more GPUs than that are
// listed instead, as are as many as won't fit on the screen.
constexpr int kMaxGridColumns = 3;
constexpr int kMaxGridRows = 3;
constexpr int kGpuListWidth = 200;

// Set once in main() after prepareConfig() succeeds, never touched again -
// "coordinator", "bounded", "continuous" or "dictionary" (window title,
// Quit's confirmation wording).
std::string g_activeMode;
// Also set once in main(): what the tray menu's Settings... shows and
// writes (the file this run was started from - kDefaultConfigPath, or
// --config's), and the instance the form is created with.
std::string g_configPath;
HINSTANCE g_hInstance = nullptr;

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
    AppendMenuA(menu, MF_STRING, kIdTraySettings, "Settings...");
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

// The tray menu's Settings...: the setup form again, over the config this
// run started from (editSettings, app_config.cpp). The search carries on
// throughout, and goes on using the settings it was started with - the file
// is only read when namebreak starts, which is what the message box says.
// (The Hardware dialog's launch settings are the exception: a coordinator run
// uses them from its next range - see showHardwareDialog.)
// The main window is disabled while the form is up, so its Pause and Quit
// can't be pressed behind what looks like a modal window (the form runs a
// message loop of its own - see showSetupDialog).
void openSettings(HWND hwnd) {
    EnableWindow(hwnd, FALSE);
    const bool saved = editSettings(g_hInstance, g_configPath, hwnd, g_activeMode);
    // The window may be gone by now: a search that ended while the form was
    // up takes it with it (see WM_APP_WORKER_STOPPED).
    if (!IsWindow(hwnd))
        return;
    EnableWindow(hwnd, TRUE);
    SetForegroundWindow(hwnd);
    if (saved)
        MessageBoxA(hwnd,
                    ("Saved to " + g_configPath +
                     ".\r\n\r\nThis run keeps searching with the settings it started with; the new ones are used the next time "
                     "namebreak starts.")
                        .c_str(),
                    "namebreak settings", MB_OK | MB_ICONINFORMATION);
}

void requestQuit(HWND hwnd) {
    if (g_quitting)
        return; // already shutting down - ignore a repeat click
    const char* message = g_activeMode == "coordinator"
                               ? "Quit namebreak? This stops searching now; the coordinator is told how far the current range got, and hands out the rest again."
                           : g_activeMode == "dictionary" || g_activeMode == "meet-in-the-middle"
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
    for (const SearchCell& cell : g_cells)
        SetWindowTextA(cell.status, "Status: shutting down...");
    // Clearing pause too (rather than leaving runCoordinator's pause-wait
    // spin to notice quitRequested on its own, which it does - see
    // coordinator_runner.cpp) just keeps this state visually consistent
    // regardless of whether the user had paused first.
    g_pauseRequested.store(false, std::memory_order_relaxed);
    g_finishRangeThenPause.store(false, std::memory_order_relaxed);
    g_quitRequested.store(true, std::memory_order_relaxed);
}

// Moves and resizes a child of the main window, in 96-DPI pixels.
void place(HWND child, int x, int y, int width, int height) {
    SetWindowPos(child, nullptr, scaled(x), scaled(y), scaled(width), scaled(height), SWP_NOZORDER | SWP_NOACTIVATE);
}

// A cell's height, frame and all, with a log `logHeight` tall - and its
// width. Its contents are 105 tall above the log; a frame adds its caption
// above them and a margin round them.
int cellHeight(bool framed, int logHeight) {
    return 105 + logHeight + (framed ? 30 : 0);
}
int cellWidth(bool framed) {
    return kContentWidth + (framed ? 2 * kMargin : 0);
}

// A cell's controls, to be placed by layoutWindow. `framed` with several
// GPUs: a group box titled `name`, made first so it's under what it holds.
SearchCell makeCell(HWND hwnd, bool framed, const std::string& name) {
    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    auto child = [&](DWORD exStyle, const char* className, const char* text, DWORD style) {
        HWND made = CreateWindowExA(exStyle, className, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 0, 0, hwnd, nullptr, g_hInstance, nullptr);
        SendMessage(made, WM_SETFONT, (WPARAM) font, TRUE);
        return made;
    };
    SearchCell cell;
    if (framed)
        cell.frame = child(0, "BUTTON", name.c_str(), BS_GROUPBOX);
    cell.target = child(0, "STATIC", "Target: (none yet)", SS_ENDELLIPSIS);
    cell.status = child(0, "STATIC", "Status: starting...", SS_ENDELLIPSIS);
    cell.progress = child(0, PROGRESS_CLASSA, nullptr, PBS_MARQUEE);
    SendMessage(cell.progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    SendMessage(cell.progress, PBM_SETMARQUEE, TRUE, 100);
    // A dictionary search with an encryption key finds basenames that match
    // it too, which go in the same log, as does how far it has got - and so
    // does a meet-in-the-middle search's own progress.
    cell.logLabel = child(0, "STATIC",
                          g_activeMode == "dictionary" || g_activeMode == "coordinator" || g_activeMode == "meet-in-the-middle"
                              ? "Search log (most recent last):"
                              : "Hash A matches (most recent last):",
                          0);
    cell.log = child(WS_EX_CLIENTEDGE, "EDIT", "", WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL);
    return cell;
}

void placeCell(const SearchCell& cell, int x, int y, int logHeight) {
    if (cell.frame) {
        place(cell.frame, x, y, cellWidth(true), cellHeight(true, logHeight));
        x += kMargin;
        y += 20;
    }
    place(cell.target, x, y, kContentWidth, 20);
    place(cell.status, x, y + 25, kContentWidth, 20);
    place(cell.progress, x, y + 50, kContentWidth, 22);
    place(cell.logLabel, x, y + 85, kContentWidth, 18);
    place(cell.log, x, y + 105, kContentWidth, logHeight);
}

void showCell(const SearchCell& cell, bool show) {
    for (HWND child : {cell.frame, cell.target, cell.status, cell.progress, cell.logLabel, cell.log}) {
        if (child)
            ShowWindow(child, show ? SW_SHOW : SW_HIDE);
    }
}

// Lays the window out for the cells there are, and sizes it to fit: one
// search's cell on its own; several GPUs' in a grid, as square as it can be,
// their logs as tall as the screen has room for; or, with more than
// kMaxGridColumns x kMaxGridRows, or more than fit on the screen even with
// the shortest logs, a list of the GPUs down the left, with the chosen
// one's cell beside it. The buttons go under it all.
void layoutWindow(HWND hwnd) {
    const size_t count = g_cells.size();
    const int buttonsHeight = 30 + (g_hwndFinishCheckbox ? 8 + 20 : 0);
    RECT workArea;
    SystemParametersInfoA(SPI_GETWORKAREA, 0, &workArea, 0);
    const DWORD style = (DWORD) GetWindowLongPtr(hwnd, GWL_STYLE), exStyle = (DWORD) GetWindowLongPtr(hwnd, GWL_EXSTYLE);
    // The window's size for a client area of `width` x `height`.
    auto windowSize = [&](int width, int height) {
        RECT r = {0, 0, scaled(width), scaled(height)};
        AdjustWindowRectEx(&r, style, FALSE, exStyle);
        return SIZE{r.right - r.left, r.bottom - r.top};
    };
    auto fits = [&](int width, int height) {
        const SIZE size = windowSize(width, height);
        return size.cx <= workArea.right - workArea.left && size.cy <= workArea.bottom - workArea.top;
    };
    // The tallest log, from `tallest` down to the shortest, that a body of
    // `width` x `bodyHeight(log)` fits with, or 0 for none.
    auto tallestLog = [&](int tallest, int width, auto bodyHeight) {
        for (int log = tallest; log >= kMinGridLogHeight; log -= 10) {
            if (fits(width, bodyHeight(log) + kMargin + buttonsHeight + kMargin))
                return log;
        }
        return 0;
    };

    int width, bodyBottom;
    bool listed = false;
    if (count == 1) {
        placeCell(g_cells[0], kMargin, kMargin, kCellLogHeight);
        width = kMargin + cellWidth(false) + kMargin;
        bodyBottom = kMargin + cellHeight(false, kCellLogHeight);
    } else {
        size_t columns = 1;
        while (columns * columns < count)
            ++columns;
        const size_t rows = (count + columns - 1) / columns;
        const int pitch = cellWidth(true) + kMargin;
        width = kMargin + (int) columns * pitch;
        const int log = columns <= kMaxGridColumns && rows <= kMaxGridRows
                            ? tallestLog(kGridLogHeight, width, [&](int log) { return kMargin + (int) rows * (cellHeight(true, log) + kMargin) - kMargin; })
                            : 0;
        if (log > 0) {
            for (size_t i = 0; i < count; ++i)
                placeCell(g_cells[i], kMargin + (int) (i % columns) * pitch, kMargin + (int) (i / columns) * (cellHeight(true, log) + kMargin), log);
            bodyBottom = kMargin + (int) rows * (cellHeight(true, log) + kMargin) - kMargin;
        } else {
            listed = true;
            width = kMargin + kGpuListWidth + kMargin + cellWidth(true) + kMargin;
            const int listLog = std::max(kMinGridLogHeight, tallestLog(kCellLogHeight, width, [&](int log) { return kMargin + cellHeight(true, log); }));
            bodyBottom = kMargin + cellHeight(true, listLog);
            if (!g_hwndGpuList) {
                g_hwndGpuList = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", nullptr, WS_CHILD | WS_VISIBLE | WS_VSCROLL | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT,
                                                0, 0, 0, 0, hwnd, (HMENU) (INT_PTR) kIdGpuList, g_hInstance, nullptr);
                SendMessage(g_hwndGpuList, WM_SETFONT, (WPARAM) GetStockObject(DEFAULT_GUI_FONT), TRUE);
            }
            SendMessage(g_hwndGpuList, LB_RESETCONTENT, 0, 0);
            for (const SearchCell& cell : g_cells) {
                char name[256];
                GetWindowTextA(cell.frame, name, sizeof name);
                SendMessage(g_hwndGpuList, LB_ADDSTRING, 0, (LPARAM) name);
            }
            SendMessage(g_hwndGpuList, LB_SETCURSEL, g_selectedCell, 0);
            place(g_hwndGpuList, kMargin, kMargin, kGpuListWidth, bodyBottom - kMargin);
            for (const SearchCell& cell : g_cells)
                placeCell(cell, kMargin + kGpuListWidth + kMargin, kMargin, listLog);
        }
    }
    for (size_t i = 0; i < count; ++i)
        showCell(g_cells[i], !listed || i == g_selectedCell);

    const int buttonsTop = bodyBottom + kMargin;
    place(g_hwndPauseButton, kMargin, buttonsTop, 150, 30);
    place(g_hwndAboutButton, kMargin + 160, buttonsTop, 150, 30);
    place(g_hwndQuitButton, kMargin + 320, buttonsTop, 150, 30);
    if (g_hwndFinishCheckbox)
        place(g_hwndFinishCheckbox, kMargin, buttonsTop + 38, kContentWidth, 20);

    // Grown or shrunk where it is - moved only as far as keeps it on the
    // screen.
    const SIZE size = windowSize(width, buttonsTop + buttonsHeight + kMargin);
    RECT window;
    GetWindowRect(hwnd, &window);
    const int x = std::max(workArea.left, std::min(window.left, workArea.right - size.cx));
    const int y = std::max(workArea.top, std::min(window.top, workArea.bottom - size.cy));
    SetWindowPos(hwnd, nullptr, x, y, size.cx, size.cy, SWP_NOZORDER | SWP_NOACTIVATE);
}

// The list's choice: that GPU's cell, in place of the one shown.
void selectCell(size_t index) {
    if (index >= g_cells.size() || index == g_selectedCell)
        return;
    showCell(g_cells[g_selectedCell], false);
    g_selectedCell = index;
    showCell(g_cells[index], true);
}

// A cell for each of `gpus` - a coordinator run's on several GPUs (see
// CoordinatorCallbacks::onWorkers), each framed and titled with its GPU: the
// window's own one search's cell is made again as the first GPU's, framed,
// and the window laid out afresh for them all.
void addGpuCells(HWND hwnd, const std::vector<std::string>& gpus) {
    if (!g_cells.empty() && !g_cells[0].frame) {
        for (HWND child : {g_cells[0].target, g_cells[0].status, g_cells[0].progress, g_cells[0].logLabel, g_cells[0].log})
            DestroyWindow(child);
        g_cells.clear();
    }
    while (g_cells.size() < gpus.size())
        g_cells.push_back(makeCell(hwnd, true, gpus[g_cells.size()]));
    layoutWindow(hwnd);
}

// Shows `search` in `bar`: how far it has got, if that's known (see
// below), full once it's done, or the marquee while nothing says.
// `latestMatch` is its matches file's last line.
void updateProgressBar(HWND bar, const SearchStatus& search, const std::string& latestMatch, bool searchEnded) {
    // Determinate (a real fraction, from matchProgressFraction below) only
    // for bounded-shaped work, which has a fixed finish line to measure
    // against - a coordinator range, or local bounded mode (see
    // onRangeClaimed/runLocalSearch, which leave lowerBound empty exactly
    // when there isn't one: continuous mode, or nothing running yet) - and
    // for a dictionary search or range, which says how far it has got itself
    // (progressFraction).
    bool determinate = search.hasActiveRange && (search.progressFraction >= 0 || !search.lowerBound.empty());
    if (determinate) {
        LONG_PTR style = GetWindowLongPtr(bar, GWL_STYLE);
        if (style & PBS_MARQUEE) {
            SendMessage(bar, PBM_SETMARQUEE, FALSE, 0);
            SetWindowLongPtr(bar, GWL_STYLE, style & ~PBS_MARQUEE);
        }
        // 0% until the first Hash-A match of this range/search arrives -
        // there's nothing else to measure progress from (see
        // CoordinatorCallbacks' doc comment on why no other counter
        // exists), and pausing simply stops new matches from arriving, so
        // this naturally freezes/resumes with no extra bookkeeping.
        double fraction = 0.0;
        if (search.progressFraction >= 0) {
            fraction = search.progressFraction;
        } else if (!latestMatch.empty()) {
            double f = matchProgressFraction(latestMatch, search.prefix, search.suffix, search.insertFromStart, search.insertFromEnd,
                                             search.alphabet, search.lowerBound, search.upperBound);
            if (f >= 0.0)
                fraction = f;
        }
        SendMessage(bar, PBM_SETPOS, (WPARAM) (fraction * 100.0), 0);
    } else if (search.rangeJustFinished || searchEnded) {
        // Continuous-mode runs (see runLocalSearch) never set a determinate
        // range, so the bar sits in marquee style for their entire run -
        // marquee has to be turned off explicitly here too, not just above.
        LONG_PTR style = GetWindowLongPtr(bar, GWL_STYLE);
        if (style & PBS_MARQUEE) {
            SendMessage(bar, PBM_SETMARQUEE, FALSE, 0);
            SetWindowLongPtr(bar, GWL_STYLE, style & ~PBS_MARQUEE);
        }
        SendMessage(bar, PBM_SETPOS, 100, 0);
    } else {
        LONG_PTR style = GetWindowLongPtr(bar, GWL_STYLE);
        if (!(style & PBS_MARQUEE)) {
            SetWindowLongPtr(bar, GWL_STYLE, style | PBS_MARQUEE);
            SendMessage(bar, PBM_SETMARQUEE, TRUE, 100);
        }
    }
}

// Re-reads the shared coordinator-status struct and the current matches
// files, and refreshes every control accordingly - the only place any of
// this GUI's controls are actually touched, always from the UI thread's own
// WM_TIMER tick (see SharedStatus's own comment on why).
void updateUiFromSharedState() {
    std::vector<SearchStatus> searches;
    std::vector<std::string> gpus;
    bool searchEnded;
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        searches = g_status.searches;
        gpus = g_status.gpus;
        searchEnded = g_status.searchEnded;
        // What only this tick shows - consumed here.
        for (SearchStatus& search : g_status.searches) {
            search.rangeJustFinished = false;
            search.newBasenames.clear();
            search.newMeetingPointMatches.clear();
            search.newSearchPosition.clear();
        }
    }
    if (gpus.size() > g_cells.size())
        addGpuCells(g_hwndMain, gpus);

    bool paused = g_pauseRequested.load(std::memory_order_relaxed);
    if (!g_quitting)
        syncPauseControls();

    // Coordinator mode may search dictionary ranges, whose basenames and
    // search positions go in the log too, as a meet-in-the-middle search's
    // own lines do - so its Hash A matches are said to be.
    const bool dictionaryLines =
        g_activeMode == "coordinator" || g_activeMode == "dictionary" || g_activeMode == "meet-in-the-middle";

    for (size_t i = 0; i < searches.size() && i < g_cells.size(); ++i) {
        const SearchStatus& search = searches[i];
        SearchCell& cell = g_cells[i];
        const std::string target = search.targetName.empty() ? "(none yet)" : search.targetName;
        SetWindowTextA(cell.target, ("Target: " + target).c_str());
        SetWindowTextA(cell.status, ("Status: " + search.statusText + (paused ? " (paused)" : "")).c_str());

        bool logChanged = false;
        auto addLogLine = [&](const std::string& line) {
            cell.recentMatches.push_back(line);
            if (cell.recentMatches.size() > kMaxMatchLines)
                cell.recentMatches.pop_front();
            logChanged = true;
        };
        if (search.outputFilePath != cell.recentMatchesPath) {
            cell.recentMatches.clear();
            cell.recentMatchesPath = search.outputFilePath;
            cell.lastMatch.clear();
            logChanged = true;
        }

        // Read once, shared below by both the progress calculation and the
        // log: the most recent Hash-A match (the file's only line - or its
        // last, in one an older version appended every match to).
        std::string latestMatch;
        if (!search.outputFilePath.empty()) {
            std::vector<std::string> lines = readLastLines(search.outputFilePath, 1);
            if (!lines.empty())
                latestMatch = lines.back();
        }
        updateProgressBar(cell.progress, search, latestMatch, searchEnded);

        if (!latestMatch.empty() && latestMatch != cell.lastMatch) {
            cell.lastMatch = latestMatch;
            addLogLine(dictionaryLines ? "Hash A match: " + latestMatch : latestMatch);
        }
        for (const std::string& basename : search.newBasenames)
            addLogLine("Basename: " + basename);
        // A directory whose state matched the meeting point - which is what a
        // meet-in-the-middle search looks for; the hashes then decide.
        for (const std::string& filename : search.newMeetingPointMatches)
            addLogLine("Matched the meeting point: " + filename);
        // Only ever the log's last line, each replacing the one before.
        if (!search.newSearchPosition.empty()) {
            if (!cell.recentMatches.empty() && cell.recentMatches.back().rfind(kSearchingPrefix, 0) == 0)
                cell.recentMatches.pop_back();
            addLogLine(kSearchingPrefix + search.newSearchPosition);
        }
        if (logChanged) {
            std::string joined;
            for (const std::string& match : cell.recentMatches) {
                if (!joined.empty())
                    joined += "\r\n";
                joined += match;
            }
            SetWindowTextA(cell.log, joined.c_str());
            SendMessage(cell.log, EM_SETSEL, (WPARAM) -1, (LPARAM) -1);
            SendMessage(cell.log, EM_SCROLLCARET, 0, 0);
        }
    }
}

void createChildControls(HWND hwnd, HINSTANCE hInstance) {
    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    auto setFont = [&](HWND child) { SendMessage(child, WM_SETFONT, (WPARAM) font, TRUE); };

    g_cells.clear();
    g_cells.push_back(makeCell(hwnd, false, ""));

    g_hwndPauseButton = CreateWindowExA(0, "BUTTON", "Pause", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd,
                                         (HMENU) (INT_PTR) kIdPauseButton, hInstance, nullptr);
    g_hwndAboutButton = CreateWindowExA(0, "BUTTON", "About...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd,
                                         (HMENU) (INT_PTR) kIdAboutButton, hInstance, nullptr);
    g_hwndQuitButton = CreateWindowExA(0, "BUTTON", "Quit", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd,
                                        (HMENU) (INT_PTR) kIdQuitButton, hInstance, nullptr);
    for (HWND child : {g_hwndPauseButton, g_hwndAboutButton, g_hwndQuitButton})
        setFont(child);

    // Local searches have no ranges to finish.
    if (g_activeMode == "coordinator") {
        g_hwndFinishCheckbox = CreateWindowExA(0, "BUTTON", kFinishRangeText, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 0, 0, 0, 0, hwnd,
                                                (HMENU) (INT_PTR) kIdFinishCheckbox, hInstance, nullptr);
        setFont(g_hwndFinishCheckbox);
    }
    // At the system DPI, as layoutWindow places everything.
    layoutWindow(hwnd);
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
                case kIdTraySettings:
                    openSettings(hwnd);
                    return 0;
                case kIdTrayShow:
                    showMainWindow(hwnd);
                    return 0;
                case kIdGpuList:
                    if (HIWORD(wParam) == LBN_SELCHANGE)
                        selectCell((size_t) SendMessage(g_hwndGpuList, LB_GETCURSEL, 0, 0));
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
                    lastStatus = g_status.lastStatus;
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

    enableDpiAwareness();
    HINSTANCE hInstance = GetModuleHandleA(nullptr);
    g_hInstance = hInstance;
    g_configPath = configPath;

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
                                  CW_USEDEFAULT, CW_USEDEFAULT, scaled(kWindowWidth), scaled(kWindowHeight), nullptr, nullptr, hInstance, nullptr);
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
