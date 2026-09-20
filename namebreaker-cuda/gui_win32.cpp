// A little native Win32 GUI for namebreak - built only on Windows (see the
// Makefile's `namebreak-gui` target), linking the same search/coordinator/
// config/platform sources the console `namebreak` binary does - everything
// but its main.cpp, so this file's own main() is the process entry point.
// Built with /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup (see the
// Makefile) so the process has no console, but the CRT still calls this
// ordinary main() rather than requiring a WinMain() - which also means this
// GUI can honor the same --config <file> the CLI does (kDefaultConfigPath,
// config.h).
//
// Supports all three of the CLI's modes (coordinator/bounded/continuous) -
// which one actually runs is read from config.conf's own `mode = ...`
// exactly like the CLI, with no --mode override here (there'd be nowhere to
// type it - see the first-run setup dialog, which has a Coordinator tab and
// a Local Search tab covering bounded/continuous together, since they share
// the same [search] config keys and only differ in that one value).

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
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "coordinator_runner.h"
#include "cpu-utils.h"
#include "icon_bmp.h"
#include "logo_bmp.h"
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
// Fired once the worker thread's runCoordinator()/runSearch() call returns,
// whichever mode this run is in - see workerThreadMain.
constexpr UINT WM_APP_WORKER_STOPPED = WM_APP + 2;

constexpr int kIdPauseButton = 101;
constexpr int kIdQuitButton = 102;
constexpr int kIdAboutButton = 103;
constexpr int kIdTrayShow = 201;
constexpr int kIdTrayPause = 202;
constexpr int kIdTrayQuit = 203;
constexpr int kIdTrayAbout = 204;
constexpr int kIdSetupOk = 301;
constexpr int kIdSetupCancel = 302;
constexpr int kIdSetupBoundedRadio = 303;
constexpr int kIdSetupContinuousRadio = 304;
constexpr int kIdSetupPruneCheckbox = 305;
constexpr int kIdSetupAbout = 306;

constexpr size_t kMaxMatchLines = 100;
// Prefilled into the setup dialog's Server URL field when the config doesn't
// already have one.
constexpr const char* kDefaultServerUrl = "https://namebreak-coordinator.fly.dev";
// Tab indices in the setup dialog's SysTabControl32 - also doubles as which
// group of controls (coordinator vs. search) is shown/enabled at a time.
constexpr int kTabCoordinator = 0;
constexpr int kTabLocalSearch = 1;

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
    // Populated only for "bounded-shaped" work - a coordinator range, or
    // local bounded mode - which has a real finish line (a fixed, single
    // candidate length between lowerBound/upperBound); left empty for
    // continuous mode, which has none by design (see runLocalSearch), and
    // before any range/search has started. When populated, the UI timer
    // derives a real progress fraction from these plus the matches file's
    // own last line (see matchProgressFraction/updateUiFromSharedState)
    // instead of guessing from elapsed time: candidates are enumerated in a
    // fixed order (cpu-utils.h's stringToIndex), so the most recent Hash-A-
    // only match's position in that order is a true measure of how far a
    // search has gotten - the same signal the coordinator server itself
    // already uses for stall detection (see coordinator/README.md). When
    // empty, the UI shows the indeterminate marquee style instead.
    std::string alphabet;
    std::string prefix;
    std::string suffix;
    std::string lowerBound; // candidate-only (no prefix/suffix)
    std::string upperBound; // candidate-only
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
HWND g_hwndAboutButton = nullptr;

NOTIFYICONDATAA g_trayIcon{};
bool g_trayIconAdded = false;

std::atomic<bool> g_pauseRequested{false};
std::atomic<bool> g_quitRequested{false};
std::thread g_workerThread;
// UI-thread-only (never touched from the worker thread): true once the user
// has confirmed Quit, so WM_APP_WORKER_STOPPED can tell "the loop stopped
// because we asked it to" apart from "the loop stopped on its own" (e.g.
// registration failed at startup) and only alarm the user for the latter.
bool g_quitting = false;

// Set once in main() after prepareConfig() succeeds, never touched again -
// "coordinator", "bounded", or "continuous". Read by both the worker thread
// (to decide which of runCoordinator()/runSearch() to call) and the UI
// thread (window title, Quit's confirmation wording).
std::string g_activeMode;

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

// Fraction (0.0-1.0) of the way `matchFilename` (a full prefix+candidate+
// suffix line from the matches file) sits between `lowerBound` and
// `upperBound` in `alphabet`'s enumeration order (cpu-utils.h's
// stringToIndex - the same fixed odometer order runSearch() itself
// enumerates candidates in). Returns -1.0 if it can't be computed (mismatched
// candidate lengths - shouldn't happen for a single bounded-shaped range/
// search, but a match line surviving from a previous, differently-shaped run
// on the same file is possible - malformed alphabet, etc.); callers should
// treat that as "no usable progress signal yet" rather than a hard error.
double matchProgressFraction(const std::string& matchFilename, const std::string& prefix, const std::string& suffix,
                              const std::string& alphabet, const std::string& lowerBound, const std::string& upperBound) {
    std::string candidate = remove_prefix_and_suffix(matchFilename, prefix, suffix);
    if (candidate.length() != lowerBound.length() || lowerBound.length() != upperBound.length())
        return -1.0;
    uint64_t lowerIdx = 0, upperIdx = 0, matchIdx = 0;
    std::string error;
    if (!stringToIndex(lowerBound, alphabet, lowerIdx, error) || !stringToIndex(upperBound, alphabet, upperIdx, error) ||
        !stringToIndex(candidate, alphabet, matchIdx, error))
        return -1.0;
    if (upperIdx <= lowerIdx)
        return -1.0;
    matchIdx = std::min(std::max(matchIdx, lowerIdx), upperIdx);
    return double(matchIdx - lowerIdx) / double(upperIdx - lowerIdx);
}

// ---------------------------------------------------------------------
// Logo + About box - shared by the setup dialog and the main window.
// ---------------------------------------------------------------------

// Builds an HBITMAP of an embedded BMP file image (logo_bmp.h/icon_bmp.h)
// scaled to `w` x `h`. The caller owns it and must DeleteObject() it. HALFTONE
// stretch mode so heavily downscaled artwork averages rather than drops pixels.
HBITMAP makeBitmapFromBmp(const unsigned char* bmp, int w, int h) {
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;
    memcpy(&fileHeader, bmp, sizeof(fileHeader));
    memcpy(&infoHeader, bmp + sizeof(fileHeader), sizeof(infoHeader));
    const unsigned char* bits = bmp + fileHeader.bfOffBits;

    HDC screen = GetDC(nullptr);
    HDC mem = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, w, h);
    HGDIOBJ old = SelectObject(mem, bitmap);
    SetStretchBltMode(mem, HALFTONE);
    SetBrushOrgEx(mem, 0, 0, nullptr);
    StretchDIBits(mem, 0, 0, w, h, 0, 0, infoHeader.biWidth, infoHeader.biHeight, bits,
                  reinterpret_cast<const BITMAPINFO*>(bmp + sizeof(fileHeader)), DIB_RGB_COLORS, SRCCOPY);
    SelectObject(mem, old);
    DeleteDC(mem);
    ReleaseDC(nullptr, screen);
    return bitmap;
}

// For a SS_BITMAP static (STM_SETIMAGE); the caller deletes it once the
// control showing it is gone.
HBITMAP makeLogoBitmap(int w, int h) {
    return makeBitmapFromBmp(kLogoBmp, w, h);
}

// The application icon (icon_bmp.h) at `size` x `size` pixels - Windows asks
// for different sizes in different places (title bar/tray vs. taskbar/Alt-Tab,
// see SM_CXSMICON/SM_CXICON, and again at other DPI scalings). Each embedded
// size is rendered separately for crispness, so this uses the smallest one
// that is at least `size` (or the largest, if none is) - an exact match is
// copied 1:1 with no resampling at all, only an in-between request (e.g. 20
// or 40) gets scaled, and only slightly. Fully opaque (an all-zero AND mask).
// Never freed: a handful of icons for the life of the process.
HICON makeAppIcon(int size) {
    const IconImage* image = &kIconImages[0];
    for (const IconImage& candidate : kIconImages) {
        image = &candidate;
        if (candidate.size >= size)
            break;
    }
    HBITMAP color = makeBitmapFromBmp(image->bmp, size, size);
    std::vector<unsigned char> maskBits(((size + 15) / 16) * 2 * size, 0); // 1bpp rows are word-aligned
    HBITMAP mask = CreateBitmap(size, size, 1, 1, maskBits.data());
    ICONINFO info{};
    info.fIcon = TRUE;
    info.hbmMask = mask;
    info.hbmColor = color;
    HICON icon = CreateIconIndirect(&info);
    DeleteObject(mask);
    DeleteObject(color);
    return icon;
}

HICON appIconLarge() {
    static HICON icon = makeAppIcon(GetSystemMetrics(SM_CXICON));
    return icon;
}

HICON appIconSmall() {
    static HICON icon = makeAppIcon(GetSystemMetrics(SM_CXSMICON));
    return icon;
}

constexpr const char* kAboutClassName = "NamebreakAboutDialog";

struct AboutState {
    HWND owner = nullptr;
    bool done = false;
};

LRESULT CALLBACK AboutWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<AboutState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_COMMAND:
            if (LOWORD(wParam) == IDOK)
                SendMessage(hwnd, WM_CLOSE, 0, 0);
            return 0;
        case WM_CLOSE:
            if (state) {
                // Re-enabled *before* this window goes away, so Windows hands
                // focus back to the owner instead of some other application.
                if (state->owner)
                    EnableWindow(state->owner, TRUE);
                state->done = true;
            }
            DestroyWindow(hwnd);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

// A classic modal About box: logo, name, one-line description, and the
// coordinator protocol version this build speaks (protocol.h - the only
// version number this project actually has). `owner` is disabled while it's
// up and gets focus back afterwards; it may be hidden (e.g. opened from the
// tray menu while the main window is minimized to the tray).
void showAboutDialog(HWND owner) {
    HINSTANCE hInstance = GetModuleHandleA(nullptr);
    static bool classRegistered = false;
    if (!classRegistered) {
        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = AboutWndProc;
        wc.hInstance = hInstance;
        wc.hIcon = appIconLarge();
        wc.hIconSm = appIconSmall();
        wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
        wc.lpszClassName = kAboutClassName;
        RegisterClassExA(&wc);
        classRegistered = true;
    }

    constexpr DWORD kStyle = WS_POPUP | WS_CAPTION | WS_SYSMENU;
    constexpr DWORD kExStyle = WS_EX_DLGMODALFRAME;
    RECT rect = {0, 0, 450, 210}; // desired client area
    AdjustWindowRectEx(&rect, kStyle, FALSE, kExStyle);
    int width = rect.right - rect.left, height = rect.bottom - rect.top;

    // Centered over the owner if it's actually showing, else on the screen.
    RECT anchor;
    if (owner && IsWindowVisible(owner) && !IsIconic(owner))
        GetWindowRect(owner, &anchor);
    else
        SetRect(&anchor, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
    int x = anchor.left + ((anchor.right - anchor.left) - width) / 2;
    int y = anchor.top + ((anchor.bottom - anchor.top) - height) / 2;

    AboutState state;
    state.owner = owner;
    HWND hwnd = CreateWindowExA(kExStyle, kAboutClassName, "About namebreak", kStyle, x, y, width, height, owner, nullptr, hInstance, nullptr);
    if (!hwnd)
        return;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR) &state);

    HBITMAP logo = makeLogoBitmap(192, 128);
    HWND logoCtl = CreateWindowExA(0, "STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_BITMAP, 16, 16, 192, 128, hwnd, nullptr, hInstance, nullptr);
    SendMessage(logoCtl, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM) logo);

    std::string protocolLine = std::string("Coordinator protocol v") + kProtocolVersion;
    HWND title = CreateWindowExA(0, "STATIC", "namebreak", WS_CHILD | WS_VISIBLE, 226, 16, 210, 28, hwnd, nullptr, hInstance, nullptr);
    // Wide-character API for this one label: the byline has a non-ASCII
    // letter, which the ANSI API would only show correctly on codepages that
    // happen to contain it.
    HWND byline = CreateWindowExW(0, L"STATIC", L"By: Ojan (Johan Sj\u00f6blom)", WS_CHILD | WS_VISIBLE, 226, 44, 210, 16, hwnd, nullptr,
                                   hInstance, nullptr);
    HWND body = CreateWindowExA(0, "STATIC",
                                 "A GPU-accelerated MPQ filename brute-forcer: finds the filename behind a pair of MPQ hashes by hashing "
                                 "every candidate name on the graphics card.",
                                 WS_CHILD | WS_VISIBLE, 226, 68, 210, 58, hwnd, nullptr, hInstance, nullptr);
    HWND protocol = CreateWindowExA(0, "STATIC", protocolLine.c_str(), WS_CHILD | WS_VISIBLE, 226, 130, 210, 16, hwnd, nullptr, hInstance, nullptr);
    HWND link = CreateWindowExA(0, "STATIC", "Background: zezula.net/en/mpq/namebreak.html", WS_CHILD | WS_VISIBLE, 226, 150, 210, 32, hwnd,
                                 nullptr, hInstance, nullptr);
    HWND ok = CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 360, 176, 74, 26, hwnd, (HMENU) (INT_PTR) IDOK,
                               hInstance, nullptr);

    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    HFONT titleFont = CreateFontA(-22, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                   CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    for (HWND h : {byline, body, protocol, link, ok})
        SendMessage(h, WM_SETFONT, (WPARAM) font, TRUE);
    SendMessage(title, WM_SETFONT, (WPARAM) titleFont, TRUE);

    if (owner)
        EnableWindow(owner, FALSE);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    SetForegroundWindow(hwnd);

    MSG msg;
    while (!state.done) {
        BOOL got = GetMessage(&msg, nullptr, 0, 0);
        if (got == 0) {
            // WM_QUIT arrived while we were modal (the app is shutting down) -
            // hand it back to the outer loop and get out of its way.
            PostQuitMessage((int) msg.wParam);
            if (owner)
                EnableWindow(owner, TRUE);
            if (IsWindow(hwnd))
                DestroyWindow(hwnd);
            break;
        }
        if (got < 0)
            break;
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    DeleteObject(logo);
    DeleteObject(titleFont);
}

// ---------------------------------------------------------------------
// Config readiness / first-run setup
// ---------------------------------------------------------------------

bool configHasServerUrl(const ConfigFile& config) {
    auto it = config.coordinator.find("server_url");
    return it != config.coordinator.end() && !it->second.empty();
}

// True if `config` has everything its own `mode` needs to actually run
// (mirrors buildCoordinatorArgs'/buildSearchRequest's own required-key
// lists, config.h) - an unset or unrecognized mode is never "ready", same as
// the CLI's own main() would refuse it.
bool isConfigReady(const ConfigFile& config) {
    if (config.mode == "coordinator")
        return configHasServerUrl(config);
    if (config.mode == "bounded" || config.mode == "continuous") {
        SearchRequest probe;
        std::string error;
        return buildSearchRequest(config.search, config.mode == "continuous", probe, error);
    }
    return false;
}

// appendKeyToConfigSection (config.h) requires the file AND the target
// [sectionName] header to already exist - true for a config.conf a CLI run
// already created (as long as that run used the same section), but not for
// one this GUI is creating from scratch, and not for one that so far only
// ever had the *other* section (e.g. a hand-written [search]-only file, and
// the setup dialog's Coordinator tab was used). Appends a bare
// "[sectionName]" header to the end of the file if one isn't already
// present; a no-op (returns true) if it already is.
bool ensureSectionExists(const std::string& path, const std::string& sectionName, std::string& error) {
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
            if (trimmed == "[" + sectionName + "]")
                return true;
        }
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        error = "cannot append to " + path;
        return false;
    }
    out << "\n[" << sectionName << "]\n";
    return true;
}

// Sets (or inserts) config.conf's single top-level `mode = ...` line - the
// one key that lives before any [section] header (config.h) - to `mode`.
// Unlike ensureSectionExists/appendKeyToConfigSection below, this can
// *replace* an existing value: switching which tab of the setup dialog was
// used (e.g. Coordinator -> Local Search on a config.conf that already had a
// mode from a previous run) needs the old mode value gone, not just another
// line added alongside it - config.cpp's parser would otherwise just take
// whichever `mode = ...` line comes last, silently ignoring the new choice
// if it happened to land above the stale one.
bool setModeKey(const std::string& path, const std::string& mode, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    in.close();

    auto trim = [](const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            return std::string();
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    };

    int firstSectionLine = -1;
    int modeLine = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::string trimmed = trim(lines[i]);
        if (!trimmed.empty() && trimmed.front() == '[') {
            firstSectionLine = (int) i;
            break;
        }
        if (trimmed.rfind("mode", 0) == 0) {
            size_t eq = trimmed.find('=');
            if (eq != std::string::npos && trim(trimmed.substr(0, eq)) == "mode") {
                modeLine = (int) i;
                break;
            }
        }
    }

    if (modeLine >= 0) {
        lines[modeLine] = "mode = " + mode;
    } else if (firstSectionLine >= 0) {
        lines.insert(lines.begin() + firstSectionLine, "mode = " + mode);
    } else {
        lines.insert(lines.begin(), "mode = " + mode);
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        error = "cannot rewrite " + path;
        return false;
    }
    for (const auto& l : lines)
        out << l << "\n";
    return true;
}

// Makes sure `path` is set up to run in `mode`: its `mode = ...` line is set
// (see setModeKey), its [sectionName] section exists, and every non-empty
// `keys` entry is present in that section - creating the file from scratch
// if it doesn't exist yet at all. Values already set in the file (that the
// caller left untouched, e.g. re-running setup after only partially filling
// it in before) are never overwritten - same "fill in only what's missing"
// behavior the CLI's own resolveMissingIdentity (coordinator_runner.cpp)
// already has for username/hostname, just generalized to any section/mode.
bool ensureConfigForMode(const std::string& path, const std::string& mode, const std::string& sectionName,
                          const std::vector<std::pair<std::string, std::string>>& keys, std::string& error) {
    std::ifstream probe(path);
    bool fileExists = probe.good();
    probe.close();

    if (!fileExists) {
        std::ofstream out(path, std::ios::trunc);
        if (!out) {
            error = "cannot create " + path;
            return false;
        }
        out << "mode = " << mode << "\n\n[" << sectionName << "]\n";
        for (const auto& kv : keys) {
            if (!kv.second.empty())
                out << kv.first << " = " << kv.second << "\n";
        }
        return true;
    }

    if (!setModeKey(path, mode, error))
        return false;
    if (!ensureSectionExists(path, sectionName, error))
        return false;

    ConfigFile existing;
    std::string loadErr;
    if (!loadConfigFile(path, existing, loadErr)) {
        error = loadErr;
        return false;
    }
    const std::map<std::string, std::string>& existingSection = (sectionName == "coordinator") ? existing.coordinator : existing.search;

    for (const auto& kv : keys) {
        if (kv.second.empty())
            continue;
        auto it = existingSection.find(kv.first);
        bool hasNonEmpty = it != existingSection.end() && !it->second.empty();
        if (hasNonEmpty)
            continue;
        if (!appendKeyToConfigSection(path, sectionName, kv.first, kv.second)) {
            error = "failed to write " + kv.first + " to " + path;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------
// First-run setup dialog - a hand-rolled modal window (CreateWindowEx +
// its own nested message loop) rather than a .rc dialog template, so the
// whole GUI stays a single .cpp with no resource-compiler build step. Only
// ever shown once, before the main window/worker thread exist, so there's
// no other message loop running concurrently to conflict with.
//
// One SysTabControl32 with two tabs - Coordinator, and Local Search (which
// covers both bounded and continuous, since they share the exact same
// [search] config keys and differ only in that one mode value - a radio
// pair inside the tab picks between them). Each tab's controls live on their
// own plain page window laid over the tab control's body (see
// SetupPageWndProc for why), created up front; only the active tab's page is
// shown at a time (see showTabPage). Whichever tab is active when OK is
// pressed is what gets validated/submitted - the user picks a mode by
// picking a tab, not through a separate control. Every option has a blue "?"
// beside it whose tooltip explains it.
// ---------------------------------------------------------------------

struct SetupDialogFields {
    // Which tab/radio to preselect - from an existing (even if incomplete)
    // config.conf's own mode, so re-opening setup on a partially-filled-in
    // file resumes where it left off instead of always defaulting to
    // Coordinator.
    int initialTab = kTabCoordinator;
    bool initialContinuous = false;

    // Coordinator tab - see CoordinatorArgs/buildCoordinatorArgs.
    std::string username;
    std::string hostname;
    std::string serverUrl;
    std::string pollIntervalSecs = "30";

    // Local Search tab - see SearchRequest/buildSearchRequest.
    std::string alphabet;
    std::string maxBackslashCount = "0";
    std::string prefix;
    std::string suffix;
    std::string startCandidate;
    std::string lowerBound;
    std::string upperBound;
    std::string hashA;
    std::string hashB;
    bool pruneSymbolRuns = false;

    // Result: "coordinator" | "bounded" | "continuous", set only if OK was
    // pressed (see showSetupDialog's return value).
    std::string mode;
};

// Marker stored in a help "?" control's GWLP_USERDATA so SetupPageWndProc can
// tell it apart from an ordinary label when coloring it.
constexpr LONG_PTR kHelpMarker = 0x4E42484C;
constexpr const char* kSetupPageClassName = "NamebreakSetupPage";

struct SetupDialogState {
    HWND hwndTab = nullptr;
    HWND hwndTooltip = nullptr;
    // One plain child window per tab, covering the tab control's body and
    // holding that tab's controls; showTabPage shows exactly one at a time.
    HWND hwndPageCoordinator = nullptr;
    HWND hwndPageSearch = nullptr;
    HWND hwndUsername = nullptr, hwndHostname = nullptr, hwndServerUrl = nullptr, hwndPollInterval = nullptr;
    HWND hwndBoundedRadio = nullptr, hwndContinuousRadio = nullptr, hwndPrune = nullptr;
    HWND hwndAlphabet = nullptr, hwndMaxBackslash = nullptr, hwndPrefix = nullptr, hwndSuffix = nullptr, hwndStartCandidate = nullptr,
         hwndLowerBound = nullptr, hwndUpperBound = nullptr, hwndHashA = nullptr, hwndHashB = nullptr;
    // Tooltip strings must outlive the tooltip (it keeps pointers, not
    // copies); a deque never moves existing elements as it grows.
    std::deque<std::string> helpTexts;
    SetupDialogFields* fields = nullptr; // caller-owned; filled in on accept
    bool accepted = false;
    bool done = false;
};

void showTabPage(SetupDialogState* state, int sel) {
    ShowWindow(state->hwndPageCoordinator, sel == kTabCoordinator ? SW_SHOW : SW_HIDE);
    ShowWindow(state->hwndPageSearch, sel == kTabLocalSearch ? SW_SHOW : SW_HIDE);
}

// The tab pages sit over the tab control's own body (rather than being
// drawn straight onto it) so their labels' background is the same plain
// dialog gray as the page itself - with visual styles on, the tab control's
// own body is drawn a different shade, which would leave every label as a
// visible gray box. All this adds over a stock window is coloring the help
// "?" marks.
LRESULT CALLBACK SetupPageWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_CTLCOLORSTATIC && GetWindowLongPtr((HWND) lParam, GWLP_USERDATA) == kHelpMarker) {
        HDC hdc = (HDC) wParam;
        SetTextColor(hdc, RGB(0, 70, 200));
        SetBkColor(hdc, GetSysColor(COLOR_BTNFACE));
        return (LRESULT) GetSysColorBrush(COLOR_BTNFACE);
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

LRESULT CALLBACK SetupDialogWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<SetupDialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_NOTIFY: {
            if (!state)
                break;
            auto* hdr = reinterpret_cast<NMHDR*>(lParam);
            if (hdr->hwndFrom == state->hwndTab && hdr->code == TCN_SELCHANGE)
                showTabPage(state, (int) SendMessage(state->hwndTab, TCM_GETCURSEL, 0, 0));
            return 0;
        }
        case WM_COMMAND:
            if (!state)
                break;
            switch (LOWORD(wParam)) {
                case kIdSetupAbout:
                    showAboutDialog(hwnd);
                    return 0;
                case kIdSetupOk: {
                    char buf[512];
                    auto getField = [&](HWND h) {
                        GetWindowTextA(h, buf, sizeof(buf));
                        return trimCopy(buf);
                    };
                    int sel = (int) SendMessage(state->hwndTab, TCM_GETCURSEL, 0, 0);
                    if (sel == kTabCoordinator) {
                        std::string serverUrl = getField(state->hwndServerUrl);
                        if (serverUrl.empty()) {
                            MessageBoxA(hwnd, "Server URL is required.", "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        state->fields->username = getField(state->hwndUsername);
                        state->fields->hostname = getField(state->hwndHostname);
                        state->fields->serverUrl = serverUrl;
                        state->fields->pollIntervalSecs = getField(state->hwndPollInterval);
                        state->fields->mode = "coordinator";
                    } else {
                        std::string alphabet = getField(state->hwndAlphabet);
                        std::string maxBackslash = getField(state->hwndMaxBackslash);
                        std::string prefix = getField(state->hwndPrefix);
                        std::string suffix = getField(state->hwndSuffix);
                        std::string startCandidate = getField(state->hwndStartCandidate);
                        std::string lowerBound = getField(state->hwndLowerBound);
                        std::string upperBound = getField(state->hwndUpperBound);
                        std::string hashA = getField(state->hwndHashA);
                        std::string hashB = getField(state->hwndHashB);
                        // Same required-key list as buildSearchRequest (config.cpp)
                        // - checked here too so a missing field is caught right
                        // where it was left blank, rather than only after
                        // writing+reloading the file.
                        if (alphabet.empty() || maxBackslash.empty() || prefix.empty() || suffix.empty() || startCandidate.empty() ||
                            lowerBound.empty() || upperBound.empty() || hashA.empty() || hashB.empty()) {
                            MessageBoxA(hwnd, "All Local Search fields are required.", "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        state->fields->alphabet = alphabet;
                        state->fields->maxBackslashCount = maxBackslash;
                        state->fields->prefix = prefix;
                        state->fields->suffix = suffix;
                        state->fields->startCandidate = startCandidate;
                        state->fields->lowerBound = lowerBound;
                        state->fields->upperBound = upperBound;
                        state->fields->hashA = hashA;
                        state->fields->hashB = hashB;
                        state->fields->pruneSymbolRuns = SendMessage(state->hwndPrune, BM_GETCHECK, 0, 0) == BST_CHECKED;
                        bool continuous = SendMessage(state->hwndContinuousRadio, BM_GETCHECK, 0, 0) == BST_CHECKED;
                        state->fields->mode = continuous ? "continuous" : "bounded";
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

BOOL CALLBACK setFontEnumProc(HWND child, LPARAM font) {
    SendMessage(child, WM_SETFONT, (WPARAM) font, TRUE);
    return TRUE;
}

// Shows the setup form pre-filled from `fields`; on OK, fills `fields` with
// whatever tab was active and its values (plus `fields.mode`) and returns
// true. Returns false (leaving `fields` untouched) on Cancel/close.
bool showSetupDialog(HINSTANCE hInstance, SetupDialogFields& fields) {
    static bool classRegistered = false;
    if (!classRegistered) {
        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = SetupDialogWndProc;
        wc.hInstance = hInstance;
        wc.hIcon = appIconLarge();
        wc.hIconSm = appIconSmall();
        wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
        wc.lpszClassName = kSetupClassName;
        RegisterClassExA(&wc);
        wc.lpfnWndProc = SetupPageWndProc;
        wc.lpszClassName = kSetupPageClassName;
        RegisterClassExA(&wc);
        classRegistered = true;
    }

    SetupDialogState state;
    state.fields = &fields;

    constexpr DWORD kStyle = WS_POPUP | WS_CAPTION | WS_SYSMENU;
    constexpr DWORD kExStyle = WS_EX_DLGMODALFRAME;
    RECT rect = {0, 0, 620, 550}; // desired client area
    AdjustWindowRectEx(&rect, kStyle, FALSE, kExStyle);
    HWND hwndDialog = CreateWindowExA(kExStyle, kSetupClassName, "namebreak - first-run setup", kStyle, CW_USEDEFAULT, CW_USEDEFAULT,
                                       rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, hInstance, nullptr);
    if (!hwndDialog)
        return false;
    SetWindowLongPtr(hwndDialog, GWLP_USERDATA, (LONG_PTR) &state);

    // --- Header: logo + what this window is for ---
    HBITMAP logo = makeLogoBitmap(120, 80);
    HWND logoCtl = CreateWindowExA(0, "STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_BITMAP, 12, 10, 120, 80, hwndDialog, nullptr, hInstance, nullptr);
    SendMessage(logoCtl, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM) logo);
    HWND title = CreateWindowExA(0, "STATIC", "namebreak", WS_CHILD | WS_VISIBLE, 150, 14, 450, 28, hwndDialog, nullptr, hInstance, nullptr);
    CreateWindowExA(0, "STATIC",
                     "First-run setup. Choose how this machine should search, then press OK - your choices are saved to the configuration "
                     "file and used on every later start. Hover over a blue ? for help with any option.",
                     WS_CHILD | WS_VISIBLE, 150, 46, 455, 46, hwndDialog, nullptr, hInstance, nullptr);

    // --- Tabs, each with a page window over the tab body ---
    state.hwndTab = CreateWindowExA(0, WC_TABCONTROLA, "", WS_CHILD | WS_VISIBLE, 10, 100, 600, 400, hwndDialog, nullptr, hInstance, nullptr);
    TCITEMA tie{};
    tie.mask = TCIF_TEXT;
    tie.pszText = (LPSTR) "Coordinator";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabCoordinator, (LPARAM) &tie);
    tie.pszText = (LPSTR) "Local Search";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabLocalSearch, (LPARAM) &tie);
    state.hwndPageCoordinator = CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, 592, 364, hwndDialog, nullptr, hInstance, nullptr);
    state.hwndPageSearch = CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, 592, 364, hwndDialog, nullptr, hInstance, nullptr);

    // One tooltip window serves every help "?" mark. Long delay before it
    // disappears (the texts are a few lines) and a max width so they wrap.
    state.hwndTooltip = CreateWindowExA(WS_EX_TOPMOST, TOOLTIPS_CLASSA, nullptr, WS_POPUP | TTS_NOPREFIX | TTS_ALWAYSTIP, CW_USEDEFAULT,
                                         CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, hwndDialog, nullptr, hInstance, nullptr);
    SendMessage(state.hwndTooltip, TTM_SETMAXTIPWIDTH, 0, 340);
    SendMessage(state.hwndTooltip, TTM_SETDELAYTIME, TTDT_INITIAL, 250);
    SendMessage(state.hwndTooltip, TTM_SETDELAYTIME, TTDT_AUTOPOP, 30000);

    // ES_AUTOHSCROLL matters here, not just cosmetically: a single-line EDIT
    // control without it refuses to accept any more typed/pasted characters
    // than fit in its visible width at once (no scrolling, no overflow) -
    // it's not a length *limit* so much as a hard stop once the box looks
    // full. Long values (e.g. a real server URL) would otherwise silently
    // get truncated right there, with no error and no obvious reason why.
    constexpr DWORD kEditStyle = WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL;
    auto makeLabel = [&](HWND page, const char* text, int x, int y, int w, int h = 18) {
        return CreateWindowExA(0, "STATIC", text, WS_CHILD | WS_VISIBLE, x, y, w, h, page, nullptr, hInstance, nullptr);
    };
    auto makeEdit = [&](HWND page, const std::string& initial, int x, int y, int w) {
        return CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", initial.c_str(), kEditStyle, x, y, w, 22, page, nullptr, hInstance, nullptr);
    };
    auto makeRadio = [&](HWND page, const char* text, int x, int y, int w, int id, bool startsGroup) {
        DWORD style = WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON | (startsGroup ? WS_GROUP : 0);
        return CreateWindowExA(0, "BUTTON", text, style, x, y, w, 20, page, (HMENU) (INT_PTR) id, hInstance, nullptr);
    };
    auto makeCheckbox = [&](HWND page, const char* text, int x, int y, int w, bool checked) {
        HWND h = CreateWindowExA(0, "BUTTON", text, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, x, y, w, 20, page,
                                  (HMENU) (INT_PTR) kIdSetupPruneCheckbox, hInstance, nullptr);
        SendMessage(h, BM_SETCHECK, checked ? BST_CHECKED : BST_UNCHECKED, 0);
        return h;
    };
    // A blue "?" that shows `text` as a tooltip on hover. SS_NOTIFY is what
    // makes a static control receive mouse messages at all (they're
    // otherwise transparent to the mouse), and TTF_SUBCLASS lets the tooltip
    // watch those messages itself, with no relaying from our message loop.
    auto addHelp = [&](HWND page, int x, int y, const char* text) {
        HWND q = CreateWindowExA(0, "STATIC", "?", WS_CHILD | WS_VISIBLE | WS_BORDER | SS_CENTER | SS_NOTIFY, x, y, 18, 18, page, nullptr,
                                  hInstance, nullptr);
        SetWindowLongPtr(q, GWLP_USERDATA, kHelpMarker);
        state.helpTexts.emplace_back(text);
        TOOLINFOA ti{};
        ti.cbSize = sizeof(ti);
        ti.uFlags = TTF_SUBCLASS | TTF_IDISHWND;
        ti.hwnd = page;
        ti.uId = (UINT_PTR) q;
        ti.lpszText = (LPSTR) state.helpTexts.back().c_str();
        SendMessage(state.hwndTooltip, TTM_ADDTOOLA, 0, (LPARAM) &ti);
        return q;
    };

    // --- Coordinator page (one column: label, field, help) ---
    HWND pc = state.hwndPageCoordinator;
    makeLabel(pc, "Join a shared search. This client registers with a coordinator server, is handed a slice of a target's search space, "
                  "searches it on this machine's GPU, reports back, and repeats until you quit. The server decides what to search - "
                  "you only say who you are and where the server is.",
              8, 6, 570, 62);
    constexpr int kCoordRow0 = 82, kRowStep = 34;
    makeLabel(pc, "Username:", 8, kCoordRow0 + 3, 100);
    state.hwndUsername = makeEdit(pc, fields.username, 112, kCoordRow0, 330);
    addHelp(pc, 450, kCoordRow0 + 2,
            "The name this client registers under on the coordinator server.\r\nLeave blank to use the detected Windows user name.");
    makeLabel(pc, "Hostname:", 8, kCoordRow0 + kRowStep + 3, 100);
    state.hwndHostname = makeEdit(pc, fields.hostname, 112, kCoordRow0 + kRowStep, 330);
    addHelp(pc, 450, kCoordRow0 + kRowStep + 2,
            "Identifies this machine to the coordinator server.\r\nLeave blank to use the detected computer name.");
    makeLabel(pc, "Server URL:", 8, kCoordRow0 + 2 * kRowStep + 3, 100);
    state.hwndServerUrl = makeEdit(pc, fields.serverUrl, 112, kCoordRow0 + 2 * kRowStep, 330);
    addHelp(pc, 450, kCoordRow0 + 2 * kRowStep + 2,
            "Required. The base address of the coordinator server, for example https://coordinator.example.com (no trailing path).");
    makeLabel(pc, "Poll interval (s):", 8, kCoordRow0 + 3 * kRowStep + 3, 104);
    state.hwndPollInterval = makeEdit(pc, fields.pollIntervalSecs, 112, kCoordRow0 + 3 * kRowStep, 80);
    addHelp(pc, 200, kCoordRow0 + 3 * kRowStep + 2,
            "How many seconds to wait before asking the server for work again when none was available. It is also the starting delay "
            "after a failed request, which then backs off. Default: 30.");

    // --- Local Search page (two columns: label, field, help) ---
    HWND ps = state.hwndPageSearch;
    makeLabel(ps, "Search on your own, without a server. Describe the filename you're after - its fixed prefix and suffix, the characters "
                  "its unknown middle may contain, and its two MPQ hash values - and this machine's GPU tries every candidate in the "
                  "range. Bounded stops once the range is exhausted; Continuous never stops on its own.",
              8, 6, 570, 62);
    constexpr int kSearchRow0 = 82;
    // Column geometry: label x / edit x / help x, for the left and right columns.
    constexpr int kL1 = 8, kE1 = 112, kH1 = 270, kL2 = 306, kE2 = 420, kH2 = 548;
    auto row = [&](int n) { return kSearchRow0 + n * kRowStep; };

    makeLabel(ps, "Search type:", kL1, row(0) + 3, 100);
    state.hwndBoundedRadio = makeRadio(ps, "Bounded", kE1, row(0) + 1, 90, kIdSetupBoundedRadio, true);
    state.hwndContinuousRadio = makeRadio(ps, "Continuous", kE1 + 100, row(0) + 1, 100, kIdSetupContinuousRadio, false);
    SendMessage(fields.initialContinuous ? state.hwndContinuousRadio : state.hwndBoundedRadio, BM_SETCHECK, BST_CHECKED, 0);
    addHelp(ps, kE1 + 208, row(0) + 2,
            "Bounded searches only candidates of exactly the start candidate's length, between the lower and upper bound, then stops.\r\n"
            "Continuous does the same, then moves on to longer candidates and keeps going until you quit.");

    makeLabel(ps, "Alphabet:", kL1, row(1) + 3, 100);
    state.hwndAlphabet = makeEdit(ps, fields.alphabet, kE1, row(1), 150);
    addHelp(ps, kH1, row(1) + 2,
            "Every character a candidate may contain, typed out with no separators. The number of characters must be 42, 43, 47, 48, 49 "
            "or 50.");
    makeLabel(ps, "Max backslash:", kL2, row(1) + 3, 110);
    state.hwndMaxBackslash = makeEdit(ps, fields.maxBackslashCount, kE2, row(1), 60);
    addHelp(ps, kE2 + 68, row(1) + 2,
            "The most backslashes a candidate may contain before it is skipped. 0 means no limit (to forbid backslashes entirely, leave "
            "them out of the alphabet instead).");

    makeLabel(ps, "Prefix:", kL1, row(2) + 3, 100);
    state.hwndPrefix = makeEdit(ps, fields.prefix, kE1, row(2), 150);
    addHelp(ps, kH1, row(2) + 2, "The fixed text every filename starts with, before the unknown part being searched for.");
    makeLabel(ps, "Suffix:", kL2, row(2) + 3, 110);
    state.hwndSuffix = makeEdit(ps, fields.suffix, kE2, row(2), 120);
    addHelp(ps, kH2, row(2) + 2, "The fixed text every filename ends with, after the unknown part (for example a file extension).");

    makeLabel(ps, "Start candidate:", kL1, row(3) + 3, 104);
    state.hwndStartCandidate = makeEdit(ps, fields.startCandidate, kE1, row(3), 150);
    addHelp(ps, kH1, row(3) + 2,
            "The full filename (prefix + candidate + suffix) to begin searching from. Lets a long search resume where it left off.");
    state.hwndPrune = makeCheckbox(ps, "Prune symbol runs", kL2, row(3) + 1, 150, fields.pruneSymbolRuns);
    addHelp(ps, kH2, row(3) + 2,
            "Skip candidates containing three or more symbols in a row that are neither letters, digits nor spaces - real filenames "
            "practically never have those. Makes the search faster.");

    makeLabel(ps, "Lower bound:", kL1, row(4) + 3, 100);
    state.hwndLowerBound = makeEdit(ps, fields.lowerBound, kE1, row(4), 150);
    addHelp(ps, kH1, row(4) + 2, "The full filename (inclusive) the search range starts at.");
    makeLabel(ps, "Upper bound:", kL2, row(4) + 3, 110);
    state.hwndUpperBound = makeEdit(ps, fields.upperBound, kE2, row(4), 120);
    addHelp(ps, kH2, row(4) + 2, "The full filename (inclusive) the search range ends at.");

    makeLabel(ps, "Hash A (hex):", kL1, row(5) + 3, 100);
    state.hwndHashA = makeEdit(ps, fields.hashA, kE1, row(5), 150);
    addHelp(ps, kH1, row(5) + 2, "The first of the two 32-bit MPQ hashes of the filename you're looking for, in hex (a 0x prefix is optional).");
    makeLabel(ps, "Hash B (hex):", kL2, row(5) + 3, 110);
    state.hwndHashB = makeEdit(ps, fields.hashB, kE2, row(5), 120);
    addHelp(ps, kH2, row(5) + 2, "The second of the two 32-bit MPQ hashes, in hex (a 0x prefix is optional).");

    // --- Buttons ---
    CreateWindowExA(0, "BUTTON", "About...", WS_CHILD | WS_VISIBLE, 12, 512, 90, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupAbout, hInstance,
                     nullptr);
    CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 430, 512, 80, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupOk,
                     hInstance, nullptr);
    CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE, 520, 512, 80, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupCancel, hInstance,
                     nullptr);

    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    HFONT titleFont = CreateFontA(-24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                   CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    EnumChildWindows(hwndDialog, setFontEnumProc, (LPARAM) font);
    SendMessage(title, WM_SETFONT, (WPARAM) titleFont, TRUE);

    SendMessage(state.hwndTab, TCM_SETCURSEL, fields.initialTab, 0);
    showTabPage(&state, fields.initialTab);

    ShowWindow(hwndDialog, SW_SHOW);
    UpdateWindow(hwndDialog);
    SetForegroundWindow(hwndDialog);

    MSG msg;
    while (!state.done && GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    DeleteObject(logo);
    DeleteObject(titleFont);
    return state.accepted;
}

// What the rest of the GUI needs to actually run, whichever mode was
// configured - exactly one of coordinatorArgs/searchRequest is meaningful,
// selected by `mode` (see workerThreadMain).
struct AppConfig {
    std::string mode;
    CoordinatorArgs coordinatorArgs;
    SearchRequest searchRequest;
};

// Loads `configPath`, running the setup dialog first if it's missing or
// doesn't have everything its own mode needs (see isConfigReady). Returns
// false (having already reported why, via MessageBox) if the user cancelled
// setup or a config error couldn't be resolved.
bool prepareConfig(HINSTANCE hInstance, const std::string& configPath, AppConfig& outConfig) {
    ConfigFile config;
    std::string error;
    bool loaded = loadConfigFile(configPath, config, error);
    bool ready = loaded && isConfigReady(config);

    if (!ready) {
        SetupDialogFields fields;
        // Prefill from whatever's already in the file (even if incomplete,
        // e.g. only some [search] keys were ever set), falling back to
        // auto-detected username/hostname and sensible defaults for
        // everything else - same spirit as the CLI's own
        // resolveMissingIdentity (coordinator_runner.cpp).
        auto get = [&](const std::map<std::string, std::string>& section, const std::string& key, const std::string& fallback) {
            if (!loaded)
                return fallback;
            auto it = section.find(key);
            return it != section.end() ? it->second : fallback;
        };
        fields.username = get(config.coordinator, "username", resolveUsername());
        fields.hostname = get(config.coordinator, "hostname", resolveHostname());
        fields.serverUrl = get(config.coordinator, "server_url", kDefaultServerUrl);
        fields.pollIntervalSecs = get(config.coordinator, "poll_interval_secs", "30");
        fields.alphabet = get(config.search, "alphabet", "");
        fields.maxBackslashCount = get(config.search, "max_backslash_count", "0");
        fields.prefix = get(config.search, "prefix", "");
        fields.suffix = get(config.search, "suffix", "");
        fields.startCandidate = get(config.search, "start_candidate", "");
        fields.lowerBound = get(config.search, "lower_bound", "");
        fields.upperBound = get(config.search, "upper_bound", "");
        fields.hashA = get(config.search, "hash_a", "");
        fields.hashB = get(config.search, "hash_b", "");
        fields.pruneSymbolRuns = get(config.search, "prune_symbol_runs", "false") == "true";
        fields.initialTab = (loaded && (config.mode == "bounded" || config.mode == "continuous")) ? kTabLocalSearch : kTabCoordinator;
        fields.initialContinuous = loaded && config.mode == "continuous";

        if (!showSetupDialog(hInstance, fields))
            return false; // user cancelled - nothing to report

        std::string writeError;
        bool writeOk;
        if (fields.mode == "coordinator") {
            writeOk = ensureConfigForMode(configPath, "coordinator", "coordinator",
                                           {{"server_url", fields.serverUrl},
                                            {"username", fields.username},
                                            {"hostname", fields.hostname},
                                            {"poll_interval_secs", fields.pollIntervalSecs}},
                                           writeError);
        } else {
            writeOk = ensureConfigForMode(configPath, fields.mode, "search",
                                           {{"alphabet", fields.alphabet},
                                            {"max_backslash_count", fields.maxBackslashCount},
                                            {"prefix", fields.prefix},
                                            {"suffix", fields.suffix},
                                            {"start_candidate", fields.startCandidate},
                                            {"lower_bound", fields.lowerBound},
                                            {"upper_bound", fields.upperBound},
                                            {"hash_a", fields.hashA},
                                            {"hash_b", fields.hashB},
                                            {"prune_symbol_runs", fields.pruneSymbolRuns ? "true" : "false"}},
                                           writeError);
        }
        if (!writeOk) {
            MessageBoxA(nullptr, ("Failed to write " + configPath + ": " + writeError).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
        if (!loadConfigFile(configPath, config, error)) {
            MessageBoxA(nullptr, ("Failed to reload " + configPath + " after setup: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
    }

    outConfig.mode = config.mode;
    if (config.mode == "coordinator") {
        if (!buildCoordinatorArgs(config.coordinator, outConfig.coordinatorArgs, error)) {
            MessageBoxA(nullptr, (configPath + " [coordinator]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
        outConfig.coordinatorArgs.configPath = configPath;
    } else if (config.mode == "bounded" || config.mode == "continuous") {
        if (!buildSearchRequest(config.search, config.mode == "continuous", outConfig.searchRequest, error)) {
            MessageBoxA(nullptr, (configPath + " [search]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
    } else {
        MessageBoxA(nullptr, ("Unknown or missing mode in " + configPath).c_str(), "namebreak", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// Worker thread - runs whichever mode was configured (see AppConfig) on a
// background thread, since both runCoordinator() and runSearch() block for
// the life of the run. Always ends with a WM_APP_WORKER_STOPPED post back to
// the UI thread (see MainWndProc) - never any other Win32 call from here,
// same rule SharedStatus's own comment already states for the coordinator
// callbacks below.
// ---------------------------------------------------------------------

void onRangeClaimed(const ClaimResponse& claim, const SearchRequest& req) {
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.targetName = claim.targetName;
    g_status.outputFilePath = req.outputFilePath;
    // A coordinator range is always bounded (a fixed candidate length - see
    // toSearchRequest, coordinator_runner.cpp), so it always has a real
    // finish line to measure progress against - see SharedStatus's comment.
    g_status.alphabet = req.alphabet;
    g_status.prefix = req.prefix;
    g_status.suffix = req.suffix;
    g_status.lowerBound = req.lowerBound;
    g_status.upperBound = req.upperBound;
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

// Runs a single bounded/continuous search directly (no coordinator server
// involved - see main.cpp's main() for the CLI's equivalent one-shot
// call). Bounded mode searches one fixed candidate length - a real finish
// line, same as a coordinator range - so alphabet/lowerBound/upperBound are
// populated for it too. Continuous mode has no such line (it moves on to
// ever-longer candidate lengths indefinitely - see README.md's "Modes"), so
// those are left empty and updateUiFromSharedState shows the indeterminate
// marquee style for its whole run instead of a fraction that would stop
// meaning anything once it moves past its first length.
void runLocalSearch(const SearchRequest& req, bool continuous) {
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        g_status.targetName = req.prefix + "..." + req.suffix;
        g_status.outputFilePath = req.outputFilePath;
        if (continuous) {
            g_status.alphabet.clear();
            g_status.prefix.clear();
            g_status.suffix.clear();
            g_status.lowerBound.clear();
            g_status.upperBound.clear();
        } else {
            g_status.alphabet = req.alphabet;
            g_status.prefix = req.prefix;
            g_status.suffix = req.suffix;
            g_status.lowerBound = req.lowerBound;
            g_status.upperBound = req.upperBound;
        }
        g_status.hasActiveRange = true;
        g_status.rangeJustFinished = false;
        g_status.statusText = std::string("Running ") + (continuous ? "continuous" : "bounded") + " search...";
    }
    SearchResult result = runSearch(req, &g_quitRequested, nullptr, &g_pauseRequested);
    std::lock_guard<std::mutex> lock(g_status.mutex);
    g_status.hasActiveRange = false;
    g_status.rangeJustFinished = true;
    if (!result.ok)
        g_status.statusText = "Search failed: " + result.error;
    else if (result.aborted)
        g_status.statusText = "Aborted";
    else if (result.found)
        g_status.statusText = "Match found: " + result.filename;
    else
        g_status.statusText = "Search exhausted, no match";
}

void workerThreadMain(AppConfig config) {
    if (config.mode == "coordinator") {
        CoordinatorCallbacks callbacks;
        callbacks.onRangeClaimed = onRangeClaimed;
        callbacks.onRangeFinished = onRangeFinished;
        callbacks.onStatus = onStatus;
        runCoordinator(config.coordinatorArgs, &g_pauseRequested, &g_quitRequested, &callbacks);
    } else {
        runLocalSearch(config.searchRequest, config.mode == "continuous");
    }
    // Marshal back onto the UI thread rather than touching any Win32 API
    // from here - see WM_APP_WORKER_STOPPED in MainWndProc.
    PostMessage(g_hwndMain, WM_APP_WORKER_STOPPED, 0, 0);
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

void togglePause() {
    bool newState = !g_pauseRequested.load(std::memory_order_relaxed);
    g_pauseRequested.store(newState, std::memory_order_relaxed);
    SetWindowTextA(g_hwndPauseButton, newState ? "Resume" : "Pause");
    // No pause-duration bookkeeping needed here (unlike an earlier,
    // time-based version of this progress bar): matches only ever get
    // appended while actually searching, so the match-based progress
    // fraction (see matchProgressFraction/updateUiFromSharedState) simply
    // stops advancing on its own while paused, with nothing to unwind on
    // resume.
}

void requestQuit(HWND hwnd) {
    if (g_quitting)
        return; // already shutting down - ignore a repeat click
    const char* message = g_activeMode == "coordinator"
                               ? "Quit namebreak? This stops searching now; the current range's lease will simply expire and get reassigned."
                               : "Quit namebreak? This stops the search now.";
    int result = MessageBoxA(hwnd, message, "Confirm Quit", MB_YESNO | MB_ICONQUESTION);
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
    std::string alphabet, prefix, suffix, lowerBound, upperBound;
    bool hasActiveRange, rangeJustFinished;
    {
        std::lock_guard<std::mutex> lock(g_status.mutex);
        targetName = g_status.targetName;
        outputFilePath = g_status.outputFilePath;
        statusText = g_status.statusText;
        alphabet = g_status.alphabet;
        prefix = g_status.prefix;
        suffix = g_status.suffix;
        lowerBound = g_status.lowerBound;
        upperBound = g_status.upperBound;
        hasActiveRange = g_status.hasActiveRange;
        rangeJustFinished = g_status.rangeJustFinished;
        g_status.rangeJustFinished = false; // one-shot edge, consumed here
    }

    bool paused = g_pauseRequested.load(std::memory_order_relaxed);

    SetWindowTextA(g_hwndTargetLabel, (targetName.empty() ? "Target: (none yet)" : "Target: " + targetName).c_str());
    SetWindowTextA(g_hwndStatusLabel, ("Status: " + statusText + (paused ? " (paused)" : "")).c_str());

    // Read once, shared below by both the progress calculation (its last
    // line is the most recent Hash-A match) and the matches box itself.
    std::vector<std::string> matchLines;
    if (!outputFilePath.empty())
        matchLines = readLastLines(outputFilePath, kMaxMatchLines);

    // Determinate (a real fraction, from matchProgressFraction below) only
    // for bounded-shaped work, which has a fixed finish line to measure
    // against - a coordinator range, or local bounded mode (see
    // onRangeClaimed/runLocalSearch, which leave lowerBound empty exactly
    // when there isn't one: continuous mode, or nothing running yet).
    bool determinate = hasActiveRange && !lowerBound.empty();
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
        if (!matchLines.empty()) {
            double f = matchProgressFraction(matchLines.back(), prefix, suffix, alphabet, lowerBound, upperBound);
            if (f >= 0.0)
                fraction = f;
        }
        SendMessage(g_hwndProgress, PBM_SETPOS, (WPARAM) (fraction * 100.0), 0);
    } else if (rangeJustFinished) {
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

    if (!outputFilePath.empty()) {
        std::string joined;
        for (size_t i = 0; i < matchLines.size(); ++i) {
            joined += matchLines[i];
            if (i + 1 < matchLines.size())
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
    g_hwndAboutButton = CreateWindowExA(0, "BUTTON", "About...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 170, 405, 150, 30, hwnd,
                                         (HMENU) (INT_PTR) kIdAboutButton, hInstance, nullptr);
    g_hwndQuitButton = CreateWindowExA(0, "BUTTON", "Quit", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 330, 405, 150, 30, hwnd,
                                        (HMENU) (INT_PTR) kIdQuitButton, hInstance, nullptr);

    for (HWND child : {g_hwndTargetLabel, g_hwndStatusLabel, g_hwndMatches, g_hwndPauseButton, g_hwndAboutButton, g_hwndQuitButton})
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

    g_workerThread = std::thread(workerThreadMain, config);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int) msg.wParam;
}
