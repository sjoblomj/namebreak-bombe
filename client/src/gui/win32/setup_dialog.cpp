// First-run setup dialog - a hand-rolled modal window (CreateWindowEx + its
// own nested message loop) rather than a .rc dialog template, so the GUI
// needs no resource-compiler build step. Shown before the main window and
// the worker thread exist at first run, and again - over a running search -
// from the tray menu's Settings... (editSettings, app_config.cpp), which is
// the one case where this window's own nested message loop runs inside the
// main window's: see the loop at the end of showSetupDialog for what that
// takes.
//
// One SysTabControl32 with four tabs - Coordinator; Local Search (which
// covers both bounded and continuous, since they share the exact same
// [search] config keys and differ only in that one mode value - a radio
// pair inside the tab picks between them); Local Dictionary, for the
// [dictionary] section; and Meet in the Middle, for the
// [meet-in-the-middle] one. Each tab's controls live on their
// own plain page window laid over the tab control's body (see
// SetupPageWndProc for why), created up front; only the active tab's page is
// shown at a time (see showTabPage). Whichever tab is active when OK is
// pressed is what gets validated/submitted - the user picks a mode by
// picking a tab, not through a separate control. Every option has a blue "?"
// beside it whose tooltip explains it.

#include "gui/win32/setup_dialog.h"

#include <commdlg.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#include "backends/backends.h"
#include "common/config.h"
#include "common/string_util.h"
#include "engine/limits.h"
#include "engine/meet_in_the_middle.h"
#include "engine/meet_in_the_middle_search.h"
#include "engine/mpq_hash.h"
#include "gui/win32/about_dialog.h"
#include "gui/win32/app_icons.h"
#include "gui/win32/dpi.h"
#include "gui/win32/optimize_dialog.h"

#pragma comment(lib, "comdlg32.lib")

namespace {

using gui::DictionaryFields;
using gui::HashTableFields;
using gui::MeetInTheMiddleFields;
using gui::SetupDialogFields;
using gui::kTabCoordinator;
using gui::kTabDictionary;
using gui::kTabLocalSearch;
using gui::kTabMeetInTheMiddle;
using gui::scaled;
using gui::showAboutDialog;

constexpr const char* kSetupClassName = "NamebreakSetupDialog";

constexpr int kIdSetupOk = 301;
constexpr int kIdSetupCancel = 302;
constexpr int kIdSetupBoundedRadio = 303;
constexpr int kIdSetupContinuousRadio = 304;
constexpr int kIdSetupPruneCheckbox = 305;
constexpr int kIdSetupAbout = 306;
constexpr int kIdSetupPruneBracketsCheckbox = 307;
constexpr int kIdSetupPruneWholeCheckbox = 308;
constexpr int kIdSetupPruneAdjacentCheckbox = 309;
constexpr int kIdSetupBrowseWordLists = 310;
constexpr int kIdSetupEncryptionKey = 311;
constexpr int kIdSetupTableMemory = 312;
constexpr int kIdSetupVerification = 313;
constexpr int kIdSetupOptimize = 314;

// The dialog's geometry: everything but the buttons is kContentHeight of a
// scroll view, and the buttons sit in a strip of their own under it. The
// pages have room for the Verification sections whether they're shown or
// not (they're kVerificationHeight of it), so that opening them never
// resizes the dialog - see layoutDialog. All of it in 96-DPI pixels, as every
// control is placed - see scaled() (dpi.h).
constexpr int kDialogWidth = 620, kContentHeight = 642, kButtonStripHeight = 44, kButtonGap = 6;
constexpr int kTabWidth = 600, kTabHeight = 536;
constexpr int kPageWidth = 592, kPageHeight = 500;
constexpr int kVerificationHeight = 46;

// Marker stored in a help "?" control's GWLP_USERDATA so SetupPageWndProc can
// tell it apart from an ordinary label when coloring it.
constexpr LONG_PTR kHelpMarker = 0x4E42484C;
constexpr const char* kSetupPageClassName = "NamebreakSetupPage";
constexpr const char* kSetupScrollClassName = "NamebreakSetupScroll";

// A tab's hash table size field and its two start slot fields - see
// HashTableFields.
struct HashTableControls {
    HWND tableSize = nullptr, firstStartSlot = nullptr, lastStartSlot = nullptr;
};

struct SetupDialogState {
    HWND hwndDialog = nullptr;
    // Holds everything but the buttons, and scrolls it when the screen is too
    // short for all of it - see layoutDialog. scrollPos is how far down.
    HWND hwndScroll = nullptr;
    int scrollPos = 0;
    HWND hwndTab = nullptr;
    HWND hwndOptimize = nullptr, hwndOk = nullptr, hwndCancel = nullptr;
    HWND hwndTooltip = nullptr;
    // One plain child window per tab, covering the tab control's body and
    // holding that tab's controls; showTabPage shows exactly one at a time.
    HWND hwndPageCoordinator = nullptr;
    HWND hwndPageSearch = nullptr;
    HWND hwndPageDictionary = nullptr;
    HWND hwndPageMeetInTheMiddle = nullptr;
    HWND hwndUsername = nullptr, hwndHostname = nullptr, hwndServerUrl = nullptr, hwndPollInterval = nullptr;
    // The Coordinator tab's memory slider and the line under it that says
    // what the stop it's at means - see tableMemoryLabel.
    HWND hwndTableMemory = nullptr, hwndTableMemoryText = nullptr;
    std::vector<uint64_t> tableMemoryStops;
    uint64_t availablePhysicalBytes = 0;
    HWND hwndBoundedRadio = nullptr, hwndContinuousRadio = nullptr, hwndPrune = nullptr, hwndPruneBrackets = nullptr,
         hwndPruneWhole = nullptr, hwndPruneAdjacent = nullptr;
    HWND hwndInsertFromStart = nullptr, hwndInsertFromEnd = nullptr;
    HWND hwndAlphabet = nullptr, hwndMaxBackslash = nullptr, hwndMinBackslash = nullptr, hwndPrefix = nullptr, hwndSuffix = nullptr, hwndStartCandidate = nullptr,
         hwndLowerBound = nullptr, hwndUpperBound = nullptr, hwndHashA = nullptr, hwndHashB = nullptr, hwndKey = nullptr;
    HashTableControls searchHashTable;
    // The Local Dictionary tab's.
    HWND hwndDictBuiltin = nullptr, hwndDictLists = nullptr, hwndDictMinWords = nullptr, hwndDictMaxWords = nullptr,
         hwndDictSeparators = nullptr, hwndDictTails = nullptr, hwndDictPrefix = nullptr, hwndDictSuffix = nullptr,
         hwndDictLowerBound = nullptr, hwndDictUpperBound = nullptr, hwndDictHashA = nullptr, hwndDictHashB = nullptr,
         hwndDictKey = nullptr, hwndDictMatchesName = nullptr, hwndDictRecordBasenames = nullptr, hwndDictRecordHashA = nullptr,
         hwndDictResume = nullptr;
    HashTableControls dictHashTable;
    // The Meet in the Middle tab's.
    HWND hwndMitmAlphabet = nullptr, hwndMitmPrefix = nullptr, hwndMitmSuffix = nullptr, hwndMitmASeed1 = nullptr,
         hwndMitmASeed2 = nullptr, hwndMitmBSeed1 = nullptr, hwndMitmBSeed2 = nullptr, hwndMitmHashA = nullptr, hwndMitmHashB = nullptr,
         hwndMitmMinLength = nullptr, hwndMitmMaxLength = nullptr, hwndMitmTableLength = nullptr, hwndMitmTableMemory = nullptr,
         hwndMitmMinBackslash = nullptr, hwndMitmMaxBackslash = nullptr, hwndMitmPrune = nullptr, hwndMitmPruneBrackets = nullptr,
         hwndMitmPruneAdjacent = nullptr, hwndMitmMatchesName = nullptr, hwndMitmResume = nullptr, hwndMitmKey = nullptr;
    HashTableControls mitmHashTable;
    // Every tab's Verification chevron, and the controls they show and hide -
    // one switch for all the tabs, so the dialog stays the same size whichever
    // tab is shown.
    std::vector<HWND> verificationChevrons, verificationControls;
    bool verificationShown = false;
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
    ShowWindow(state->hwndPageDictionary, sel == kTabDictionary ? SW_SHOW : SW_HIDE);
    ShowWindow(state->hwndPageMeetInTheMiddle, sel == kTabMeetInTheMiddle ? SW_SHOW : SW_HIDE);
}

// Scrolls the scroll view to `pos`, as far as its range goes, moving its
// children with it.
void scrollContentTo(SetupDialogState* state, int pos) {
    SCROLLINFO si{};
    si.cbSize = sizeof si;
    si.fMask = SIF_RANGE | SIF_PAGE;
    GetScrollInfo(state->hwndScroll, SB_VERT, &si);
    pos = (std::max) (0, (std::min) (pos, si.nMax + 1 - (int) si.nPage));
    if (pos != state->scrollPos)
        ScrollWindowEx(state->hwndScroll, 0, state->scrollPos - pos, nullptr, nullptr, nullptr, nullptr,
                       SW_SCROLLCHILDREN | SW_INVALIDATE | SW_ERASE);
    state->scrollPos = pos;
    si.fMask = SIF_POS;
    si.nPos = pos;
    SetScrollInfo(state->hwndScroll, SB_VERT, &si, TRUE);
}

// Sizes the dialog, before it's shown, to its content - as tall as the
// screen's work area lets it be, and no taller: what doesn't fit then
// scrolls, in the scroll view that holds everything but the buttons, so that
// the buttons are always on screen.
void layoutDialog(SetupDialogState* state) {
    const int dialogWidth = scaled(kDialogWidth), contentHeight = scaled(kContentHeight), buttonStripHeight = scaled(kButtonStripHeight);
    RECT frame = {0, 0, 0, 0};
    AdjustWindowRectEx(&frame, (DWORD) GetWindowLongPtr(state->hwndDialog, GWL_STYLE), FALSE,
                       (DWORD) GetWindowLongPtr(state->hwndDialog, GWL_EXSTYLE));
    const int frameWidth = frame.right - frame.left, frameHeight = frame.bottom - frame.top;
    MONITORINFO monitor{};
    monitor.cbSize = sizeof monitor;
    GetMonitorInfoA(MonitorFromWindow(state->hwndDialog, MONITOR_DEFAULTTONEAREST), &monitor);
    const RECT& work = monitor.rcWork;
    const int viewHeight =
        (std::max) (buttonStripHeight, (std::min) (contentHeight, (int) (work.bottom - work.top) - frameHeight - buttonStripHeight));
    const int scrollbar = viewHeight < contentHeight ? GetSystemMetrics(SM_CXVSCROLL) : 0;
    const int width = dialogWidth + scrollbar + frameWidth, height = viewHeight + buttonStripHeight + frameHeight;
    // Moved up, if it has to be, to keep the bottom - and the buttons - on
    // screen.
    RECT window;
    GetWindowRect(state->hwndDialog, &window);
    const int top = (std::max) ((int) work.top, (std::min) ((int) window.top, (int) work.bottom - height));
    SetWindowPos(state->hwndDialog, nullptr, window.left, top, width, height, SWP_NOZORDER | SWP_NOACTIVATE);

    SetWindowPos(state->hwndScroll, nullptr, 0, 0, dialogWidth + scrollbar, viewHeight, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    SCROLLINFO si{};
    si.cbSize = sizeof si;
    si.fMask = SIF_RANGE | SIF_PAGE;
    si.nMax = contentHeight - 1;
    si.nPage = (UINT) viewHeight;
    SetScrollInfo(state->hwndScroll, SB_VERT, &si, TRUE);
    scrollContentTo(state, state->scrollPos);

    const int buttonY = viewHeight + scaled(kButtonGap);
    SetWindowPos(state->hwndOptimize, nullptr, scaled(12), buttonY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    SetWindowPos(state->hwndOk, nullptr, scaled(430) + scrollbar, buttonY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    SetWindowPos(state->hwndCancel, nullptr, scaled(520) + scrollbar, buttonY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
}

// Shows or hides every tab's Verification fields, in the room the pages keep
// for them. Win32 has no expander control, so this is
// the one Windows' own dialogs draw (a task dialog's "See details"): a
// chevron - Marlett's 4 points right, its 6 down - beside a heading, both
// clickable.
void showVerification(SetupDialogState* state, bool shown) {
    state->verificationShown = shown;
    for (HWND h : state->verificationChevrons)
        SetWindowTextA(h, shown ? "6" : "4");
    for (HWND h : state->verificationControls)
        ShowWindow(h, shown ? SW_SHOW : SW_HIDE);
    // On a screen too short for them, the fields just opened are below the
    // bottom of the view: bring them up into it.
    if (shown)
        scrollContentTo(state, state->scrollPos + scaled(kVerificationHeight));
}

bool isChecked(HWND h) {
    return SendMessage(h, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

std::string windowText(HWND h) {
    std::string text(GetWindowTextLengthA(h) + 1, '\0');
    text.resize(GetWindowTextA(h, &text[0], (int) text.size()));
    return text;
}

// --- The Coordinator tab's meet-in-the-middle memory slider ---------------
//
// How much memory this client will give a meet-in-the-middle target's table
// is a coarse choice: a character more of the table needs about as many
// times more memory as the alphabet has characters, so the useful values are
// powers of two far apart, and a slider over them says more than a number
// box would (what you are giving, out of what this machine has).

// The sizes the slider stops at - none, then powers of two from 256 MB up.
constexpr uint64_t kTableMemoryStops[] = {0,            256ull << 20, 512ull << 20, 1ull << 30,  2ull << 30,
                                           4ull << 30,  8ull << 30,   16ull << 30,  32ull << 30, 64ull << 30};

// Those of them a machine with `totalBytes` of memory could hold at all -
// always at least None and the smallest size, so there is something to drag
// even on a very small machine.
std::vector<uint64_t> tableMemoryStopsFor(uint64_t totalBytes) {
    std::vector<uint64_t> stops;
    for (uint64_t stop : kTableMemoryStops)
        if (stop <= totalBytes || stops.size() < 2)
            stops.push_back(stop);
    return stops;
}

// What to propose on a machine this size: an eighth of its memory at most,
// and never more than 4 GB unasked - enough for five characters of a
// 43-character alphabet, which is what the directories looked for in
// practice need.
uint64_t proposedTableMemory(const std::vector<uint64_t>& stops, uint64_t totalBytes) {
    const uint64_t cap = (std::min) (totalBytes / 8, 4ull << 30);
    uint64_t proposed = 0;
    for (uint64_t stop : stops)
        if (stop <= cap)
            proposed = stop;
    return proposed;
}

// The stop at or below `bytes` - which stop a value from config.conf sits
// at, rounded down so that the slider never offers more than the file asked
// for.
int stopIndexFor(const std::vector<uint64_t>& stops, uint64_t bytes) {
    int index = 0;
    for (int i = 0; i < (int) stops.size(); ++i)
        if (stops[i] <= bytes)
            index = i;
    return index;
}

// A size as config.conf writes it: 0, 256M, 4G.
std::string bytesForConfig(uint64_t bytes) {
    if (bytes == 0)
        return "0";
    if (bytes % (1ull << 30) == 0)
        return std::to_string(bytes >> 30) + "G";
    return std::to_string(bytes >> 20) + "M";
}

// A size as this dialog says it: None, 256 MB, 4 GB (every stop is a whole
// number of its unit).
std::string stopText(uint64_t bytes) {
    if (bytes == 0)
        return "None";
    if (bytes < (1ull << 30))
        return std::to_string(bytes >> 20) + " MB";
    return std::to_string(bytes >> 30) + " GB";
}

// A machine's memory, as the line above the slider says it: 31.8 GB.
std::string memoryText(uint64_t bytes) {
    char buf[32];
    snprintf(buf, sizeof buf, "%.1f GB", (double) bytes / (double) (1ull << 30));
    return buf;
}

// What a stop means, for the line under the slider: the size, and how long a
// table of it holds - which is the whole point of giving the memory. Two
// alphabet sizes, because what one buys depends on the target's own alphabet
// and the steps are wide: 2 GB and 32 GB are the same five characters of a
// 43-character alphabet, but six and seven of a 26-character one.
std::string tableMemoryLabel(uint64_t bytes, uint64_t availableBytes) {
    if (bytes == 0)
        return "None: the server hands this client no meet-in-the-middle work.";
    std::string text = stopText(bytes) + " - a table of " +
                       std::to_string(meetInTheMiddleTableLengthForBytes(43, MAX_CANDIDATE_LEN, bytes)) +
                       " characters of a 43-character alphabet, " +
                       std::to_string(meetInTheMiddleTableLengthForBytes(26, MAX_CANDIDATE_LEN, bytes)) + " of a 26-character one.";
    if (bytes > availableBytes)
        text += "\r\nThat is more than is free on this machine right now.";
    return text;
}

void syncTableMemoryText(SetupDialogState* state) {
    const int pos = (int) SendMessage(state->hwndTableMemory, TBM_GETPOS, 0, 0);
    const uint64_t bytes = state->tableMemoryStops[(size_t) (pos < 0 ? 0 : pos)];
    SetWindowTextA(state->hwndTableMemoryText, tableMemoryLabel(bytes, state->availablePhysicalBytes).c_str());
}

// What `controls` say, trimmed.
HashTableFields readHashTable(const HashTableControls& controls) {
    HashTableFields fields;
    fields.tableSize = trim(windowText(controls.tableSize));
    fields.firstStartSlot = trim(windowText(controls.firstStartSlot));
    fields.lastStartSlot = trim(windowText(controls.lastStartSlot));
    return fields;
}

// The hash table keys for `fields`, unquoted - empty for a key to leave out,
// which a blank field is.
std::vector<std::pair<std::string, std::string>> hashTableValues(const HashTableFields& fields) {
    return {
        {"hash_table_size", fields.tableSize},
        {"hash_table_first_start_slot", fields.firstStartSlot},
        {"hash_table_last_start_slot", fields.lastStartSlot},
    };
}

// Every [meet-in-the-middle] key for `fields`, with the values as a loaded
// ConfigFile's own map holds them - unquoted. Empty for a key to leave out.
// gui::meetInTheMiddleConfigKeys adds the quoting the file itself needs.
std::vector<std::pair<std::string, std::string>> meetInTheMiddleValues(const MeetInTheMiddleFields& f) {
    std::vector<std::pair<std::string, std::string>> values = {
        {"alphabet", f.alphabet},
        {"prefix", f.prefix},
        {"suffix", f.suffix},
        {"meeting_point_a_seed1", f.meetingPointASeed1},
        {"meeting_point_a_seed2", f.meetingPointASeed2},
        {"meeting_point_b_seed1", f.meetingPointBSeed1},
        {"meeting_point_b_seed2", f.meetingPointBSeed2},
        {"hash_a", f.hashA},
        {"hash_b", f.hashB},
        {"min_length", f.minLength},
        {"max_length", f.maxLength},
        // Left out, rather than empty, for a table as long as the memory allows.
        {"table_length", f.tableLength},
        {"table_memory", f.tableMemory},
        {"max_backslash_count", f.maxBackslashCount},
        {"min_backslash_count", f.minBackslashCount},
        {"prune_symbol_runs", f.pruneSymbolRuns ? "true" : "false"},
        {"prune_unopened_brackets", f.pruneUnopenedBrackets ? "true" : "false"},
        {"prune_adjacent_backslashes", f.pruneAdjacentBackslashes ? "true" : "false"},
        {"encryption_key", f.encryptionKey},
        {"matches_name", f.matchesName},
        {"resume_from_last_candidate", f.resume ? "true" : "false"},
    };
    for (const auto& kv : hashTableValues(f.hashTable))
        values.push_back(kv);
    return values;
}

// Recording basenames means something only with a key to compare them to.
void syncRecordBasenames(SetupDialogState* state) {
    EnableWindow(state->hwndDictRecordBasenames, !trim(windowText(state->hwndDictKey)).empty());
}

// `path` relative to the current directory, which config.conf's relative
// paths are read from, if it's in it - otherwise as it is.
std::string relativeToCurrentDirectory(const std::string& path) {
    char cwd[MAX_PATH];
    const DWORD n = GetCurrentDirectoryA(MAX_PATH, cwd);
    if (n == 0 || n >= MAX_PATH || cwd[n - 1] == '\\')
        return path;
    if (path.size() > n + 1 && _strnicmp(path.c_str(), cwd, n) == 0 && path[n] == '\\')
        return path.substr(n + 1);
    return path;
}

// Lets the user pick word lists, and adds them to the Word lists field's
// list - each quoted if it has a comma, which would otherwise end it.
void browseWordLists(HWND owner, SetupDialogState* state) {
    std::string buffer(32768, '\0');
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = "Word lists (*.txt)\0*.txt\0All files (*.*)\0*.*\0";
    ofn.lpstrFile = &buffer[0];
    ofn.nMaxFile = (DWORD) buffer.size();
    ofn.lpstrTitle = "Choose word lists";
    // OFN_NOCHANGEDIR: the dialog would otherwise move the process's
    // current directory, which config.conf's relative paths are read from.
    ofn.Flags = OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameA(&ofn))
        return;
    // One file: its whole path. Several: the directory, then each name, each
    // ended by a '\0' and the last by two.
    std::vector<std::string> parts;
    for (const char* p = buffer.c_str(); *p; p += strlen(p) + 1)
        parts.emplace_back(p);
    std::vector<std::string> paths;
    if (parts.size() == 1)
        paths = parts;
    for (size_t i = 1; i < parts.size(); ++i)
        paths.push_back(parts[0] + (parts[0].back() == '\\' ? "" : "\\") + parts[i]);
    std::string list = trim(windowText(state->hwndDictLists));
    for (std::string path : paths) {
        path = relativeToCurrentDirectory(path);
        list += (list.empty() ? "" : ", ") + (path.find(',') != std::string::npos ? "\"" + path + "\"" : path);
    }
    SetWindowTextA(state->hwndDictLists, list.c_str());
}

// The tab pages sit over the tab control's own body (rather than being
// drawn straight onto it) so their labels' background is the same plain
// dialog gray as the page itself - with visual styles on, the tab control's
// own body is drawn a different shade, which would leave every label as a
// visible gray box. All this adds over a stock window is coloring the help
// "?" marks, and handing its controls' notifications on to the dialog.
LRESULT CALLBACK SetupPageWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    // WM_HSCROLL is how a trackbar reports being dragged, and it goes to its
    // parent - this page - the same way a button's WM_COMMAND does.
    if (msg == WM_COMMAND || msg == WM_HSCROLL)
        return SendMessage(GetParent(hwnd), msg, wParam, lParam);
    // The Verification heading is clicked like a link, so it points like one.
    if (msg == WM_SETCURSOR && GetDlgCtrlID((HWND) wParam) == kIdSetupVerification) {
        SetCursor(LoadCursor(nullptr, IDC_HAND));
        return TRUE;
    }
    if (msg == WM_CTLCOLORSTATIC && GetWindowLongPtr((HWND) lParam, GWLP_USERDATA) == kHelpMarker) {
        HDC hdc = (HDC) wParam;
        SetTextColor(hdc, RGB(0, 70, 200));
        SetBkColor(hdc, GetSysColor(COLOR_BTNFACE));
        return (LRESULT) GetSysColorBrush(COLOR_BTNFACE);
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

// The scroll view (see layoutDialog): scrolls with its scrollbar and the
// mouse wheel - which a control that doesn't use it hands on up to here -
// and hands everything its children report on to the dialog, as a page does.
LRESULT CALLBACK SetupScrollWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<SetupDialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_COMMAND:
        case WM_HSCROLL:
        case WM_NOTIFY:
            return SendMessage(GetParent(hwnd), msg, wParam, lParam);
        case WM_VSCROLL: {
            if (!state)
                break;
            SCROLLINFO si{};
            si.cbSize = sizeof si;
            si.fMask = SIF_ALL;
            GetScrollInfo(hwnd, SB_VERT, &si);
            int pos = state->scrollPos;
            switch (LOWORD(wParam)) {
                case SB_LINEUP: pos -= scaled(20); break;
                case SB_LINEDOWN: pos += scaled(20); break;
                case SB_PAGEUP: pos -= (int) si.nPage; break;
                case SB_PAGEDOWN: pos += (int) si.nPage; break;
                case SB_THUMBTRACK:
                case SB_THUMBPOSITION: pos = si.nTrackPos; break;
                case SB_TOP: pos = 0; break;
                case SB_BOTTOM: pos = si.nMax; break;
            }
            scrollContentTo(state, pos);
            return 0;
        }
        case WM_MOUSEWHEEL:
            if (!state)
                break;
            scrollContentTo(state, state->scrollPos - GET_WHEEL_DELTA_WPARAM(wParam) * scaled(60) / WHEEL_DELTA);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

LRESULT CALLBACK SetupDialogWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<SetupDialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_HSCROLL:
            if (state && (HWND) lParam == state->hwndTableMemory)
                syncTableMemoryText(state);
            return 0;
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
                case kIdSetupOptimize:
                    gui::optimizeForHardware(hwnd, state->fields->configPath, state->fields->backend, state->fields->runningMode,
                                             state->fields->searchPause);
                    return 0;
                case kIdSetupBrowseWordLists:
                    browseWordLists(hwnd, state);
                    return 0;
                case kIdSetupEncryptionKey:
                    if (HIWORD(wParam) == EN_CHANGE)
                        syncRecordBasenames(state);
                    return 0;
                case kIdSetupVerification:
                    // A quick second click is a double-click to a static.
                    if (HIWORD(wParam) == STN_CLICKED || HIWORD(wParam) == STN_DBLCLK)
                        showVerification(state, !state->verificationShown);
                    return 0;
                case kIdSetupOk: {
                    char buf[512];
                    auto getField = [&](HWND h) {
                        GetWindowTextA(h, buf, sizeof(buf));
                        return trim(buf);
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
                        const int memoryPos = (int) SendMessage(state->hwndTableMemory, TBM_GETPOS, 0, 0);
                        state->fields->meetInTheMiddleTableMemory =
                            bytesForConfig(state->tableMemoryStops[(size_t) (memoryPos < 0 ? 0 : memoryPos)]);
                        state->fields->mode = "coordinator";
                    } else if (sel == kTabDictionary) {
                        DictionaryFields d;
                        d.builtinDictionary = isChecked(state->hwndDictBuiltin);
                        d.dictionaries = getField(state->hwndDictLists);
                        d.minWords = getField(state->hwndDictMinWords);
                        d.maxWords = getField(state->hwndDictMaxWords);
                        d.separators = getField(state->hwndDictSeparators);
                        d.tails = getField(state->hwndDictTails);
                        d.prefix = getField(state->hwndDictPrefix);
                        d.suffix = getField(state->hwndDictSuffix);
                        d.lowerBound = getField(state->hwndDictLowerBound);
                        d.upperBound = getField(state->hwndDictUpperBound);
                        d.hashA = getField(state->hwndDictHashA);
                        d.hashB = getField(state->hwndDictHashB);
                        d.encryptionKey = getField(state->hwndDictKey);
                        d.recordBasenames = isChecked(state->hwndDictRecordBasenames);
                        d.recordHashAMatches = isChecked(state->hwndDictRecordHashA);
                        d.hashTable = readHashTable(state->dictHashTable);
                        d.matchesName = getField(state->hwndDictMatchesName);
                        d.resume = isChecked(state->hwndDictResume);
                        if (d.maxWords.empty() || d.hashA.empty() || d.hashB.empty()) {
                            MessageBoxA(hwnd, "Max words, Hash A and Hash B are required.", "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        // Everything else a dictionary search checks before it
                        // starts - the word lists among it - checked here, where
                        // it can still be put right, the same way it will be.
                        ConfigFile probe;
                        probe.matchesDir = state->fields->matchesDir;
                        for (const auto& kv : gui::dictionaryConfigKeys(d))
                            if (!kv.second.empty())
                                probe.dictionary[kv.first] = kv.second;
                        DictionaryRequest request;
                        std::vector<std::string> warnings;
                        std::string error;
                        HCURSOR cursor = SetCursor(LoadCursorA(nullptr, IDC_WAIT));
                        const bool built = buildDictionaryRequest(probe, request, warnings, error);
                        SetCursor(cursor);
                        if (!built) {
                            MessageBoxA(hwnd, ("Dictionary: " + error).c_str(), "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        state->fields->dictionary = d;
                        state->fields->mode = "dictionary";
                    } else if (sel == kTabMeetInTheMiddle) {
                        MeetInTheMiddleFields m;
                        // The alphabet, prefix and suffix as typed, not
                        // trimmed: a space is a character like any other in
                        // them, and the usual alphabet has one.
                        m.alphabet = windowText(state->hwndMitmAlphabet);
                        m.prefix = windowText(state->hwndMitmPrefix);
                        m.suffix = windowText(state->hwndMitmSuffix);
                        m.meetingPointASeed1 = getField(state->hwndMitmASeed1);
                        m.meetingPointASeed2 = getField(state->hwndMitmASeed2);
                        m.meetingPointBSeed1 = getField(state->hwndMitmBSeed1);
                        m.meetingPointBSeed2 = getField(state->hwndMitmBSeed2);
                        m.hashA = getField(state->hwndMitmHashA);
                        m.hashB = getField(state->hwndMitmHashB);
                        m.minLength = getField(state->hwndMitmMinLength);
                        m.maxLength = getField(state->hwndMitmMaxLength);
                        m.tableLength = getField(state->hwndMitmTableLength);
                        m.tableMemory = getField(state->hwndMitmTableMemory);
                        m.maxBackslashCount = getField(state->hwndMitmMaxBackslash);
                        m.minBackslashCount = getField(state->hwndMitmMinBackslash);
                        m.pruneSymbolRuns = isChecked(state->hwndMitmPrune);
                        m.pruneUnopenedBrackets = isChecked(state->hwndMitmPruneBrackets);
                        m.pruneAdjacentBackslashes = isChecked(state->hwndMitmPruneAdjacent);
                        m.encryptionKey = getField(state->hwndMitmKey);
                        m.hashTable = readHashTable(state->mitmHashTable);
                        m.matchesName = getField(state->hwndMitmMatchesName);
                        m.resume = isChecked(state->hwndMitmResume);
                        // The same required-key list as
                        // buildMeetInTheMiddleRequest (config.cpp) - hashB's
                        // two seeds aren't in it, and the probe below is
                        // what says so if only one of them is filled in.
                        if (m.alphabet.empty() || m.suffix.empty() || m.maxLength.empty() || m.hashA.empty() || m.hashB.empty() ||
                            m.meetingPointASeed1.empty() || m.meetingPointASeed2.empty()) {
                            MessageBoxA(hwnd,
                                        "Alphabet, Suffix, Max length, both hashes and both of Hash A's meeting point seeds are "
                                        "required.",
                                        "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        // Everything else the search checks before it starts,
                        // checked here where it can still be put right - the
                        // values themselves, and that hashing the suffix onto
                        // the meeting point really gives the file's hashes,
                        // which is the mistake worth catching early (the
                        // seeds and the hashes come from different places).
                        ConfigFile probe;
                        probe.matchesDir = state->fields->matchesDir;
                        for (const auto& kv : meetInTheMiddleValues(m))
                            if (!kv.second.empty())
                                probe.meetInTheMiddle[kv.first] = kv.second;
                        MeetInTheMiddleRequest request;
                        std::string error;
                        if (!buildMeetInTheMiddleRequest(probe, request, error)) {
                            MessageBoxA(hwnd, ("Meet in the Middle: " + error).c_str(), "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        std::vector<uint32_t> cryptTable(0x500);
                        prepareCryptTable(cryptTable.data());
                        if (!meetingPointFitsHashes(request, cryptTable.data(), error)) {
                            MessageBoxA(hwnd, ("Meet in the Middle: " + error).c_str(), "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        state->fields->meetInTheMiddle = m;
                        state->fields->mode = "meet-in-the-middle";
                    } else {
                        SetupDialogFields f = *state->fields;
                        f.alphabet = getField(state->hwndAlphabet);
                        f.maxBackslashCount = getField(state->hwndMaxBackslash);
                        const std::string minBackslash = getField(state->hwndMinBackslash);
                        f.minBackslashCount = minBackslash.empty() ? "0" : minBackslash;
                        f.prefix = getField(state->hwndPrefix);
                        f.suffix = getField(state->hwndSuffix);
                        f.startCandidate = getField(state->hwndStartCandidate);
                        f.lowerBound = getField(state->hwndLowerBound);
                        f.upperBound = getField(state->hwndUpperBound);
                        f.hashA = getField(state->hwndHashA);
                        f.hashB = getField(state->hwndHashB);
                        f.pruneSymbolRuns = isChecked(state->hwndPrune);
                        f.pruneUnopenedBrackets = isChecked(state->hwndPruneBrackets);
                        f.pruneWholeCandidate = isChecked(state->hwndPruneWhole);
                        f.pruneAdjacentBackslashes = isChecked(state->hwndPruneAdjacent);
                        f.insertFromStart = getField(state->hwndInsertFromStart);
                        f.insertFromEnd = getField(state->hwndInsertFromEnd);
                        f.encryptionKey = getField(state->hwndKey);
                        f.hashTable = readHashTable(state->searchHashTable);
                        const bool continuous = isChecked(state->hwndContinuousRadio);
                        // Same required-key list as buildSearchRequest (config.cpp)
                        // - checked here too so a missing field is caught right
                        // where it was left blank, rather than only after
                        // writing+reloading the file.
                        if (f.alphabet.empty() || f.maxBackslashCount.empty() || f.prefix.empty() || f.suffix.empty() ||
                            f.lowerBound.empty() || f.upperBound.empty() || f.hashA.empty() || f.hashB.empty()) {
                            MessageBoxA(hwnd, "Alphabet, Prefix, Suffix, both bounds, both hashes and Max backslash are required.",
                                        "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        // And the values themselves - the encryption key and
                        // the hash table among them - as the search will check
                        // them, while they can still be put right.
                        ConfigFile probe;
                        probe.matchesDir = f.matchesDir;
                        for (const auto& kv : gui::searchConfigKeys(f))
                            if (!kv.second.empty())
                                probe.search[kv.first] = kv.second;
                        SearchRequest request;
                        std::string error;
                        if (!buildSearchRequest(probe, continuous, request, error)) {
                            MessageBoxA(hwnd, ("Local Search: " + error).c_str(), "namebreak setup", MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        *state->fields = f;
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

} // namespace

namespace gui {

bool showSetupDialog(HINSTANCE hInstance, SetupDialogFields& fields, bool reopened) {
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
        wc.lpfnWndProc = SetupScrollWndProc;
        wc.lpszClassName = kSetupScrollClassName;
        RegisterClassExA(&wc);
        classRegistered = true;
    }

    SetupDialogState state;
    state.fields = &fields;

    // WS_CLIPCHILDREN here and on the scroll view: each would otherwise
    // paint its background over its children - the buttons, the pages - as
    // it's resized or scrolled, and leave them blank.
    constexpr DWORD kStyle = WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_CLIPCHILDREN;
    constexpr DWORD kExStyle = WS_EX_DLGMODALFRAME;
    RECT rect = {0, 0, kDialogWidth, kContentHeight + kButtonStripHeight}; // desired client area; see layoutDialog
    AdjustWindowRectEx(&rect, kStyle, FALSE, kExStyle);
    HWND hwndDialog = CreateWindowExA(kExStyle, kSetupClassName, reopened ? "namebreak - settings" : "namebreak - first-run setup",
                                       kStyle, CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top, nullptr,
                                       nullptr, hInstance, nullptr);
    if (!hwndDialog)
        return false;
    SetWindowLongPtr(hwndDialog, GWLP_USERDATA, (LONG_PTR) &state);
    state.hwndDialog = hwndDialog;
    state.hwndScroll = CreateWindowExA(0, kSetupScrollClassName, "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_CLIPCHILDREN, 0, 0, kDialogWidth,
                                       kContentHeight, hwndDialog, nullptr, hInstance, nullptr);
    SetWindowLongPtr(state.hwndScroll, GWLP_USERDATA, (LONG_PTR) &state);
    // Everything but the buttons goes in it.
    HWND content = state.hwndScroll;

    // --- Header: logo + what this window is for ---
    // Its image is set once the controls are scaled - see below.
    HWND logoCtl = CreateWindowExA(0, "STATIC", nullptr, WS_CHILD | WS_VISIBLE | SS_BITMAP, 12, 10, 120, 80, content, nullptr, hInstance, nullptr);
    HWND title = CreateWindowExA(0, "STATIC", "namebreak-bombe", WS_CHILD | WS_VISIBLE, 150, 14, 350, 28, content, nullptr, hInstance, nullptr);
    // On the title's line, at the right: it scrolls with the header.
    CreateWindowExA(0, "BUTTON", "About...", WS_CHILD | WS_VISIBLE, kDialogWidth - 12 - 90, 14, 90, 26, content,
                    (HMENU) (INT_PTR) kIdSetupAbout, hInstance, nullptr);
    CreateWindowExA(0, "STATIC",
                     reopened ? "Settings. Change how this machine searches, then press OK - your choices replace what the configuration "
                                "file says, and are used the next time namebreak starts. Hover over a blue ? for help with any option."
                              : "First-run setup. Choose how this machine should search, then press OK - your choices are saved to the "
                                "configuration file and used on every later start. Hover over a blue ? for help with any option.",
                     WS_CHILD | WS_VISIBLE, 150, 46, 455, 46, content, nullptr, hInstance, nullptr);

    // --- Tabs, each with a page window over the tab body ---
    // WS_CLIPSIBLINGS, and the pages put above it below: the tab control
    // never paints over the page that lies over its body.
    state.hwndTab = CreateWindowExA(0, WC_TABCONTROLA, "", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS, 10, 100, kTabWidth, kTabHeight,
                                    content, nullptr, hInstance, nullptr);
    TCITEMA tie{};
    tie.mask = TCIF_TEXT;
    tie.pszText = (LPSTR) "Coordinator";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabCoordinator, (LPARAM) &tie);
    tie.pszText = (LPSTR) "Local Search";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabLocalSearch, (LPARAM) &tie);
    tie.pszText = (LPSTR) "Local Dictionary";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabDictionary, (LPARAM) &tie);
    tie.pszText = (LPSTR) "Meet in the Middle";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabMeetInTheMiddle, (LPARAM) &tie);
    state.hwndPageCoordinator =
        CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, kPageWidth, kPageHeight, content, nullptr, hInstance, nullptr);
    state.hwndPageSearch =
        CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, kPageWidth, kPageHeight, content, nullptr, hInstance, nullptr);
    state.hwndPageDictionary =
        CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, kPageWidth, kPageHeight, content, nullptr, hInstance, nullptr);
    state.hwndPageMeetInTheMiddle =
        CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, kPageWidth, kPageHeight, content, nullptr, hInstance, nullptr);
    for (HWND page : {state.hwndPageCoordinator, state.hwndPageSearch, state.hwndPageDictionary, state.hwndPageMeetInTheMiddle})
        SetWindowPos(page, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

    // One tooltip window serves every help "?" mark. Long delay before it
    // disappears (the texts are a few lines) and a max width so they wrap.
    state.hwndTooltip = CreateWindowExA(WS_EX_TOPMOST, TOOLTIPS_CLASSA, nullptr, WS_POPUP | TTS_NOPREFIX | TTS_ALWAYSTIP, CW_USEDEFAULT,
                                         CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, hwndDialog, nullptr, hInstance, nullptr);
    SendMessage(state.hwndTooltip, TTM_SETMAXTIPWIDTH, 0, scaled(340));
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
    auto makeEdit = [&](HWND page, const std::string& initial, int x, int y, int w, int id = 0) {
        return CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", initial.c_str(), kEditStyle, x, y, w, 22, page, (HMENU) (INT_PTR) id, hInstance,
                               nullptr);
    };
    auto makeRadio = [&](HWND page, const char* text, int x, int y, int w, int id, bool startsGroup) {
        DWORD style = WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON | (startsGroup ? WS_GROUP : 0);
        return CreateWindowExA(0, "BUTTON", text, style, x, y, w, 20, page, (HMENU) (INT_PTR) id, hInstance, nullptr);
    };
    auto makeCheckbox = [&](HWND page, const char* text, int x, int y, int w, int id, bool checked) {
        HWND h = CreateWindowExA(0, "BUTTON", text, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, x, y, w, 20, page,
                                  (HMENU) (INT_PTR) id, hInstance, nullptr);
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
            "Computer name. Please make unique if you are running the client on several machines.");
    makeLabel(pc, "Server URL:", 8, kCoordRow0 + 2 * kRowStep + 3, 100);
    state.hwndServerUrl = makeEdit(pc, fields.serverUrl, 112, kCoordRow0 + 2 * kRowStep, 330);
    addHelp(pc, 450, kCoordRow0 + 2 * kRowStep + 2,
            "Required. The base address of the coordinator server, for example https://coordinator.example.com (no trailing path).");
    makeLabel(pc, "Poll interval (s):", 8, kCoordRow0 + 3 * kRowStep + 3, 104);
    state.hwndPollInterval = makeEdit(pc, fields.pollIntervalSecs, 112, kCoordRow0 + 3 * kRowStep, 80);
    addHelp(pc, 200, kCoordRow0 + 3 * kRowStep + 2,
            "How many seconds to wait before asking the server for work again when none was available. It is also the starting delay "
            "after a failed request, which then backs off. Default: 30.");

    // How much memory this client will give a meet-in-the-middle target's
    // table. The server only hands out such a target if its table fits in
    // this, so None is a plain "none of that work, thanks" - see the Meet in
    // the Middle tab for what the table is for.
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof memory;
    if (!GlobalMemoryStatusEx(&memory)) {
        memory.ullTotalPhys = 4ull << 30; // modest, if Windows won't say
        memory.ullAvailPhys = 1ull << 30;
    }
    state.tableMemoryStops = tableMemoryStopsFor(memory.ullTotalPhys);
    state.availablePhysicalBytes = memory.ullAvailPhys;
    uint64_t tableMemory = 0;
    if (fields.meetInTheMiddleTableMemory.empty())
        tableMemory = proposedTableMemory(state.tableMemoryStops, memory.ullTotalPhys);
    else if (!parseConfigBytes(fields.meetInTheMiddleTableMemory, tableMemory))
        tableMemory = 0;

    // A group box of its own, because none of it is about joining a server
    // as such: it only matters if that server has meet-in-the-middle work,
    // and the memory is the one thing a client has to agree to spare before
    // it is handed any.
    constexpr int kMemoryY = 222, kMemoryX = 20;
    CreateWindowExA(0, "BUTTON", "Meet-in-the-middle work", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 8, kMemoryY, 570, 186, pc, nullptr,
                     hInstance, nullptr);
    makeLabel(pc,
              "Only relevant if the coordinator has meet-in-the-middle targets to hand out - a file whose directory is unknown, "
              "searched from both ends at once (see the Meet in the Middle tab). That needs a table in memory, and the server hands "
              "this client such a target only if the table fits in what you give here. Everything else about coordinator mode works "
              "the same either way.",
              kMemoryX, kMemoryY + 20, 546, 62);
    const std::string machineMemory = "This machine has " + memoryText(memory.ullTotalPhys) + " of memory, " +
                                      memoryText(memory.ullAvailPhys) + " of it free right now.";
    makeLabel(pc, machineMemory.c_str(), kMemoryX, kMemoryY + 86, 546);
    makeLabel(pc, "Give at most:", kMemoryX, kMemoryY + 115, 100);
    state.hwndTableMemory =
        CreateWindowExA(0, TRACKBAR_CLASSA, "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | TBS_AUTOTICKS | TBS_HORZ, kMemoryX + 104,
                         kMemoryY + 109, 320, 32, pc, (HMENU) (INT_PTR) kIdSetupTableMemory, hInstance, nullptr);
    SendMessage(state.hwndTableMemory, TBM_SETRANGE, TRUE, MAKELONG(0, (int) state.tableMemoryStops.size() - 1));
    SendMessage(state.hwndTableMemory, TBM_SETPAGESIZE, 0, 1);
    SendMessage(state.hwndTableMemory, TBM_SETPOS, TRUE, stopIndexFor(state.tableMemoryStops, tableMemory));
    addHelp(pc, kMemoryX + 432, kMemoryY + 115,
            "How much memory this client will give the table of a meet-in-the-middle target. The server hands out such a target only "
            "if its table fits in this, so None means no work of that kind at all - and no other kind is affected by it.\r\n"
            "The table is only held while such a range is being searched, and on a GPU it is held in the card's own memory too - so "
            "keep this under what your GPU has free if you search with one.");
    state.hwndTableMemoryText = makeLabel(pc, "", kMemoryX, kMemoryY + 145, 546, 30);
    syncTableMemoryText(&state);

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

    // What a match of both hashes is checked against before it's taken for
    // the file (see MatchCheck, engine/match_check.h) - the same on Local
    // Search and Meet in the Middle, so the same words. (The dictionary's key
    // is its own: only basenames with it are compared to the hashes at all.)
    const char* keyHelp =
        "Optional. The file's raw encryption key, in hex - the hash of its basename (what follows the last \\). A name that matches "
        "both hashes but whose basename doesn't have this key isn't the file: it's a false match, written to false_matches.txt, and "
        "the search carries on.";
    // Everything below a tab's Verification toggle is hidden with it.
    auto verification = [&](HWND h) {
        state.verificationControls.push_back(h);
        return h;
    };
    // The heading that shows and hides them (see showVerification): the
    // chevron and the text, SS_NOTIFY so that a click on either reaches the
    // dialog, then an etched line to the edge of the page, so that what it
    // opens reads as a section of its own.
    auto makeVerificationToggle = [&](HWND page, int y) {
        constexpr DWORD kStyle = WS_CHILD | WS_VISIBLE | SS_NOTIFY;
        state.verificationChevrons.push_back(CreateWindowExA(0, "STATIC", "", kStyle, kL1, y + 3, 14, 16, page,
                                                             (HMENU) (INT_PTR) kIdSetupVerification, hInstance, nullptr));
        CreateWindowExA(0, "STATIC", "Verification (Optional)", kStyle, kL1 + 16, y + 3, 124, 18, page,
                         (HMENU) (INT_PTR) kIdSetupVerification, hInstance, nullptr);
        CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_ETCHEDHORZ, kL1 + 146, y + 11, kH2 + 18 - kL1 - 146, 2, page,
                         nullptr, hInstance, nullptr);
    };
    // The hash table row: its size in the left column, the start slots -
    // two numbers, as Words on the dictionary tab - in the right.
    auto makeHashTableRow = [&](HWND page, int y, const HashTableFields& values, HashTableControls& controls) {
        verification(makeLabel(page, "Hash table size:", kL1, y + 3, 104));
        controls.tableSize = verification(makeEdit(page, values.tableSize, kE1, y, 150));
        verification(addHelp(page, kH1, y + 2,
                             "Optional, with Start slots: how many slots the hash table of the file's archive has - a power of "
                             "two. A name that matches both hashes but that a lookup in that table wouldn't find the file's entry "
                             "by isn't the file: it's a false match, written to false_matches.txt, and the search carries on. Two "
                             "hashes match by chance about once in 2^64 names, which a long search does get to."));
        verification(makeLabel(page, "Start slots:", kL2, y + 3, 110));
        controls.firstStartSlot = verification(makeEdit(page, values.firstStartSlot, kE2, y, 54));
        verification(makeLabel(page, "to", kE2 + 58, y + 3, 12));
        controls.lastStartSlot = verification(makeEdit(page, values.lastStartSlot, kE2 + 72, y, 54));
        verification(addHelp(page, kH2, y + 2,
                             "Optional, with Hash table size: the slots a lookup can start at and still reach the file's entry - "
                             "from the one just after the nearest never-used slot before the entry, to the entry's own slot. The "
                             "first is larger than the second if they wrap past the end of the table."));
    };

    makeLabel(ps, "Search type:", kL1, row(0) + 3, 100);
    state.hwndBoundedRadio = makeRadio(ps, "Bounded", kE1, row(0) + 1, 90, kIdSetupBoundedRadio, true);
    state.hwndContinuousRadio = makeRadio(ps, "Continuous", kE1 + 100, row(0) + 1, 100, kIdSetupContinuousRadio, false);
    SendMessage(fields.initialContinuous ? state.hwndContinuousRadio : state.hwndBoundedRadio, BM_SETCHECK, BST_CHECKED, 0);
    addHelp(ps, kE1 + 208, row(0) + 2,
            "Bounded searches only candidates of exactly the start candidate's length, between the lower and upper bound, then stops.\r\n"
            "Continuous does the same, then moves on to longer candidates and keeps going until you quit.");

    makeLabel(ps, "Alphabet:", kL1, row(1) + 3, 100);
    state.hwndAlphabet = makeEdit(ps, fields.alphabet, kE1, row(1), 150);
    const std::string alphabetHelp = "Every character a candidate may contain, typed out with no separators - at most " +
                                     std::to_string(MAX_ALPHABET_SIZE) + " of them.";
    addHelp(ps, kH1, row(1) + 2, alphabetHelp.c_str());
    makeLabel(ps, "Start candidate:", kL2, row(1) + 3, 110);
    state.hwndStartCandidate = makeEdit(ps, fields.startCandidate, kE2, row(1), 120);
    addHelp(ps, kH2, row(1) + 2,
            "Optional. The full filename (prefix + candidate + suffix) to begin searching from, to resume a long search where it left "
            "off. Leave blank to start from the beginning.");

    makeLabel(ps, "Prefix:", kL1, row(2) + 3, 100);
    state.hwndPrefix = makeEdit(ps, fields.prefix, kE1, row(2), 150);
    addHelp(ps, kH1, row(2) + 2, "The fixed text every filename starts with, before the unknown part being searched for.");
    makeLabel(ps, "Suffix:", kL2, row(2) + 3, 110);
    state.hwndSuffix = makeEdit(ps, fields.suffix, kE2, row(2), 120);
    addHelp(ps, kH2, row(2) + 2, "The fixed text every filename ends with, after the unknown part (for example a file extension).");

    makeLabel(ps, "Lower bound:", kL1, row(3) + 3, 100);
    state.hwndLowerBound = makeEdit(ps, fields.lowerBound, kE1, row(3), 150);
    addHelp(ps, kH1, row(3) + 2, "The full filename (inclusive) the search range starts at.");
    makeLabel(ps, "Upper bound:", kL2, row(3) + 3, 110);
    state.hwndUpperBound = makeEdit(ps, fields.upperBound, kE2, row(3), 120);
    addHelp(ps, kH2, row(3) + 2, "The full filename (inclusive) the search range ends at.");

    makeLabel(ps, "Hash A (hex):", kL1, row(4) + 3, 100);
    state.hwndHashA = makeEdit(ps, fields.hashA, kE1, row(4), 150);
    addHelp(ps, kH1, row(4) + 2, "The first of the two 32-bit MPQ hashes of the filename you're looking for, in hex (a 0x prefix is optional).");
    makeLabel(ps, "Hash B (hex):", kL2, row(4) + 3, 110);
    state.hwndHashB = makeEdit(ps, fields.hashB, kE2, row(4), 120);
    addHelp(ps, kH2, row(4) + 2, "The second of the two 32-bit MPQ hashes, in hex (a 0x prefix is optional).");

    makeLabel(ps, "Insert from start:", kL1, row(5) + 3, 104);
    state.hwndInsertFromStart = makeEdit(ps, fields.insertFromStart, kE1, row(5), 150);
    addHelp(ps, kH1, row(5) + 2,
            "Optional. Text inserted into every candidate, and the position it goes at, counted from the start - for example \\, 3 "
            "puts a backslash after the candidate's first three characters. A candidate shorter than the position gets nothing "
            "inserted. The bounds and the start candidate are without it; the matches have it.");
    makeLabel(ps, "Insert from end:", kL2, row(5) + 3, 110);
    state.hwndInsertFromEnd = makeEdit(ps, fields.insertFromEnd, kE2, row(5), 120);
    addHelp(ps, kH2, row(5) + 2,
            "Optional. Like Insert from start, counted from the end - for example \\, 4 puts a backslash before the candidate's last "
            "four characters.");

    makeLabel(ps, "Min backslash:", kL1, row(6) + 3, 100);
    state.hwndMinBackslash = makeEdit(ps, fields.minBackslashCount, kE1, row(6), 60);
    addHelp(ps, kE1 + 68, row(6) + 2,
            "Optional. The fewest backslashes a candidate may contain without being skipped. 0 (or blank) means none are needed. "
            "Backslashes in the prefix don't count.");
    makeLabel(ps, "Max backslash:", kL2, row(6) + 3, 110);
    state.hwndMaxBackslash = makeEdit(ps, fields.maxBackslashCount, kE2, row(6), 60);
    addHelp(ps, kE2 + 68, row(6) + 2,
            "The most backslashes a candidate may contain before it is skipped. 0 means no limit (to forbid backslashes entirely, leave "
            "them out of the alphabet instead).");

    // Checkboxes go beneath all the input fields, one per column.
    state.hwndPrune = makeCheckbox(ps, "Prune symbol runs", kL1, row(7) + 1, 150, kIdSetupPruneCheckbox, fields.pruneSymbolRuns);
    addHelp(ps, kH1, row(7) + 2,
            "Skip candidates containing three or more symbols in a row that are neither letters, digits nor spaces - real filenames "
            "practically never have those. Makes the search faster.");
    state.hwndPruneBrackets =
        makeCheckbox(ps, "Prune unopened brackets", kL2, row(7) + 1, 170, kIdSetupPruneBracketsCheckbox, fields.pruneUnopenedBrackets);
    addHelp(ps, kH2, row(7) + 2,
            "Skip candidates that close a bracket that was never opened - a ) with no ( before it, or a ] with no [ before it "
            "(brackets opened in the prefix count). Real filenames practically never have those. Makes the search faster.");
    state.hwndPruneWhole =
        makeCheckbox(ps, "Prune the whole candidate", kL1, row(8) + 1, 190, kIdSetupPruneWholeCheckbox, fields.pruneWholeCandidate);
    addHelp(ps, kH1, row(8) + 2,
            "Apply the pruning rules, and Max and Min backslash, to every character of a candidate but the last - not only to the "
            "characters before the last five or so, which the CPU goes through. About a fifth fewer candidates are searched, and "
            "the search is about 10% faster on a GPU.");
    state.hwndPruneAdjacent = makeCheckbox(ps, "Prune adjacent backslashes", kL2, row(8) + 1, 190, kIdSetupPruneAdjacentCheckbox,
                                           fields.pruneAdjacentBackslashes);
    addHelp(ps, kH2, row(8) + 2,
            "Skip candidates with two backslashes next to each other - counting one the prefix ends with. Real filenames never have "
            "those. Makes the search faster.");

    makeVerificationToggle(ps, row(9));
    verification(makeLabel(ps, "Encryption key:", kL1, row(10) + 3, 104));
    state.hwndKey = verification(makeEdit(ps, fields.encryptionKey, kE1, row(10), 150));
    verification(addHelp(ps, kH1, row(10) + 2, keyHelp));
    makeHashTableRow(ps, row(11), fields.hashTable, state.searchHashTable);

    // --- Dictionary page (two columns, as Local Search) ---
    HWND pd = state.hwndPageDictionary;
    const DictionaryFields& dict = fields.dictionary;
    makeLabel(pd, "Search on your own for a filename made of words, such as DOOR_KNOB.TXT. The words come from the built-in English "
                  "dictionary and your own word lists. If you know the file's encryption key, every basename that matches it is "
                  "recorded as well. The search stops once every combination has been tried.",
              8, 6, 570, 62);

    makeLabel(pd, "Word lists:", kL1, row(0) + 3, 100);
    state.hwndDictLists = makeEdit(pd, dict.dictionaries, kE1, row(0), 340);
    CreateWindowExA(0, "BUTTON", "Browse...", WS_CHILD | WS_VISIBLE, kE1 + 348, row(0) - 1, 80, 24, pd,
                     (HMENU) (INT_PTR) kIdSetupBrowseWordLists, hInstance, nullptr);
    addHelp(pd, kH2, row(0) + 2,
            "Optional. Files of words to search, one word per line, separated by commas (a path with a comma in it goes in \"quotes\"). "
            "Lines in the word lists that start with # are skipped. Their words are searched together with the built-in dictionary's, if that's ticked.");

    state.hwndDictBuiltin =
        makeCheckbox(pd, "Built-in English dictionary", kL1, row(1) + 1, 190, 0, dict.builtinDictionary);
    addHelp(pd, kH1, row(1) + 2,
            "Search the 63,875 English words compiled into namebreak (english-1), as well as any word lists. Untick to search only "
            "the word lists.");
    makeLabel(pd, "Words:", kL2, row(1) + 3, 110);
    state.hwndDictMinWords = makeEdit(pd, dict.minWords, kE2, row(1), 40);
    makeLabel(pd, "to", kE2 + 48, row(1) + 3, 16);
    state.hwndDictMaxWords = makeEdit(pd, dict.maxWords, kE2 + 68, row(1), 40);
    addHelp(pd, kH2, row(1) + 2,
            "How many words a candidate has: every count from the first to the second (at most 8), fewest first. The second is "
            "required. Each word more multiplies the candidates by the number of words.");

    makeLabel(pd, "Separators:", kL1, row(2) + 3, 100);
    state.hwndDictSeparators = makeEdit(pd, dict.separators, kE1, row(2), 150);
    addHelp(pd, kH1, row(2) + 2,
            "What goes between two words, separated by commas, each in \"quotes\" - for example \"\", \"_\", \"-\", \" \". \"\" writes the "
            "words together, and \"\\\" makes the words before it directories. Blank: \"\" only.");
    makeLabel(pd, "Tails:", kL2, row(2) + 3, 110);
    state.hwndDictTails = makeEdit(pd, dict.tails, kE2, row(2), 120);
    addHelp(pd, kH2, row(2) + 2,
            "Optional. What follows the last word, separated by commas: digits:1-2 is 0-9 and 00-99, letters:0-1 is nothing or one "
            "of A-Z, and _| is an underscore or nothing. Every combination of one of each, in turn, follows every candidate. "
            "Blank: nothing.");

    makeLabel(pd, "Prefix:", kL1, row(3) + 3, 100);
    state.hwndDictPrefix = makeEdit(pd, dict.prefix, kE1, row(3), 150);
    addHelp(pd, kH1, row(3) + 2,
            "The fixed text every filename starts with, before the words - may be blank, for a file whose directory isn't known.");
    makeLabel(pd, "Suffix:", kL2, row(3) + 3, 110);
    state.hwndDictSuffix = makeEdit(pd, dict.suffix, kE2, row(3), 120);
    addHelp(pd, kH2, row(3) + 2, "The fixed text every filename ends with, after the words (for example a file extension).");

    makeLabel(pd, "Lower bound:", kL1, row(4) + 3, 100);
    state.hwndDictLowerBound = makeEdit(pd, dict.lowerBound, kE1, row(4), 150);
    addHelp(pd, kH1, row(4) + 2,
            "Optional. A whole filename (inclusive): only filenames that sort at or after it are searched - typically the unknown "
            "file's neighbour in the archive.");
    makeLabel(pd, "Upper bound:", kL2, row(4) + 3, 110);
    state.hwndDictUpperBound = makeEdit(pd, dict.upperBound, kE2, row(4), 120);
    addHelp(pd, kH2, row(4) + 2, "Optional. A whole filename (inclusive): only filenames that sort at or before it are searched.");

    makeLabel(pd, "Hash A (hex):", kL1, row(5) + 3, 100);
    state.hwndDictHashA = makeEdit(pd, dict.hashA, kE1, row(5), 150);
    addHelp(pd, kH1, row(5) + 2, "The first of the two 32-bit MPQ hashes of the filename you're looking for, in hex (a 0x prefix is optional).");
    makeLabel(pd, "Hash B (hex):", kL2, row(5) + 3, 110);
    state.hwndDictHashB = makeEdit(pd, dict.hashB, kE2, row(5), 120);
    addHelp(pd, kH2, row(5) + 2, "The second of the two 32-bit MPQ hashes, in hex (a 0x prefix is optional).");

    makeLabel(pd, "Encryption key:", kL1, row(6) + 3, 104);
    state.hwndDictKey = makeEdit(pd, dict.encryptionKey, kE1, row(6), 150, kIdSetupEncryptionKey);
    addHelp(pd, kH1, row(6) + 2,
            "Optional. The file's raw encryption key, in hex - the hash of its basename (what follows the last \\). Then only "
            "candidates with that basename are compared to the hashes, which makes the search about a third faster.");
    makeLabel(pd, "Matches name:", kL2, row(6) + 3, 110);
    state.hwndDictMatchesName = makeEdit(pd, dict.matchesName, kE2, row(6), 120);
    addHelp(pd, kH2, row(6) + 2,
            "Optional. Names the files the search writes - matches-<name>.txt, basenames-<name>.txt and wordnumber-<name>.txt - so "
            "searches for different files keep theirs apart. Blank: matches.txt and so on.");

    state.hwndDictRecordBasenames = makeCheckbox(pd, "Record basenames", kL1, row(7) + 1, 150, 0, dict.recordBasenames);
    addHelp(pd, kH1, row(7) + 2,
            "With an encryption key: write every basename that matches it to basenames.txt - the file's basename is among them, "
            "even when its directory is wrong.");
    state.hwndDictRecordHashA = makeCheckbox(pd, "Compare every candidate", kL2, row(7) + 1, 190, 0, dict.recordHashAMatches);
    addHelp(pd, kH2, row(7) + 2,
            "With an encryption key: compare every candidate to the hashes all the same, as if the key weren't known, recording every "
            "Hash A match - but the search is about a third slower.");

    state.hwndDictResume = makeCheckbox(pd, "Resume from last candidate", kL1, row(8) + 1, 190, 0, dict.resume);
    addHelp(pd, kH1, row(8) + 2,
            "Carry on where the last search with these settings stopped - its progress is saved in wordnumber.txt every 30 seconds "
            "and when you quit - instead of starting from the beginning.");

    makeVerificationToggle(pd, row(9));
    makeHashTableRow(pd, row(10), dict.hashTable, state.dictHashTable);
    syncRecordBasenames(&state);

    // --- Meet in the Middle page (two columns, as Local Search) ---
    HWND pm = state.hwndPageMeetInTheMiddle;
    const MeetInTheMiddleFields& mitm = fields.meetInTheMiddle;
    makeLabel(pm,
              "Search on your own for a file whose directory is unknown, but where the hash state right after that directory - the "
              "meeting point - is already known, for both hashes. (Two files of the same unknown directory whose basenames are known "
              "pin it down.) The search then comes at the directory from both ends at once: forward from the prefix into a table of "
              "every beginning, backward from the meeting point, meeting in the middle - so every character the table holds is one "
              "character less to brute-force.",
              8, 6, 570, 78);
    makeLabel(pm,
              "Mind the memory: that table can be large. With a 43-character alphabet it takes about 30 MB for 4 of the directory's "
              "characters, 1.4 GB for 5 and 59 GB for 6 - Table memory below is the cap, and on a GPU the table has to fit in the "
              "card's own memory as well.",
              8, 88, 570, 46);
    // Ten rows of fields and the Verification toggle, so this page's rows sit
    // a little closer together than the other pages' (kRowStep).
    constexpr int kMitmRow0 = 140, kMitmRowStep = 28;
    auto rowM = [&](int n) { return kMitmRow0 + n * kMitmRowStep; };
    // Every field in a column is the same width, and every "?" sits in that
    // column's own help position - so the whole page reads as two columns,
    // however little a field has to hold (a length is three characters, an
    // alphabet is forty).
    constexpr int kMitmEdit1 = 150, kMitmEdit2 = 120;

    makeLabel(pm, "Alphabet:", kL1, rowM(0) + 3, 100);
    state.hwndMitmAlphabet = makeEdit(pm, mitm.alphabet, kE1, rowM(0), kMitmEdit1);
    const std::string mitmAlphabetHelp = "Every character the unknown directory may be made of, typed out with no separators - at most " +
                                         std::to_string(MAX_ALPHABET_SIZE) + " of them. A space is one of them if you type one, and "
                                         "spaces at either end are kept, so type it exactly. A backslash searches directories with "
                                         "subdirectories in them.";
    addHelp(pm, kH1, rowM(0) + 2, mitmAlphabetHelp.c_str());
    makeLabel(pm, "Matches name:", kL2, rowM(0) + 3, 110);
    state.hwndMitmMatchesName = makeEdit(pm, mitm.matchesName, kE2, rowM(0), kMitmEdit2);
    addHelp(pm, kH2, rowM(0) + 2,
            "Optional. Names the files the search writes - matches-<name>.txt and meet-in-the-middle-<name>.txt (where its progress "
            "is kept) - so searches for different files keep theirs apart. Blank: matches.txt and meet-in-the-middle.txt.");

    makeLabel(pm, "Prefix:", kL1, rowM(1) + 3, 100);
    state.hwndMitmPrefix = makeEdit(pm, mitm.prefix, kE1, rowM(1), kMitmEdit1);
    addHelp(pm, kH1, rowM(1) + 2,
            "Optional. Known text before the unknown directory. The meeting point must be the hash state after the prefix and the "
            "directory together.");
    makeLabel(pm, "Suffix:", kL2, rowM(1) + 3, 110);
    state.hwndMitmSuffix = makeEdit(pm, mitm.suffix, kE2, rowM(1), kMitmEdit2);
    addHelp(pm, kH2, rowM(1) + 2,
            "What follows the directory - for a file inside it, the backslash and the basename, for example \\BWUNIN.PIF.");

    // A seed in each column, rather than both of them in the left one -
    // seed1 above seed1, seed2 above seed2.
    makeLabel(pm, "Hash A, seed 1:", kL1, rowM(2) + 3, 104);
    state.hwndMitmASeed1 = makeEdit(pm, mitm.meetingPointASeed1, kE1, rowM(2), kMitmEdit1);
    addHelp(pm, kH1, rowM(2) + 2,
            "The meeting point: hashA's first seed (seed1) right after the directory, in hex (a 0x prefix is optional). These four "
            "seeds together are what the search looks for - a directory that takes the hash from the prefix's state to them. Not the "
            "file's own hash, which is Hash A (hex) below.");
    makeLabel(pm, "Hash A, seed 2:", kL2, rowM(2) + 3, 110);
    state.hwndMitmASeed2 = makeEdit(pm, mitm.meetingPointASeed2, kE2, rowM(2), kMitmEdit2);
    addHelp(pm, kH2, rowM(2) + 2, "The meeting point: hashA's second seed (seed2) right after the directory, in hex.");

    makeLabel(pm, "Hash B, seed 1:", kL1, rowM(3) + 3, 104);
    state.hwndMitmBSeed1 = makeEdit(pm, mitm.meetingPointBSeed1, kE1, rowM(3), kMitmEdit1);
    addHelp(pm, kH1, rowM(3) + 2,
            "Optional, but give both of hashB's seeds or neither. The search itself only walks hashA's state - these two are what "
            "it checks that meeting point against before it starts, which is what catches a meeting point taken for the wrong place "
            "in the name. Without them the search runs on hashA alone, and such a mistake looks just like a file that isn't there.");
    makeLabel(pm, "Hash B, seed 2:", kL2, rowM(3) + 3, 110);
    state.hwndMitmBSeed2 = makeEdit(pm, mitm.meetingPointBSeed2, kE2, rowM(3), kMitmEdit2);
    addHelp(pm, kH2, rowM(3) + 2, "Optional, with Hash B, seed 1: the meeting point's second seed for hashB, in hex.");

    makeLabel(pm, "Hash A (hex):", kL1, rowM(4) + 3, 100);
    state.hwndMitmHashA = makeEdit(pm, mitm.hashA, kE1, rowM(4), kMitmEdit1);
    addHelp(pm, kH1, rowM(4) + 2,
            "The file's own first MPQ hash, in hex. Hashing the suffix onto the meeting point must give it, or the search refuses to "
            "start - a meeting point for the wrong place in the name would otherwise search everything and find nothing.");
    makeLabel(pm, "Hash B (hex):", kL2, rowM(4) + 3, 110);
    state.hwndMitmHashB = makeEdit(pm, mitm.hashB, kE2, rowM(4), kMitmEdit2);
    addHelp(pm, kH2, rowM(4) + 2, "The file's own second MPQ hash, in hex - checked against the meeting point the same way.");

    makeLabel(pm, "Min length:", kL1, rowM(5) + 3, 100);
    state.hwndMitmMinLength = makeEdit(pm, mitm.minLength, kE1, rowM(5), kMitmEdit1);
    addHelp(pm, kH1, rowM(5) + 2,
            "The shortest the unknown directory may be, in characters - the directory alone, not the whole filename: the prefix and "
            "the suffix are known and aren't counted. Default: 1.");
    makeLabel(pm, "Max length:", kL2, rowM(5) + 3, 110);
    state.hwndMitmMaxLength = makeEdit(pm, mitm.maxLength, kE2, rowM(5), kMitmEdit2);
    const std::string mitmMaxLengthHelp = "Required. The longest the unknown directory may be, in characters - at most " +
                                          std::to_string(MAX_CANDIDATE_LEN) +
                                          ". Every length from Min length up to it is searched, shortest first, so this is as long a "
                                          "directory as the search will consider; one character more costs about as many times more "
                                          "work as the alphabet has characters.";
    addHelp(pm, kH2, rowM(5) + 2, mitmMaxLengthHelp.c_str());

    makeLabel(pm, "Table length:", kL1, rowM(6) + 3, 100);
    state.hwndMitmTableLength = makeEdit(pm, mitm.tableLength, kE1, rowM(6), kMitmEdit1);
    addHelp(pm, kH1, rowM(6) + 2,
            "Optional. How many of the directory's first characters go in the table, which holds the hash state of every beginning of "
            "that length. Only the characters after them are brute-forced - walked backwards from the meeting point - so this is how "
            "many characters the memory buys you, and each one of them makes the search about as many times shorter as the alphabet "
            "has characters.\r\n"
            "Blank: as many as Table memory allows, which is what you usually want.");
    makeLabel(pm, "Table memory:", kL2, rowM(6) + 3, 110);
    state.hwndMitmTableMemory = makeEdit(pm, mitm.tableMemory, kE2, rowM(6), kMitmEdit2);
    addHelp(pm, kH2, rowM(6) + 2,
            "The most memory that table may take - a plain number of bytes, or one with a K, M or G after it (1G by default). It is "
            "what decides Table length when that's left blank: a character more needs about as many times more memory as the alphabet "
            "has characters, so with a 43-character one it's about 30 MB for 4 characters, 1.4 GB for 5 and 59 GB for 6. On a GPU the "
            "table has to fit in the card's own memory too.");

    makeLabel(pm, "Min backslash:", kL1, rowM(7) + 3, 100);
    state.hwndMitmMinBackslash = makeEdit(pm, mitm.minBackslashCount, kE1, rowM(7), kMitmEdit1);
    addHelp(pm, kH1, rowM(7) + 2,
            "Optional. The fewest backslashes the directory may contain without being skipped. 0 (or blank) means none are needed.");
    makeLabel(pm, "Max backslash:", kL2, rowM(7) + 3, 110);
    state.hwndMitmMaxBackslash = makeEdit(pm, mitm.maxBackslashCount, kE2, rowM(7), kMitmEdit2);
    addHelp(pm, kH2, rowM(7) + 2,
            "The most backslashes the directory may contain before it is skipped. 0 means no limit (to forbid them entirely, leave "
            "the backslash out of the alphabet instead).");

    state.hwndMitmPrune = makeCheckbox(pm, "Prune symbol runs", kL1, rowM(8) + 1, 150, 0, mitm.pruneSymbolRuns);
    addHelp(pm, kH1, rowM(8) + 2,
            "Skip directories containing three or more symbols in a row that are neither letters, digits nor spaces - real names "
            "practically never have those. Applies to every character of the directory, and makes the search faster.");
    state.hwndMitmPruneBrackets = makeCheckbox(pm, "Prune unopened brackets", kL2, rowM(8) + 1, 170, 0, mitm.pruneUnopenedBrackets);
    addHelp(pm, kH2, rowM(8) + 2,
            "Skip directories that close a bracket that was never opened - a ) with no ( before it, or a ] with no [ (brackets "
            "opened in the prefix count).");
    state.hwndMitmPruneAdjacent = makeCheckbox(pm, "Prune adjacent backslashes", kL1, rowM(9) + 1, 190, 0, mitm.pruneAdjacentBackslashes);
    addHelp(pm, kH1, rowM(9) + 2,
            "Skip directories with two backslashes next to each other - counting one the prefix ends with, and one the suffix begins "
            "with. Real names never have those.");
    state.hwndMitmResume = makeCheckbox(pm, "Resume from last candidate", kL2, rowM(9) + 1, 190, 0, mitm.resume);
    addHelp(pm, kH2, rowM(9) + 2,
            "Carry on where the last search with these settings stopped - its progress is saved in meet-in-the-middle.txt every 30 "
            "seconds and when you quit - instead of starting from the beginning.");

    makeVerificationToggle(pm, rowM(10));
    verification(makeLabel(pm, "Encryption key:", kL1, rowM(11) + 3, 104));
    state.hwndMitmKey = verification(makeEdit(pm, mitm.encryptionKey, kE1, rowM(11), kMitmEdit1));
    verification(addHelp(pm, kH1, rowM(11) + 2, keyHelp));
    makeHashTableRow(pm, rowM(12), mitm.hashTable, state.mitmHashTable);

    // --- Buttons, under the scroll view (placed by layoutDialog) ---
    state.hwndOptimize = CreateWindowExA(0, "BUTTON", "Optimize for hardware", WS_CHILD | WS_VISIBLE, 12, 0, 150, 26, hwndDialog,
                                         (HMENU) (INT_PTR) kIdSetupOptimize, hInstance, nullptr);
    state.hwndOk = CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 430, 0, 80, 26, hwndDialog,
                                   (HMENU) (INT_PTR) kIdSetupOk, hInstance, nullptr);
    state.hwndCancel = CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE, 520, 0, 80, 26, hwndDialog,
                                       (HMENU) (INT_PTR) kIdSetupCancel, hInstance, nullptr);

    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    HFONT titleFont = CreateFontA(-scaled(24), 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                   CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    EnumChildWindows(hwndDialog, setFontEnumProc, (LPARAM) font);
    SendMessage(title, WM_SETFONT, (WPARAM) titleFont, TRUE);
    HFONT chevronFont = CreateFontA(-scaled(12), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, SYMBOL_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                     CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Marlett");
    for (HWND h : state.verificationChevrons)
        SendMessage(h, WM_SETFONT, (WPARAM) chevronFont, TRUE);

    // Everything above is placed in 96-DPI pixels; to the screen's now.
    // The logo's image only after: a bitmap static takes the bitmap's own
    // size, which is then already the scaled one.
    gui::scaleChildWindows(hwndDialog);
    HBITMAP logo = makeLogoBitmap(scaled(120), scaled(80));
    SendMessage(logoCtl, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM) logo);

    SendMessage(state.hwndTab, TCM_SETCURSEL, fields.initialTab, 0);
    showTabPage(&state, fields.initialTab);
    // Shown from the start if config.conf already says any of it, so that
    // nothing that applies is out of sight.
    auto anyHashTable = [](const HashTableFields& h) {
        return !h.tableSize.empty() || !h.firstStartSlot.empty() || !h.lastStartSlot.empty();
    };
    layoutDialog(&state);
    showVerification(&state, !fields.encryptionKey.empty() || anyHashTable(fields.hashTable) || anyHashTable(fields.dictionary.hashTable) ||
                                 !fields.meetInTheMiddle.encryptionKey.empty() || anyHashTable(fields.meetInTheMiddle.hashTable));

    ShowWindow(hwndDialog, SW_SHOW);
    UpdateWindow(hwndDialog);
    SetForegroundWindow(hwndDialog);

    // Shown from the tray menu's Settings..., this loop is pumping the whole
    // thread's messages, the main window's among them - so it can be handed
    // the WM_QUIT of a search that stopped while the dialog was up (see
    // WM_APP_WORKER_STOPPED, main_window.cpp). Swallowing it here would
    // leave the process with no window and nothing to wake it; the outer
    // loop is the one waiting for it, so put it back and leave.
    MSG msg;
    while (!state.done) {
        const BOOL got = GetMessage(&msg, nullptr, 0, 0);
        if (got <= 0) {
            if (got == 0)
                PostQuitMessage((int) msg.wParam);
            break;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    // The window is already gone if OK/Cancel/close ended the loop; it isn't
    // if a WM_QUIT did, and it must not outlive `state`, which it holds a
    // pointer to.
    if (IsWindow(hwndDialog)) {
        SetWindowLongPtr(hwndDialog, GWLP_USERDATA, 0);
        DestroyWindow(hwndDialog);
    }

    DeleteObject(logo);
    DeleteObject(titleFont);
    DeleteObject(chevronFont);
    return state.accepted;
}

std::vector<std::pair<std::string, std::string>> searchConfigKeys(const SetupDialogFields& fields) {
    std::vector<std::pair<std::string, std::string>> keys = {
        {"alphabet", fields.alphabet},
        {"max_backslash_count", fields.maxBackslashCount},
        {"min_backslash_count", fields.minBackslashCount},
        {"prefix", fields.prefix},
        {"suffix", fields.suffix},
        {"start_candidate", fields.startCandidate},
        {"lower_bound", fields.lowerBound},
        {"upper_bound", fields.upperBound},
        {"hash_a", fields.hashA},
        {"hash_b", fields.hashB},
        {"prune_symbol_runs", fields.pruneSymbolRuns ? "true" : "false"},
        {"prune_unopened_brackets", fields.pruneUnopenedBrackets ? "true" : "false"},
        {"prune_whole_candidate", fields.pruneWholeCandidate ? "true" : "false"},
        {"prune_adjacent_backslashes", fields.pruneAdjacentBackslashes ? "true" : "false"},
        {"insert_from_start", fields.insertFromStart},
        {"insert_from_end", fields.insertFromEnd},
        {"encryption_key", fields.encryptionKey},
    };
    for (const auto& kv : hashTableValues(fields.hashTable))
        keys.push_back(kv);
    return keys;
}

std::vector<std::pair<std::string, std::string>> dictionaryConfigKeys(const DictionaryFields& fields) {
    auto optional = [](const std::string& text) { return text.empty() ? text : quoteDictionaryValue(text); };
    const bool hasKey = !fields.encryptionKey.empty();
    std::vector<std::pair<std::string, std::string>> keys = {
        {"builtin_dictionary", fields.builtinDictionary ? "english-1" : "none"},
        {"dictionaries", fields.dictionaries},
        {"min_words", optional(fields.minWords)},
        {"max_words", optional(fields.maxWords)},
        {"separators", fields.separators},
        {"tails", fields.tails},
        // Required, but may be empty - written as "".
        {"prefix", quoteDictionaryValue(fields.prefix)},
        {"suffix", quoteDictionaryValue(fields.suffix)},
        // Left out, rather than empty, for no bound.
        {"lower_bound", optional(fields.lowerBound)},
        {"upper_bound", optional(fields.upperBound)},
        {"hash_a", optional(fields.hashA)},
        {"hash_b", optional(fields.hashB)},
        {"encryption_key", optional(fields.encryptionKey)},
        // Only with a key - record_basenames = true is an error without one.
        {"record_basenames", hasKey ? (fields.recordBasenames ? "true" : "false") : ""},
        {"record_hasha_matches", fields.recordHashAMatches ? "true" : "false"},
        {"matches_name", optional(fields.matchesName)},
        {"resume_from_last_candidate", fields.resume ? "true" : "false"},
    };
    for (const auto& kv : hashTableValues(fields.hashTable))
        keys.emplace_back(kv.first, optional(kv.second));
    return keys;
}

std::vector<std::pair<std::string, std::string>> meetInTheMiddleConfigKeys(const MeetInTheMiddleFields& fields) {
    std::vector<std::pair<std::string, std::string>> keys = meetInTheMiddleValues(fields);
    // The three values a space can be part of - and the file keeps a
    // leading or trailing one only inside quotes.
    for (auto& kv : keys)
        if ((kv.first == "alphabet" || kv.first == "prefix" || kv.first == "suffix" || kv.first == "matches_name") && !kv.second.empty())
            kv.second = quoteDictionaryValue(kv.second);
    return keys;
}

} // namespace gui
