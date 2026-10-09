// First-run setup dialog - a hand-rolled modal window (CreateWindowEx + its
// own nested message loop) rather than a .rc dialog template, so the GUI
// needs no resource-compiler build step. Only ever shown once, before the
// main window/worker thread exist, so there's no other message loop running
// concurrently to conflict with.
//
// One SysTabControl32 with three tabs - Coordinator; Local Search (which
// covers both bounded and continuous, since they share the exact same
// [search] config keys and differ only in that one mode value - a radio
// pair inside the tab picks between them); and Local Dictionary, for the
// [dictionary] section. Each tab's controls live on their
// own plain page window laid over the tab control's body (see
// SetupPageWndProc for why), created up front; only the active tab's page is
// shown at a time (see showTabPage). Whichever tab is active when OK is
// pressed is what gets validated/submitted - the user picks a mode by
// picking a tab, not through a separate control. Every option has a blue "?"
// beside it whose tooltip explains it.

#include "gui/win32/setup_dialog.h"

#include <commdlg.h>

#include <cstring>
#include <deque>
#include <vector>

#include "backends/backends.h"
#include "common/config.h"
#include "common/string_util.h"
#include "engine/limits.h"
#include "gui/win32/about_dialog.h"
#include "gui/win32/app_icons.h"

#pragma comment(lib, "comdlg32.lib")

namespace {

using gui::DictionaryFields;
using gui::SetupDialogFields;
using gui::kTabCoordinator;
using gui::kTabDictionary;
using gui::kTabLocalSearch;
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
    HWND hwndPageDictionary = nullptr;
    HWND hwndUsername = nullptr, hwndHostname = nullptr, hwndServerUrl = nullptr, hwndPollInterval = nullptr;
    HWND hwndBoundedRadio = nullptr, hwndContinuousRadio = nullptr, hwndPrune = nullptr, hwndPruneBrackets = nullptr,
         hwndPruneWhole = nullptr, hwndPruneAdjacent = nullptr;
    HWND hwndInsertFromStart = nullptr, hwndInsertFromEnd = nullptr;
    HWND hwndAlphabet = nullptr, hwndMaxBackslash = nullptr, hwndMinBackslash = nullptr, hwndPrefix = nullptr, hwndSuffix = nullptr, hwndStartCandidate = nullptr,
         hwndLowerBound = nullptr, hwndUpperBound = nullptr, hwndHashA = nullptr, hwndHashB = nullptr;
    // The Local Dictionary tab's.
    HWND hwndDictBuiltin = nullptr, hwndDictLists = nullptr, hwndDictMinWords = nullptr, hwndDictMaxWords = nullptr,
         hwndDictSeparators = nullptr, hwndDictTails = nullptr, hwndDictPrefix = nullptr, hwndDictSuffix = nullptr,
         hwndDictLowerBound = nullptr, hwndDictUpperBound = nullptr, hwndDictHashA = nullptr, hwndDictHashB = nullptr,
         hwndDictKey = nullptr, hwndDictMatchesName = nullptr, hwndDictRecordBasenames = nullptr, hwndDictRecordHashA = nullptr,
         hwndDictResume = nullptr;
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
}

bool isChecked(HWND h) {
    return SendMessage(h, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

std::string windowText(HWND h) {
    std::string text(GetWindowTextLengthA(h) + 1, '\0');
    text.resize(GetWindowTextA(h, &text[0], (int) text.size()));
    return text;
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
    if (msg == WM_COMMAND)
        return SendMessage(GetParent(hwnd), msg, wParam, lParam);
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
                case kIdSetupBrowseWordLists:
                    browseWordLists(hwnd, state);
                    return 0;
                case kIdSetupEncryptionKey:
                    if (HIWORD(wParam) == EN_CHANGE)
                        syncRecordBasenames(state);
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
                    } else {
                        std::string alphabet = getField(state->hwndAlphabet);
                        std::string maxBackslash = getField(state->hwndMaxBackslash);
                        std::string minBackslash = getField(state->hwndMinBackslash);
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
                        if (alphabet.empty() || maxBackslash.empty() || prefix.empty() || suffix.empty() || lowerBound.empty() ||
                            upperBound.empty() || hashA.empty() || hashB.empty()) {
                            MessageBoxA(hwnd, "All Local Search fields except Start candidate are required.", "namebreak setup",
                                        MB_OK | MB_ICONWARNING);
                            return 0;
                        }
                        state->fields->alphabet = alphabet;
                        state->fields->maxBackslashCount = maxBackslash;
                        state->fields->minBackslashCount = minBackslash.empty() ? "0" : minBackslash;
                        state->fields->prefix = prefix;
                        state->fields->suffix = suffix;
                        state->fields->startCandidate = startCandidate;
                        state->fields->lowerBound = lowerBound;
                        state->fields->upperBound = upperBound;
                        state->fields->hashA = hashA;
                        state->fields->hashB = hashB;
                        state->fields->pruneSymbolRuns = SendMessage(state->hwndPrune, BM_GETCHECK, 0, 0) == BST_CHECKED;
                        state->fields->pruneUnopenedBrackets = SendMessage(state->hwndPruneBrackets, BM_GETCHECK, 0, 0) == BST_CHECKED;
                        state->fields->pruneWholeCandidate = SendMessage(state->hwndPruneWhole, BM_GETCHECK, 0, 0) == BST_CHECKED;
                        state->fields->pruneAdjacentBackslashes = SendMessage(state->hwndPruneAdjacent, BM_GETCHECK, 0, 0) == BST_CHECKED;
                        state->fields->insertFromStart = getField(state->hwndInsertFromStart);
                        state->fields->insertFromEnd = getField(state->hwndInsertFromEnd);
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

} // namespace

namespace gui {

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
    RECT rect = {0, 0, 620, 580}; // desired client area
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
    HWND title = CreateWindowExA(0, "STATIC", "namebreak-bombe", WS_CHILD | WS_VISIBLE, 150, 14, 450, 28, hwndDialog, nullptr, hInstance, nullptr);
    CreateWindowExA(0, "STATIC",
                     "First-run setup. Choose how this machine should search, then press OK - your choices are saved to the configuration "
                     "file and used on every later start. Hover over a blue ? for help with any option.",
                     WS_CHILD | WS_VISIBLE, 150, 46, 455, 46, hwndDialog, nullptr, hInstance, nullptr);

    // --- Tabs, each with a page window over the tab body ---
    state.hwndTab = CreateWindowExA(0, WC_TABCONTROLA, "", WS_CHILD | WS_VISIBLE, 10, 100, 600, 430, hwndDialog, nullptr, hInstance, nullptr);
    TCITEMA tie{};
    tie.mask = TCIF_TEXT;
    tie.pszText = (LPSTR) "Coordinator";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabCoordinator, (LPARAM) &tie);
    tie.pszText = (LPSTR) "Local Search";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabLocalSearch, (LPARAM) &tie);
    tie.pszText = (LPSTR) "Local Dictionary";
    SendMessage(state.hwndTab, TCM_INSERTITEMA, kTabDictionary, (LPARAM) &tie);
    state.hwndPageCoordinator = CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, 592, 394, hwndDialog, nullptr, hInstance, nullptr);
    state.hwndPageSearch = CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, 592, 394, hwndDialog, nullptr, hInstance, nullptr);
    state.hwndPageDictionary = CreateWindowExA(0, kSetupPageClassName, "", WS_CHILD, 14, 130, 592, 394, hwndDialog, nullptr, hInstance, nullptr);

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
    syncRecordBasenames(&state);

    // --- Buttons ---
    CreateWindowExA(0, "BUTTON", "About...", WS_CHILD | WS_VISIBLE, 12, 542, 90, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupAbout, hInstance,
                     nullptr);
    CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 430, 542, 80, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupOk,
                     hInstance, nullptr);
    CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE, 520, 542, 80, 26, hwndDialog, (HMENU) (INT_PTR) kIdSetupCancel, hInstance,
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

std::vector<std::pair<std::string, std::string>> dictionaryConfigKeys(const DictionaryFields& fields) {
    auto optional = [](const std::string& text) { return text.empty() ? text : quoteDictionaryValue(text); };
    const bool hasKey = !fields.encryptionKey.empty();
    return {
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
}

} // namespace gui
