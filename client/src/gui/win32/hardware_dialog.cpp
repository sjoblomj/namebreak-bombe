// The Hardware dialog: [hardware] - the CPU, or which GPUs search, and the
// OpenCL and Metal backends' launch settings for each GPU - and the memory a
// coordinator run gives meet-in-the-middle work, behind a window. A
// hand-rolled modal window with its own message loop, as the setup dialog
// is, laid out the same way: everything but the buttons in a scroll view, as
// large as the screen lets it be - what doesn't fit scrolls, either way -
// and the buttons in a strip under it.
//
// What the file says is read once, when the window opens, and on OK only
// what was changed from that is written back: a GPU whose settings weren't
// touched keeps its gpu<N> keys as they are - even ones for another card
// that has its number now, which gpuTuningFor leaves unused - and `gpus` is
// only written if the choice of GPUs changed, so that `auto` stays `auto`.

#include "gui/win32/hardware_dialog.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <thread>
#include <vector>

#include "backends/backends.h"
#include "common/config.h"
#include "gui/win32/app_icons.h"
#include "gui/win32/dpi.h"
#include "gui/win32/optimize_dialog.h"
#include "gui/win32/table_memory.h"

namespace {

using gui::scaled;

constexpr const char* kDialogClassName = "NamebreakHardwareDialog";
constexpr const char* kScrollClassName = "NamebreakHardwareScroll";
constexpr const char* kPageClassName = "NamebreakHardwarePage";
constexpr const char* kTitle = "namebreak - hardware";

constexpr int kIdOk = 501;
constexpr int kIdCancel = 502;
constexpr int kIdOptimize = 503;
constexpr int kIdGpuRadio = 504;
constexpr int kIdCpuRadio = 505;
constexpr int kIdGpuCheckbox = 506;
constexpr int kIdSetting = 507; // every launch setting's and the threads' edit
constexpr int kIdTableMemory = 508;

// Geometry, in 96-DPI pixels, as every control is placed - see scaled()
// (dpi.h). The GPUs are cells of a grid two wide, under the GPU radio; a
// setting's row is its name, its edit, its "?" and its note.
constexpr int kDialogWidth = 780, kButtonStripHeight = 44, kButtonGap = 6;
constexpr int kCellWidth = 360, kCellGap = 10, kCellsLeft = 28;
constexpr int kSettingRowHeight = 26;
constexpr int kEditX = 164, kHelpX = 224, kNoteX = 248;

// Markers in a control's GWLP_USERDATA, for PageWndProc's colors: a note
// saying a value is the default (greyed), a value that isn't (in the
// system's selection colors - which every theme, high-contrast ones too,
// makes readable together), and a help "?" (blue, as the setup dialog's).
constexpr LONG_PTR kDefaultMarker = 0x4E424446;
constexpr LONG_PTR kChangedMarker = 0x4E424443;
constexpr LONG_PTR kHelpMarker = 0x4E42484C;

// What each launch setting is, for its "?" - README.md says more.
std::string settingHelp(const GpuTuningKey& key, int defaultValue) {
    const std::string name = key.name;
    std::string what;
    if (name == "rows_per_thread")
        what = "About how many rows of a row group one work-item searches - 64 is a whole group.";
    else if (name == "batches_per_launch")
        what = "How many batches one launch searches at most.";
    else if (name == "window_chars")
        what = "How many trailing characters the GPU enumerates itself.";
    else if (name == "work_group_size")
        what = "Work-items (threads) per work-group of a row search - fewer if the GPU allows fewer.";
    else if (name == "dictionary_words_per_thread")
        what = "How many cells (words) of a batch one work-item hashes in a dictionary search.";
    else if (name == "dictionary_work_group_size")
        what = "Work-items per work-group of a dictionary search - fewer if the GPU allows fewer.";
    else if (name == "dictionary_tails_per_cell")
        what = "How many tails a dictionary cell has - only searches with tails use it.";
    return what + " None of these change what a search finds, only how fast it is.\r\n" + name + ", from " + std::to_string(key.min) +
           " to " + std::to_string(key.max) + ". Default: " + std::to_string(defaultValue) + ".";
}

std::string backendTitle(const std::string& name) {
    if (name == "opencl")
        return "OpenCL";
    if (name == "cuda")
        return "CUDA";
    if (name == "metal")
        return "Metal";
    if (name == "hip")
        return "HIP";
    return name;
}

// One launch setting of one GPU: its edit, the note beside it that says
// what the default is, and the value it had when the window opened.
struct SettingRow {
    GpuTuningKey key;
    int defaultValue = 0;
    int loaded = 0;
    HWND edit = nullptr, note = nullptr;
};

struct GpuCell {
    GpuDevice device;
    HWND frame = nullptr, checkbox = nullptr;
    bool loadedChecked = false;
    std::vector<SettingRow> rows;
};

struct DialogState {
    std::string configPath, backend, runningMode;
    std::atomic<bool>* searchPause = nullptr;
    // The backend that searches, and what [hardware] said when the window
    // opened.
    std::string searching;
    HardwareConfig loaded;
    std::vector<GpuCell> cells;
    bool loadedCpu = false;
    int hardwareThreads = 1;
    int loadedThreads = 0;
    // The meet-in-the-middle memory slider's sizes, the one it was at, and
    // whether that was [coordinator]'s, where v2026-10-22 had it.
    std::vector<uint64_t> memoryStops;
    uint64_t availableBytes = 0;
    int loadedMemoryStop = 0;
    bool memoryInCoordinator = false;
    HWND dialog = nullptr, scroll = nullptr, tooltip = nullptr;
    HWND gpuRadio = nullptr, cpuRadio = nullptr, threads = nullptr, threadsNote = nullptr;
    HWND tableMemory = nullptr, tableMemoryText = nullptr;
    HWND optimize = nullptr, ok = nullptr, cancel = nullptr;
    // How far the scroll view is scrolled, across and down.
    int scrollX = 0, scrollY = 0;
    std::deque<std::string> helpTexts; // the tooltip keeps pointers to these
    HWND owner = nullptr;
    bool done = false;
};

std::string windowText(HWND h) {
    char buf[64] = {0};
    GetWindowTextA(h, buf, sizeof buf);
    return buf;
}

bool isChecked(HWND h) {
    return SendMessage(h, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

// An edit's number: `fallback` if it's empty, -1 if it isn't a number.
int editValue(HWND edit, int fallback) {
    const std::string text = windowText(edit);
    if (text.empty())
        return fallback;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    return *end || value < 0 || value > 1 << 20 ? -1 : (int) value;
}

// What the default is, beside `edit` - greyed while the value is it, while
// the value itself is shown as changed when it isn't - or the range, for a
// value that can't be used.
void showNote(HWND edit, HWND note, int value, int defaultValue, int min, int max) {
    const bool usable = value >= min && value <= max;
    const bool isDefault = usable && value == defaultValue;
    const LONG_PTR noteMarker = isDefault ? kDefaultMarker : 0, editMarker = isDefault ? 0 : kChangedMarker;
    const std::string text =
        usable ? "(default: " + std::to_string(defaultValue) + ")" : "(must be " + std::to_string(min) + "-" + std::to_string(max) + ")";
    if (GetWindowLongPtr(edit, GWLP_USERDATA) != editMarker || windowText(note) != text) {
        SetWindowLongPtr(note, GWLP_USERDATA, noteMarker);
        SetWindowLongPtr(edit, GWLP_USERDATA, editMarker);
        SetWindowTextA(note, text.c_str());
        InvalidateRect(note, nullptr, TRUE);
        InvalidateRect(edit, nullptr, TRUE);
    }
}

int rowValue(const SettingRow& row) {
    return editValue(row.edit, row.defaultValue);
}

void syncNotes(DialogState* state) {
    for (const GpuCell& cell : state->cells) {
        for (const SettingRow& row : cell.rows)
            showNote(row.edit, row.note, rowValue(row), row.defaultValue, row.key.min, row.key.max);
    }
    showNote(state->threads, state->threadsNote, editValue(state->threads, state->hardwareThreads), state->hardwareThreads, 1, 4096);
}

void syncTableMemoryText(DialogState* state) {
    const int pos = (int) SendMessage(state->tableMemory, TBM_GETPOS, 0, 0);
    SetWindowTextA(state->tableMemoryText,
                   gui::tableMemoryLabel(state->memoryStops[(size_t) std::max(0, pos)], state->availableBytes).c_str());
}

// The CPU radio's controls, or the GPU radio's, usable - the two never
// search together.
void syncEnabled(DialogState* state) {
    const bool gpus = isChecked(state->gpuRadio);
    for (const GpuCell& cell : state->cells) {
        EnableWindow(cell.checkbox, gpus);
        for (const SettingRow& row : cell.rows)
            EnableWindow(row.edit, gpus);
    }
    EnableWindow(state->threads, !gpus);
    EnableWindow(state->optimize, gpus);
}

std::vector<GpuDevice> checkedGpus(const DialogState* state) {
    std::vector<GpuDevice> checked;
    for (const GpuCell& cell : state->cells) {
        if (isChecked(cell.checkbox))
            checked.push_back(cell.device);
    }
    return checked;
}

// The file's [hardware], or the defaults if it can't be read - with `error`
// saying why.
HardwareConfig readHardware(const std::string& configPath, std::string& error) {
    ConfigFile config;
    HardwareConfig hardware;
    if (!loadConfigFile(configPath, config, error) || !buildHardwareConfig(config, hardware, error))
        return HardwareConfig();
    return hardware;
}

// `cell`'s settings as `hardware` makes its GPU with, every one set.
void loadCellValues(GpuCell& cell, const HardwareConfig& hardware, const GpuTuning& defaults) {
    const GpuTuning tuning = withDefaults(gpuTuningFor(hardware, cell.device), defaults);
    for (SettingRow& row : cell.rows) {
        row.loaded = (tuning.*row.key.field).value_or(row.defaultValue);
        SetWindowTextA(row.edit, std::to_string(row.loaded).c_str());
    }
}

// Every value is one that can be used - or, with `error` set, the edit of
// the first that can't.
HWND firstInvalid(const DialogState* state, std::string& error) {
    for (const GpuCell& cell : state->cells) {
        for (const SettingRow& row : cell.rows) {
            const int value = rowValue(row);
            if (value < row.key.min || value > row.key.max) {
                error = "GPU " + std::to_string(cell.device.index) + "'s " + row.key.name + " must be a number from " +
                        std::to_string(row.key.min) + " to " + std::to_string(row.key.max) + ".";
                return row.edit;
            }
        }
    }
    const int threads = editValue(state->threads, state->hardwareThreads);
    if (threads < 1 || threads > 4096) {
        error = "Threads must be a number from 1 to 4096.";
        return state->threads;
    }
    return nullptr;
}

// What OK writes to [hardware]: only what changed since the window opened.
std::vector<std::pair<std::string, std::string>> changedKeys(const DialogState* state) {
    std::vector<std::pair<std::string, std::string>> keys;
    const bool cpu = isChecked(state->cpuRadio);
    bool choiceChanged = cpu != state->loadedCpu;
    for (const GpuCell& cell : state->cells)
        choiceChanged = choiceChanged || (!cpu && isChecked(cell.checkbox) != cell.loadedChecked);
    if (choiceChanged) {
        // As auto picks them, auto; every one, all; else by number.
        std::vector<GpuDevice> devices;
        for (const GpuCell& cell : state->cells)
            devices.push_back(cell.device);
        HardwareConfig automatic;
        std::vector<int> checked, picked, all;
        for (const GpuDevice& device : checkedGpus(state))
            checked.push_back(device.index);
        for (const GpuDevice& device : selectedGpus(devices, automatic))
            picked.push_back(device.index);
        for (const GpuDevice& device : devices)
            all.push_back(device.index);
        std::string gpus;
        if (cpu) {
            gpus = "none";
        } else if (checked == all && checked != picked) {
            gpus = "all";
        } else if (checked != picked) {
            for (int index : checked)
                gpus += (gpus.empty() ? "" : ", ") + std::to_string(index);
        }
        keys.emplace_back("gpus", gpus);
    }
    const int threads = editValue(state->threads, state->hardwareThreads);
    if (threads != (state->loadedThreads > 0 ? state->loadedThreads : state->hardwareThreads))
        keys.emplace_back("cpu_threads", threads == state->hardwareThreads ? "" : std::to_string(threads));

    // A GPU's own keys, if any of its values changed: each that is what it
    // would be without one - [hardware]'s key for every GPU, or the
    // default - taken out, and gpu<N>.name saying whose the rest are.
    for (const GpuCell& cell : state->cells) {
        bool changed = false;
        for (const SettingRow& row : cell.rows)
            changed = changed || rowValue(row) != row.loaded;
        if (!changed)
            continue;
        const std::string prefix = "gpu" + std::to_string(cell.device.index) + ".";
        bool own = false;
        for (const SettingRow& row : cell.rows) {
            const int value = rowValue(row);
            const int inherited = (state->loaded.tuning.*row.key.field).value_or(row.defaultValue);
            keys.emplace_back(prefix + row.key.name, value == inherited ? "" : std::to_string(value));
            own = own || value != inherited;
        }
        keys.emplace_back(prefix + "name", own ? cell.device.name : "");
    }
    const int memoryStop = (int) SendMessage(state->tableMemory, TBM_GETPOS, 0, 0);
    if (memoryStop != state->loadedMemoryStop)
        keys.emplace_back("meet_in_the_middle_table_memory", gui::bytesForConfig(state->memoryStops[(size_t) std::max(0, memoryStop)]));
    return keys;
}

void closeDialog(DialogState* state) {
    // Re-enabled before this window goes, so Windows gives the owner the
    // focus back rather than another application.
    if (state->owner)
        EnableWindow(state->owner, TRUE);
    state->done = true;
    DestroyWindow(state->dialog);
}

void accept(DialogState* state) {
    std::string error;
    if (HWND invalid = firstInvalid(state, error)) {
        MessageBoxA(state->dialog, error.c_str(), kTitle, MB_OK | MB_ICONWARNING);
        SetFocus(invalid);
        return;
    }
    if (isChecked(state->gpuRadio) && checkedGpus(state).empty()) {
        MessageBoxA(state->dialog, "Choose at least one GPU to search on - or the CPU.", kTitle, MB_OK | MB_ICONWARNING);
        return;
    }
    const auto keys = changedKeys(state);
    if (!keys.empty()) {
        if (!updateConfigSection(state->configPath, "hardware", keys, error)) {
            MessageBoxA(state->dialog, ("Failed to write " + state->configPath + ": " + error).c_str(), kTitle, MB_OK | MB_ICONERROR);
            return;
        }
        // The memory, now in [hardware], taken out of where it was.
        const bool memoryMoved = state->memoryInCoordinator && std::any_of(keys.begin(), keys.end(), [](const auto& key) {
                                     return key.first == "meet_in_the_middle_table_memory";
                                 });
        if (memoryMoved && !updateConfigSection(state->configPath, "coordinator", {{"meet_in_the_middle_table_memory", ""}}, error)) {
            MessageBoxA(state->dialog, ("Failed to write " + state->configPath + ": " + error).c_str(), kTitle, MB_OK | MB_ICONERROR);
            return;
        }
        // The launch settings, for every backend made from now on - a
        // coordinator run's next range's among them.
        const HardwareConfig hardware = readHardware(state->configPath, error);
        if (error.empty())
            setHardwareConfig(hardware);
    }
    closeDialog(state);
}

// Times the checked GPUs' settings, and shows what it saved for them. The
// other GPUs' values are left as they are in the window, changed or not.
void optimize(DialogState* state) {
    const std::vector<GpuDevice> checked = checkedGpus(state);
    if (checked.empty()) {
        MessageBoxA(state->dialog, "Check the GPUs to optimize first.", "Optimize for hardware", MB_OK | MB_ICONINFORMATION);
        return;
    }
    if (!gui::optimizeForHardware(state->dialog, state->configPath, state->backend, state->runningMode, state->searchPause, &checked))
        return;
    std::string error;
    const HardwareConfig hardware = readHardware(state->configPath, error);
    if (!error.empty())
        return;
    state->loaded = hardware;
    const GpuTuning defaults = defaultGpuTuning(state->searching);
    for (GpuCell& cell : state->cells) {
        if (isChecked(cell.checkbox))
            loadCellValues(cell, hardware, defaults);
    }
    syncNotes(state);
}

// Scrolls the view to `x` across and `y` down, each as far as it goes.
void scrollContentTo(DialogState* state, int x, int y) {
    auto clamp = [&](int bar, int pos) {
        SCROLLINFO si{};
        si.cbSize = sizeof si;
        si.fMask = SIF_RANGE | SIF_PAGE;
        GetScrollInfo(state->scroll, bar, &si);
        pos = (std::max) (0, (std::min) (pos, si.nMax + 1 - (int) si.nPage));
        si.fMask = SIF_POS;
        si.nPos = pos;
        SetScrollInfo(state->scroll, bar, &si, TRUE);
        return pos;
    };
    x = clamp(SB_HORZ, x);
    y = clamp(SB_VERT, y);
    if (x != state->scrollX || y != state->scrollY)
        ScrollWindowEx(state->scroll, state->scrollX - x, state->scrollY - y, nullptr, nullptr, nullptr, nullptr,
                       SW_SCROLLCHILDREN | SW_INVALIDATE | SW_ERASE);
    state->scrollX = x;
    state->scrollY = y;
}

// Sizes the dialog to its content, `contentHeight` tall - as large as the
// screen's work area lets it be: what doesn't fit scrolls, down or across,
// and the buttons stay on screen. As the setup dialog's layoutDialog does,
// across as well.
void layoutDialog(DialogState* state, int contentHeight) {
    const int contentWidth = scaled(kDialogWidth), buttonStripHeight = scaled(kButtonStripHeight);
    contentHeight = scaled(contentHeight);
    RECT frame = {0, 0, 0, 0};
    AdjustWindowRectEx(&frame, (DWORD) GetWindowLongPtr(state->dialog, GWL_STYLE), FALSE,
                       (DWORD) GetWindowLongPtr(state->dialog, GWL_EXSTYLE));
    const int frameWidth = frame.right - frame.left, frameHeight = frame.bottom - frame.top;
    MONITORINFO monitor{};
    monitor.cbSize = sizeof monitor;
    GetMonitorInfoA(MonitorFromWindow(state->owner ? state->owner : state->dialog, MONITOR_DEFAULTTONEAREST), &monitor);
    const RECT& work = monitor.rcWork;
    const int roomWidth = (int) (work.right - work.left) - frameWidth;
    const int roomHeight = (std::max) (buttonStripHeight, (int) (work.bottom - work.top) - frameHeight - buttonStripHeight);
    // A scrollbar each way takes room from the other way, so it can take
    // the other one: worked out until nothing changes, which twice does.
    const int vertical = GetSystemMetrics(SM_CXVSCROLL), horizontal = GetSystemMetrics(SM_CYHSCROLL);
    bool down = false, across = false;
    for (int i = 0; i < 2; ++i) {
        down = contentHeight > roomHeight - (across ? horizontal : 0);
        across = contentWidth > roomWidth - (down ? vertical : 0);
    }
    const int viewWidth = (std::min) (contentWidth + (down ? vertical : 0), roomWidth);
    const int viewHeight = (std::min) (contentHeight + (across ? horizontal : 0), roomHeight);
    const int width = viewWidth + frameWidth, height = viewHeight + buttonStripHeight + frameHeight;
    // Centered over the owner, as far as the screen allows.
    RECT anchor = work;
    if (state->owner && IsWindowVisible(state->owner))
        GetWindowRect(state->owner, &anchor);
    const int left = (std::max) ((int) work.left, (std::min) ((int) (anchor.left + anchor.right - width) / 2, (int) work.right - width));
    const int top = (std::max) ((int) work.top, (std::min) ((int) (anchor.top + anchor.bottom - height) / 2, (int) work.bottom - height));
    SetWindowPos(state->dialog, nullptr, left, top, width, height, SWP_NOZORDER | SWP_NOACTIVATE);

    SetWindowPos(state->scroll, nullptr, 0, 0, viewWidth, viewHeight, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    // A bar whose page is all of its range isn't shown.
    SCROLLINFO si{};
    si.cbSize = sizeof si;
    si.fMask = SIF_RANGE | SIF_PAGE;
    si.nMax = contentHeight - 1;
    si.nPage = (UINT) (viewHeight - (across ? horizontal : 0));
    SetScrollInfo(state->scroll, SB_VERT, &si, TRUE);
    si.nMax = contentWidth - 1;
    si.nPage = (UINT) (viewWidth - (down ? vertical : 0));
    SetScrollInfo(state->scroll, SB_HORZ, &si, TRUE);
    scrollContentTo(state, state->scrollX, state->scrollY);

    const int buttonY = viewHeight + scaled(kButtonGap);
    SetWindowPos(state->optimize, nullptr, scaled(12), buttonY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    SetWindowPos(state->ok, nullptr, viewWidth - scaled(12 + 170), buttonY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    SetWindowPos(state->cancel, nullptr, viewWidth - scaled(12 + 80), buttonY, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
}

void command(DialogState* state, int id, int code) {
    switch (id) {
        case kIdOk:
            accept(state);
            return;
        case kIdCancel:
            closeDialog(state);
            return;
        case kIdOptimize:
            optimize(state);
            return;
        case kIdGpuRadio:
        case kIdCpuRadio:
            if (!IsWindowEnabled(id == kIdGpuRadio ? state->gpuRadio : state->cpuRadio))
                return;
            SendMessage(state->gpuRadio, BM_SETCHECK, id == kIdGpuRadio ? BST_CHECKED : BST_UNCHECKED, 0);
            SendMessage(state->cpuRadio, BM_SETCHECK, id == kIdCpuRadio ? BST_CHECKED : BST_UNCHECKED, 0);
            syncEnabled(state);
            return;
        case kIdSetting:
            if (code == EN_CHANGE)
                syncNotes(state);
            return;
    }
}

// How far one of the scroll view's bars wants to go, from `pos`.
int scrolledTo(HWND hwnd, int bar, WPARAM wParam, int pos) {
    SCROLLINFO si{};
    si.cbSize = sizeof si;
    si.fMask = SIF_ALL;
    GetScrollInfo(hwnd, bar, &si);
    switch (LOWORD(wParam)) {
        case SB_LINEUP: return pos - scaled(20);
        case SB_LINEDOWN: return pos + scaled(20);
        case SB_PAGEUP: return pos - (int) si.nPage;
        case SB_PAGEDOWN: return pos + (int) si.nPage;
        case SB_THUMBTRACK:
        case SB_THUMBPOSITION: return si.nTrackPos;
        case SB_TOP: return 0;
        case SB_BOTTOM: return si.nMax;
    }
    return pos;
}

// The scroll view: scrolls, with its scrollbars and the mouse wheel.
LRESULT CALLBACK ScrollWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<DialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    if (!state)
        return DefWindowProcA(hwnd, msg, wParam, lParam);
    switch (msg) {
        case WM_VSCROLL:
            scrollContentTo(state, state->scrollX, scrolledTo(hwnd, SB_VERT, wParam, state->scrollY));
            return 0;
        case WM_HSCROLL:
            scrollContentTo(state, scrolledTo(hwnd, SB_HORZ, wParam, state->scrollX), state->scrollY);
            return 0;
        case WM_MOUSEWHEEL:
            scrollContentTo(state, state->scrollX, state->scrollY - GET_WHEEL_DELTA_WPARAM(wParam) * scaled(60) / WHEEL_DELTA);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

// The page in the scroll view that holds the controls (see
// showHardwareDialog): colors them by their markers, and hands what they
// report on to the dialog - the slider's WM_HSCROLL as a button's
// WM_COMMAND.
LRESULT CALLBACK PageWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_COMMAND:
        case WM_HSCROLL:
            return SendMessage(GetParent(GetParent(hwnd)), msg, wParam, lParam);
        case WM_CTLCOLORSTATIC:
        case WM_CTLCOLOREDIT: {
            HWND control = (HWND) lParam;
            const LONG_PTR marker = GetWindowLongPtr(control, GWLP_USERDATA);
            // A disabled edit is greyed as Windows greys it.
            if (!IsWindowEnabled(control) || (marker != kDefaultMarker && marker != kChangedMarker && marker != kHelpMarker))
                break;
            const int background = marker == kChangedMarker ? COLOR_HIGHLIGHT : COLOR_BTNFACE;
            HDC hdc = (HDC) wParam;
            SetTextColor(hdc, marker == kHelpMarker      ? RGB(0, 70, 200)
                              : marker == kChangedMarker ? GetSysColor(COLOR_HIGHLIGHTTEXT)
                                                         : GetSysColor(COLOR_GRAYTEXT));
            SetBkColor(hdc, GetSysColor(background));
            return (LRESULT) GetSysColorBrush(background);
        }
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

LRESULT CALLBACK DialogWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<DialogState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_COMMAND:
            if (state)
                command(state, LOWORD(wParam), HIWORD(wParam));
            return 0;
        case WM_HSCROLL:
            if (state && (HWND) lParam == state->tableMemory)
                syncTableMemoryText(state);
            return 0;
        case WM_CLOSE:
            if (state)
                closeDialog(state);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

BOOL CALLBACK setFontEnumProc(HWND child, LPARAM font) {
    SendMessage(child, WM_SETFONT, (WPARAM) font, TRUE);
    return TRUE;
}

// The backend that searches: the config's, or the first with GPUs here -
// as createBackend picks one - or the CPU's.
std::string searchingBackend(const std::string& requested) {
    if (!requested.empty())
        return requested;
    for (const std::string& name : backendNames()) {
        if (!gpuDevices(name).empty())
            return name;
    }
    return "cpu";
}

} // namespace

namespace gui {

void showHardwareDialog(HWND owner, const std::string& configPath, const std::string& backend, const std::string& runningMode,
                        std::atomic<bool>* searchPause) {
    HINSTANCE hInstance = GetModuleHandleA(nullptr);
    static bool classRegistered = false;
    if (!classRegistered) {
        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = DialogWndProc;
        wc.hInstance = hInstance;
        wc.hIcon = appIconLarge();
        wc.hIconSm = appIconSmall();
        wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
        wc.lpszClassName = kDialogClassName;
        RegisterClassExA(&wc);
        wc.lpfnWndProc = ScrollWndProc;
        wc.lpszClassName = kScrollClassName;
        RegisterClassExA(&wc);
        wc.lpfnWndProc = PageWndProc;
        wc.lpszClassName = kPageClassName;
        RegisterClassExA(&wc);
        classRegistered = true;
    }

    // Finding the GPUs makes the backends' runtimes look for them: a moment.
    HCURSOR cursor = SetCursor(LoadCursorA(nullptr, IDC_WAIT));
    DialogState state;
    state.configPath = configPath;
    state.backend = backend;
    state.runningMode = runningMode;
    state.searchPause = searchPause;
    state.owner = owner;
    state.searching = searchingBackend(backend);
    const std::vector<GpuDevice> devices = gpuDevices(state.searching);
    SetCursor(cursor);
    ConfigFile config;
    std::string readError;
    const bool fileRead = loadConfigFile(configPath, config, readError);
    if (fileRead && !buildHardwareConfig(config, state.loaded, readError)) {
        state.loaded = HardwareConfig();
        MessageBoxA(owner, (configPath + ": " + readError + "\r\n\r\nThe window shows the defaults instead.").c_str(), kTitle,
                    MB_OK | MB_ICONWARNING);
    }
    const std::vector<GpuDevice> selected = selectedGpus(devices, state.loaded);
    state.loadedCpu = selected.empty();
    state.hardwareThreads = (int) std::max(1u, std::thread::hardware_concurrency());
    state.loadedThreads = state.loaded.cpuThreads;
    const std::vector<GpuTuningKey>& keys = gpuTuningKeys(state.searching);
    const GpuTuning defaults = defaultGpuTuning(state.searching);
    const bool anyDiscrete = std::any_of(devices.begin(), devices.end(), [](const GpuDevice& d) { return !d.integrated; });

    // The memory a coordinator run gives meet-in-the-middle work: what the
    // file says - in [hardware], or [coordinator] as buildCoordinatorArgs
    // still reads it - or, with nothing there, what first-run setup
    // proposes.
    const MachineMemory memory = machineMemory();
    state.memoryStops = tableMemoryStopsFor(memory.totalBytes);
    state.availableBytes = memory.availableBytes;
    {
        const std::string key = "meet_in_the_middle_table_memory";
        const std::string* value = nullptr;
        if (const auto it = config.hardware.find(key); it != config.hardware.end()) {
            value = &it->second;
        } else if (const auto old = config.coordinator.find(key); old != config.coordinator.end()) {
            value = &old->second;
            state.memoryInCoordinator = true;
        }
        uint64_t bytes = 0;
        if (!value || value->empty())
            bytes = proposedTableMemory(state.memoryStops, memory.totalBytes);
        else if (!parseConfigBytes(*value, bytes))
            bytes = 0;
        state.loadedMemoryStop = stopIndexFor(state.memoryStops, bytes);
    }

    constexpr DWORD kStyle = WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_CLIPCHILDREN;
    constexpr DWORD kExStyle = WS_EX_DLGMODALFRAME;
    HWND hwnd = CreateWindowExA(kExStyle, kDialogClassName, kTitle, kStyle, CW_USEDEFAULT, CW_USEDEFAULT, 100, 100, owner, nullptr, hInstance,
                                nullptr);
    if (!hwnd)
        return;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR) &state);
    state.dialog = hwnd;
    state.scroll = CreateWindowExA(0, kScrollClassName, "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | WS_CLIPCHILDREN, 0, 0,
                                   kDialogWidth, 100, hwnd, nullptr, hInstance, nullptr);
    SetWindowLongPtr(state.scroll, GWLP_USERDATA, (LONG_PTR) &state);
    // Everything but the buttons goes on a page in the scroll view - which
    // needs WS_CLIPCHILDREN, so as not to paint over what it holds as it
    // scrolls, and would then leave the inside of a group box unpainted:
    // the page, without it, paints under its controls. Made as tall as its
    // content once that's laid out.
    HWND content = CreateWindowExA(0, kPageClassName, "", WS_CHILD | WS_VISIBLE, 0, 0, kDialogWidth, 100, state.scroll, nullptr, hInstance,
                                   nullptr);
    state.tooltip = CreateWindowExA(WS_EX_TOPMOST, TOOLTIPS_CLASSA, nullptr, WS_POPUP | TTS_NOPREFIX | TTS_ALWAYSTIP, CW_USEDEFAULT,
                                    CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, hwnd, nullptr, hInstance, nullptr);
    SendMessage(state.tooltip, TTM_SETMAXTIPWIDTH, 0, scaled(340));
    SendMessage(state.tooltip, TTM_SETDELAYTIME, TTDT_INITIAL, 250);
    SendMessage(state.tooltip, TTM_SETDELAYTIME, TTDT_AUTOPOP, 30000);

    auto control = [&](DWORD exStyle, const char* className, const std::string& text, DWORD style, int x, int y, int w, int h, int id = 0) {
        return CreateWindowExA(exStyle, className, text.c_str(), WS_CHILD | WS_VISIBLE | style, x, y, w, h, content, (HMENU) (INT_PTR) id,
                               hInstance, nullptr);
    };
    auto label = [&](const std::string& text, int x, int y, int w, int h = 18) { return control(0, "STATIC", text, 0, x, y, w, h); };
    // A blue "?" that shows `text` as a tooltip on hover - as the setup
    // dialog's addHelp makes them.
    auto addHelp = [&](int x, int y, const std::string& text) {
        HWND q = control(0, "STATIC", "?", WS_BORDER | SS_CENTER | SS_NOTIFY, x, y, 18, 18);
        SetWindowLongPtr(q, GWLP_USERDATA, kHelpMarker);
        state.helpTexts.push_back(text);
        TOOLINFOA ti{};
        ti.cbSize = sizeof(ti);
        ti.uFlags = TTF_SUBCLASS | TTF_IDISHWND;
        ti.hwnd = content;
        ti.uId = (UINT_PTR) q;
        ti.lpszText = (LPSTR) state.helpTexts.back().c_str();
        SendMessage(state.tooltip, TTM_ADDTOOLA, 0, (LPARAM) &ti);
    };

    // When what's saved is used - as Optimize for hardware says it.
    const std::string choiceWhen = runningMode.empty() ? "as soon as the search starts" : "once namebreak is restarted";
    const std::string settingsWhen = runningMode.empty()            ? "as soon as the search starts"
                                     : runningMode == "coordinator" ? "after the current range finishes"
                                                                    : "once namebreak is restarted";
    label("What this machine searches with, and how. OK saves what you change to " + configPath + " straight away: " +
              (keys.empty() || choiceWhen == settingsWhen
                   ? "it's used " + choiceWhen + "."
                   : "the GPUs' launch settings are used " + settingsWhen + ", the rest " + choiceWhen + "."),
          12, 12, kDialogWidth - 24, 34);

    // --- The CPU ---
    int y = 56;
    state.cpuRadio = control(0, "BUTTON", "Search on the CPU", BS_RADIOBUTTON | WS_GROUP, 12, y, kDialogWidth - 24, 20, kIdCpuRadio);
    y += 28;
    label("Threads:", kCellsLeft, y + 3, 60);
    state.threads = control(WS_EX_CLIENTEDGE, "EDIT", std::to_string(state.loadedThreads > 0 ? state.loadedThreads : state.hardwareThreads),
                            ES_AUTOHSCROLL | ES_NUMBER, kCellsLeft + 64, y, 56, 22, kIdSetting);
    addHelp(kCellsLeft + 124, y + 2,
            "How many threads the CPU searches with - [hardware]'s cpu_threads, from 1 to 4096. Default: one per hardware thread, " +
                std::to_string(state.hardwareThreads) + " on this machine.");
    state.threadsNote = label("", kCellsLeft + 148, y + 3, 200);
    y += 22 + 16;

    // --- The GPUs ---
    state.gpuRadio = control(0, "BUTTON", "Search on the GPUs, with the " + backendTitle(state.searching) + " backend:", BS_RADIOBUTTON, 12, y,
                             kDialogWidth - 24, 20, kIdGpuRadio);
    y += 28;
    if (devices.empty()) {
        label(backend == "cpu" || backend == "reference" ? configPath + " says backend = " + backend + ", which searches on the CPU."
                                                         : "No GPU that this build of namebreak can search with was found on this machine.",
              kCellsLeft, y, kDialogWidth - kCellsLeft - 12);
        y += 26;
        EnableWindow(state.gpuRadio, FALSE);
    } else if (keys.empty()) {
        label("The " + backendTitle(state.searching) +
                  " backend's launch settings are compiled into it - scripts/autotune tunes them (see its README).",
              kCellsLeft, y, kDialogWidth - kCellsLeft - 12);
        y += 26;
    }

    // Two to a row, each row as tall as its tallest cell.
    for (size_t first = 0; first < devices.size(); first += 2) {
        int rowHeight = 0;
        for (size_t i = first; i < devices.size() && i < first + 2; ++i) {
            const bool warn = devices[i].integrated && anyDiscrete;
            rowHeight = std::max(rowHeight, 46 + (warn ? 58 : 0) + (int) keys.size() * kSettingRowHeight + 8);
        }
        for (size_t i = first; i < devices.size() && i < first + 2; ++i) {
            const GpuDevice& device = devices[i];
            const int x = kCellsLeft + (int) (i - first) * (kCellWidth + kCellGap);
            GpuCell cell;
            cell.device = device;
            cell.frame = control(0, "BUTTON", "GPU " + std::to_string(device.index) + ": " + device.name, BS_GROUPBOX, x, y, kCellWidth,
                                 rowHeight);
            cell.loadedChecked = std::any_of(selected.begin(), selected.end(), [&](const GpuDevice& d) { return d.index == device.index; });
            cell.checkbox = control(0, "BUTTON", device.integrated ? "Search on this GPU (integrated)" : "Search on this GPU",
                                    BS_AUTOCHECKBOX, x + 10, y + 20, kCellWidth - 20, 20, kIdGpuCheckbox);
            SendMessage(cell.checkbox, BM_SETCHECK, cell.loadedChecked ? BST_CHECKED : BST_UNCHECKED, 0);
            int rowY = y + 46;
            if (device.integrated && anyDiscrete) {
                label("Built into the CPU, it shares the CPU's power and memory - and the CPU prepares the other GPUs' work, so "
                      "searching on it as well can slow them down by more than it adds.",
                      x + 28, rowY - 2, kCellWidth - 38, 56);
                rowY += 58;
            }
            for (const GpuTuningKey& key : keys) {
                SettingRow row;
                row.key = key;
                row.defaultValue = (defaults.*key.field).value_or(0);
                label(key.name, x + 10, rowY + 3, kEditX - 14);
                row.edit = control(WS_EX_CLIENTEDGE, "EDIT", "", ES_AUTOHSCROLL | ES_NUMBER, x + kEditX, rowY, 56, 22, kIdSetting);
                addHelp(x + kHelpX, rowY + 2, settingHelp(key, row.defaultValue));
                row.note = label("", x + kNoteX, rowY + 3, kCellWidth - kNoteX - 4);
                cell.rows.push_back(row);
                rowY += kSettingRowHeight;
            }
            loadCellValues(cell, state.loaded, defaults);
            state.cells.push_back(cell);
        }
        y += rowHeight + kCellGap;
    }

    // --- Meet-in-the-middle work ---
    // A group box of its own: it's about a coordinator run's memory, not
    // which parts of the machine search - but the memory is a part of the
    // machine a client agrees to spare, as its GPUs are.
    y += 6;
    const int boxX = 12, boxWidth = kDialogWidth - 24, textX = boxX + 12, textWidth = boxWidth - 24;
    control(0, "BUTTON", "Meet-in-the-middle work", BS_GROUPBOX, boxX, y, boxWidth, 186);
    label("Only used in coordinator mode, and only if the coordinator has meet-in-the-middle targets to hand out - which it is unlikely "
          "to: they are rare. Such a target is a file whose directory is unknown, searched from both ends at once (see the setup's Meet "
          "in the Middle tab), which needs a table in memory - one for each GPU that searches. The server hands this client such a target "
          "only if the table fits in what you give here; nothing else about coordinator mode depends on it.",
          textX, y + 20, textWidth, 62);
    label("This machine has " + memoryText(memory.totalBytes) + " of memory, " + memoryText(memory.availableBytes) + " of it free right now.",
          textX, y + 86, textWidth);
    label("Give at most:", textX, y + 115, 100);
    state.tableMemory = control(0, TRACKBAR_CLASSA, "", WS_TABSTOP | TBS_AUTOTICKS | TBS_HORZ, textX + 104, y + 109, 320, 32, kIdTableMemory);
    SendMessage(state.tableMemory, TBM_SETRANGE, TRUE, MAKELONG(0, (int) state.memoryStops.size() - 1));
    SendMessage(state.tableMemory, TBM_SETPAGESIZE, 0, 1);
    SendMessage(state.tableMemory, TBM_SETPOS, TRUE, state.loadedMemoryStop);
    addHelp(textX + 432, y + 115,
            "How much memory this client will give the table of a meet-in-the-middle target - [hardware]'s "
            "meet_in_the_middle_table_memory. The server hands out such a target only if its table fits in this, so None means no work "
            "of that kind at all - and no other kind is affected by it.\r\n"
            "The table is only held while such a range is being searched, and on a GPU it is held in the card's own memory too - so "
            "keep this under what your GPU has free if you search with one. Default: none.");
    state.tableMemoryText = label("", textX, y + 145, textWidth, 30);
    y += 186 + 14;
    const int contentHeight = y;
    SetWindowPos(content, nullptr, 0, 0, kDialogWidth, contentHeight, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);

    SendMessage(state.gpuRadio, BM_SETCHECK, state.loadedCpu ? BST_UNCHECKED : BST_CHECKED, 0);
    SendMessage(state.cpuRadio, BM_SETCHECK, state.loadedCpu ? BST_CHECKED : BST_UNCHECKED, 0);

    // --- Buttons, under the scroll view (placed by layoutDialog) ---
    state.optimize = CreateWindowExA(0, "BUTTON", "Optimize for hardware", WS_CHILD | WS_VISIBLE, 0, 0, 150, 26, hwnd,
                                     (HMENU) (INT_PTR) kIdOptimize, hInstance, nullptr);
    state.ok = CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 0, 0, 80, 26, hwnd, (HMENU) (INT_PTR) kIdOk,
                               hInstance, nullptr);
    state.cancel = CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE, 0, 0, 80, 26, hwnd, (HMENU) (INT_PTR) kIdCancel, hInstance,
                                   nullptr);
    EnumChildWindows(hwnd, setFontEnumProc, (LPARAM) GetStockObject(DEFAULT_GUI_FONT));
    gui::scaleChildWindows(hwnd);
    syncNotes(&state);
    syncEnabled(&state);
    syncTableMemoryText(&state);
    layoutDialog(&state, contentHeight);

    if (owner)
        EnableWindow(owner, FALSE);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    SetForegroundWindow(hwnd);

    // As the setup dialog's loop: a WM_QUIT is handed back to the loop
    // waiting for it.
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
    if (IsWindow(hwnd)) {
        SetWindowLongPtr(hwnd, GWLP_USERDATA, 0);
        if (owner)
            EnableWindow(owner, TRUE);
        DestroyWindow(hwnd);
    }
}

} // namespace gui
