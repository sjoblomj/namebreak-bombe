// "Optimize for hardware": namebreak --autotune behind a progress window. A
// hand-rolled modal window with its own message loop, as the About and setup
// dialogs are. The tuning runs on a thread of its own and never touches a
// window: it reports into ProgressState, which a timer on this thread shows.

#include "gui/win32/optimize_dialog.h"

#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include "gui/win32/app_icons.h"
#include "gui/win32/dpi.h"

namespace {

using gui::scaled;

constexpr const char* kProgressClassName = "NamebreakAutotuneProgress";
constexpr const char* kTitle = "Optimize for hardware";
constexpr int kIdCancel = 401;
constexpr UINT_PTR kTimerId = 1;
// The progress bar's range: tenths of a percent.
constexpr int kProgressRange = 1000;

// What the tuning thread reports, for the timer to show.
struct ProgressState {
    std::mutex mutex;
    std::string status = "Starting...";
    std::vector<std::string> newLines;
    std::string error;
    double fraction = 0;
    double secondsLeft = -1;
    bool finished = false;
    int result = 1;
};

struct WindowState {
    ProgressState progress;
    std::atomic<bool> abortRequested{false};
    HWND owner = nullptr;
    HWND status = nullptr, bar = nullptr, timeLeft = nullptr, log = nullptr, button = nullptr;
    bool finished = false;
    bool done = false;
};

std::string timeLeftText(double seconds) {
    if (seconds < 0)
        return "Estimated time left: working it out...";
    if (seconds < 60)
        return "Estimated time left: less than a minute";
    const int minutes = (int) std::ceil(seconds / 60);
    return minutes == 1 ? "Estimated time left: about a minute" : "Estimated time left: about " + std::to_string(minutes) + " minutes";
}

// `text` with its line breaks as an edit control wants them.
std::string withCrLf(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '\n')
            out += '\r';
        out += c;
    }
    return out;
}

void appendToLog(HWND log, const std::string& line) {
    const std::string text = (GetWindowTextLengthA(log) > 0 ? "\r\n" : "") + withCrLf(line);
    const int end = GetWindowTextLengthA(log);
    SendMessage(log, EM_SETSEL, end, end);
    SendMessage(log, EM_REPLACESEL, FALSE, (LPARAM) text.c_str());
    SendMessage(log, EM_SCROLLCARET, 0, 0);
}

// Shows what the thread has reported since the last tick - and, once it's
// over, how it ended.
void refresh(WindowState* state) {
    std::string status, error;
    std::vector<std::string> lines;
    double fraction, secondsLeft;
    bool finished;
    int result;
    {
        std::lock_guard<std::mutex> lock(state->progress.mutex);
        status = state->progress.status;
        error = state->progress.error;
        lines.swap(state->progress.newLines);
        fraction = state->progress.fraction;
        secondsLeft = state->progress.secondsLeft;
        finished = state->progress.finished;
        result = state->progress.result;
    }
    for (const std::string& line : lines)
        appendToLog(state->log, line);
    SendMessage(state->bar, PBM_SETPOS, (WPARAM) (fraction * kProgressRange), 0);
    if (!finished) {
        if (!state->abortRequested) {
            SetWindowTextA(state->status, status.c_str());
            SetWindowTextA(state->timeLeft, timeLeftText(secondsLeft).c_str());
        }
        return;
    }
    state->finished = true;
    KillTimer(GetParent(state->status), kTimerId);
    if (result == 0) {
        SendMessage(state->bar, PBM_SETPOS, kProgressRange, 0);
        SetWindowTextA(state->status, status.c_str());
        SetWindowTextA(state->timeLeft, "Finished. The settings are used from the next time namebreak starts.");
    } else if (state->abortRequested) {
        SetWindowTextA(state->status, "Stopped - nothing was changed.");
        SetWindowTextA(state->timeLeft, "");
    } else {
        SetWindowTextA(state->status, ("Failed: " + error).c_str());
        SetWindowTextA(state->timeLeft, "Nothing was changed.");
    }
    SetWindowTextA(state->button, "Close");
    EnableWindow(state->button, TRUE);
}

// The button, and closing the window: cancels while the tuning runs - it
// stops after the trial in progress, a few seconds - and closes once it's
// over.
void cancelOrClose(HWND hwnd, WindowState* state) {
    if (state->finished) {
        // Re-enabled before this window goes, so Windows gives the owner
        // the focus back rather than another application.
        if (state->owner)
            EnableWindow(state->owner, TRUE);
        state->done = true;
        DestroyWindow(hwnd);
        return;
    }
    if (state->abortRequested)
        return;
    state->abortRequested = true;
    SetWindowTextA(state->status, "Cancelling - finishing the trial in progress...");
    SetWindowTextA(state->timeLeft, "");
    SetWindowTextA(state->button, "Cancelling...");
    EnableWindow(state->button, FALSE);
}

LRESULT CALLBACK ProgressWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    auto* state = reinterpret_cast<WindowState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (msg) {
        case WM_TIMER:
            if (state && wParam == kTimerId)
                refresh(state);
            return 0;
        case WM_COMMAND:
            if (state && LOWORD(wParam) == kIdCancel)
                cancelOrClose(hwnd, state);
            return 0;
        case WM_CLOSE:
            if (state)
                cancelOrClose(hwnd, state);
            return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

} // namespace

namespace gui {

int showAutotuneProgress(HWND owner, AutotuneJob job) {
    HINSTANCE hInstance = GetModuleHandleA(nullptr);
    static bool classRegistered = false;
    if (!classRegistered) {
        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = ProgressWndProc;
        wc.hInstance = hInstance;
        wc.hIcon = appIconLarge();
        wc.hIconSm = appIconSmall();
        wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH) (COLOR_BTNFACE + 1);
        wc.lpszClassName = kProgressClassName;
        RegisterClassExA(&wc);
        classRegistered = true;
    }

    constexpr DWORD kStyle = WS_POPUP | WS_CAPTION | WS_SYSMENU;
    constexpr DWORD kExStyle = WS_EX_DLGMODALFRAME;
    RECT rect = {0, 0, scaled(500), scaled(340)}; // desired client area
    AdjustWindowRectEx(&rect, kStyle, FALSE, kExStyle);
    const int width = rect.right - rect.left, height = rect.bottom - rect.top;
    // Centered over the owner if it's showing, else on the screen.
    RECT anchor;
    if (owner && IsWindowVisible(owner) && !IsIconic(owner))
        GetWindowRect(owner, &anchor);
    else
        SetRect(&anchor, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
    const int x = anchor.left + ((anchor.right - anchor.left) - width) / 2;
    const int y = anchor.top + ((anchor.bottom - anchor.top) - height) / 2;

    WindowState state;
    state.owner = owner;
    HWND hwnd = CreateWindowExA(kExStyle, kProgressClassName, kTitle, kStyle, x, y, width, height, owner, nullptr, hInstance, nullptr);
    if (!hwnd)
        return 1;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR) &state);
    state.status = CreateWindowExA(0, "STATIC", "Starting...", WS_CHILD | WS_VISIBLE | SS_ENDELLIPSIS, 12, 12, 476, 18, hwnd, nullptr,
                                   hInstance, nullptr);
    state.bar = CreateWindowExA(0, PROGRESS_CLASSA, nullptr, WS_CHILD | WS_VISIBLE, 12, 36, 476, 20, hwnd, nullptr, hInstance, nullptr);
    SendMessage(state.bar, PBM_SETRANGE32, 0, kProgressRange);
    state.timeLeft = CreateWindowExA(0, "STATIC", timeLeftText(-1).c_str(), WS_CHILD | WS_VISIBLE, 12, 62, 476, 18, hwnd, nullptr,
                                     hInstance, nullptr);
    state.log = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL,
                                12, 88, 476, 206, hwnd, nullptr, hInstance, nullptr);
    SendMessage(state.log, EM_SETLIMITTEXT, 1 << 20, 0);
    state.button = CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 398, 304, 90, 26, hwnd,
                                   (HMENU) (INT_PTR) kIdCancel, hInstance, nullptr);
    HFONT font = (HFONT) GetStockObject(DEFAULT_GUI_FONT);
    for (HWND h : {state.status, state.timeLeft, state.log, state.button})
        SendMessage(h, WM_SETFONT, (WPARAM) font, TRUE);
    scaleChildWindows(hwnd);

    // The hooks only ever touch ProgressState - no window from that thread.
    ProgressState& progress = state.progress;
    AutotuneHooks hooks;
    hooks.onLine = [&progress](const std::string& line) {
        std::lock_guard<std::mutex> lock(progress.mutex);
        progress.newLines.push_back(line);
    };
    hooks.onError = [&progress](const std::string& error) {
        std::lock_guard<std::mutex> lock(progress.mutex);
        progress.error = error;
        progress.newLines.push_back(error);
    };
    hooks.onStatus = [&progress](const std::string& status) {
        std::lock_guard<std::mutex> lock(progress.mutex);
        progress.status = status;
    };
    hooks.onProgress = [&progress](double fraction, double secondsLeft) {
        std::lock_guard<std::mutex> lock(progress.mutex);
        progress.fraction = fraction;
        progress.secondsLeft = secondsLeft;
    };
    std::thread worker([&] {
        const int result = job(hooks, &state.abortRequested);
        std::lock_guard<std::mutex> lock(progress.mutex);
        progress.result = result;
        progress.finished = true;
    });

    if (owner)
        EnableWindow(owner, FALSE);
    SetTimer(hwnd, kTimerId, 250, nullptr);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    SetForegroundWindow(hwnd);

    MSG msg;
    while (!state.done) {
        const BOOL got = GetMessage(&msg, nullptr, 0, 0);
        if (got <= 0) {
            // A WM_QUIT while this was up - the app is shutting down: stop
            // the tuning, and hand the WM_QUIT back to the loop waiting
            // for it.
            state.abortRequested = true;
            if (got == 0)
                PostQuitMessage((int) msg.wParam);
            break;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    worker.join();
    if (IsWindow(hwnd)) {
        SetWindowLongPtr(hwnd, GWLP_USERDATA, 0);
        if (owner)
            EnableWindow(owner, TRUE);
        DestroyWindow(hwnd);
    }
    return state.progress.result;
}

void optimizeForHardware(HWND owner, const std::string& configPath, const std::string& backend, std::atomic<bool>* searchPause) {
    // Finding the backend makes it, and self-tests it: a moment.
    HCURSOR cursor = SetCursor(LoadCursorA(nullptr, IDC_WAIT));
    std::string why;
    const std::string tuned = autotuneBackend(backend, why);
    SetCursor(cursor);
    if (tuned.empty()) {
        if (why.find(kAutotuneScriptUrl) != std::string::npos) {
            if (MessageBoxA(owner, (why + "\r\n\r\nOpen that page in your browser?").c_str(), kTitle, MB_YESNO | MB_ICONWARNING) == IDYES)
                ShellExecuteA(owner, "open", kAutotuneScriptUrl, nullptr, nullptr, SW_SHOWNORMAL);
        } else {
            MessageBoxA(owner, why.c_str(), kTitle, MB_OK | MB_ICONWARNING);
        }
        return;
    }
    const bool pause = searchPause && !searchPause->load();
    const std::string question =
        "This times the " + std::string(tuned == "opencl" ? "OpenCL" : "Metal") +
        " backend's settings on this machine's GPU and saves the fastest to " + configPath +
        ", for the next time namebreak starts. It takes about five to ten minutes, with the GPU fully busy. " +
        (pause ? "The running search is paused meanwhile, and resumed afterwards. " : "") +
        "Close games and anything else that uses the GPU first: they would make the measurements meaningless.\r\n\r\nStart now?";
    if (MessageBoxA(owner, question.c_str(), kTitle, MB_OKCANCEL | MB_ICONINFORMATION) != IDOK)
        return;
    if (pause)
        searchPause->store(true);
    showAutotuneProgress(owner, [&](const AutotuneHooks& hooks, std::atomic<bool>* abortRequested) {
        return runAutotune(configPath, tuned, 2.0, abortRequested, hooks);
    });
    if (pause)
        searchPause->store(false);
}

} // namespace gui
