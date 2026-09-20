#include "gui/win32/about_dialog.h"

#include <string>

#include "gui/win32/app_icons.h"
#include "net/protocol.h"

namespace {

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

} // namespace

namespace gui {

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

} // namespace gui
