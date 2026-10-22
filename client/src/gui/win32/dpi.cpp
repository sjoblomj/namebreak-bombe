#include "gui/win32/dpi.h"

namespace {

int systemDpi() {
    static const int dpi = [] {
        HDC screen = GetDC(nullptr);
        const int d = GetDeviceCaps(screen, LOGPIXELSY);
        ReleaseDC(nullptr, screen);
        return d > 0 ? d : USER_DEFAULT_SCREEN_DPI;
    }();
    return dpi;
}

BOOL CALLBACK scaleChildEnumProc(HWND child, LPARAM) {
    // Its rectangle in its own parent's client coordinates - which is all a
    // child's position is, so the order windows are scaled in doesn't matter.
    RECT r;
    GetWindowRect(child, &r);
    MapWindowPoints(nullptr, GetParent(child), (POINT*) &r, 2);
    SetWindowPos(child, nullptr, gui::scaled(r.left), gui::scaled(r.top), gui::scaled(r.right - r.left), gui::scaled(r.bottom - r.top),
                 SWP_NOZORDER | SWP_NOACTIVATE);
    return TRUE;
}

} // namespace

namespace gui {

void enableDpiAwareness() {
    // System-aware, as SetProcessDpiAwarenessContext's SYSTEM_AWARE would
    // be, but back to Windows 7 (the floor, see win32.h).
    SetProcessDPIAware();
}

int scaled(int px) {
    return MulDiv(px, systemDpi(), USER_DEFAULT_SCREEN_DPI);
}

void scaleChildWindows(HWND parent) {
    if (systemDpi() != USER_DEFAULT_SCREEN_DPI)
        EnumChildWindows(parent, scaleChildEnumProc, 0);
}

} // namespace gui
