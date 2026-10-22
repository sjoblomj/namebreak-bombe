#ifndef NAMEBREAK_GUI_WIN32_DPI_H
#define NAMEBREAK_GUI_WIN32_DPI_H

#include "gui/win32/win32.h"

// Display scaling. The GUI is system-DPI-aware: every window lays its
// controls out in 96-DPI pixels, as if at 100%, and is then scaled once to
// the system's DPI (scaleChildWindows) - so it is sharp at 150% or 200%
// instead of being bitmap-stretched by Windows, and the right size under
// Wine, which doesn't stretch at all. Not per-monitor: a window dragged to a
// monitor of another scaling keeps the size it was made at, which Windows
// then stretches.

namespace gui {

// Tells Windows this process scales itself - before any window exists, as
// that's when Windows decides how to treat them.
void enableDpiAwareness();

// `px`, a length in 96-DPI pixels, at the system DPI. Unchanged at 100%, and
// in a process that isn't DPI-aware (which Windows tells 96).
int scaled(int px);

// Moves and resizes every child window of `parent`, at every depth, from
// 96-DPI pixels to the system DPI. Run once, after a window's controls are
// created and before it's shown: run twice, it scales them twice.
void scaleChildWindows(HWND parent);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_DPI_H
