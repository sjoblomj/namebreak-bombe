#ifndef NAMEBREAK_GUI_WIN32_APP_ICONS_H
#define NAMEBREAK_GUI_WIN32_APP_ICONS_H

#include "gui/win32/win32.h"

// The logo and application icon, from the images embedded in resources/.

namespace gui {

// For a SS_BITMAP static (STM_SETIMAGE); the caller deletes it once the
// control showing it is gone.
HBITMAP makeLogoBitmap(int w, int h);

// The application icon at the system's large (taskbar, Alt-Tab) and small
// (title bar, tray) icon sizes. Never freed: a handful of icons for the life
// of the process.
HICON appIconLarge();
HICON appIconSmall();

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_APP_ICONS_H
