#ifndef NAMEBREAK_GUI_WIN32_ABOUT_DIALOG_H
#define NAMEBREAK_GUI_WIN32_ABOUT_DIALOG_H

#include "gui/win32/win32.h"

namespace gui {

// A classic modal About box: logo, name, one-line description, and the
// coordinator protocol version this build speaks (protocol.h - the only
// version number this project actually has). `owner` is disabled while it's
// up and gets focus back afterwards; it may be hidden (e.g. opened from the
// tray menu while the main window is minimized to the tray).
void showAboutDialog(HWND owner);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_ABOUT_DIALOG_H
