#ifndef NAMEBREAK_GUI_WIN32_WIN32_H
#define NAMEBREAK_GUI_WIN32_WIN32_H

// <windows.h> and the common controls, with the settings every GUI source
// needs - include this instead of <windows.h> directly.

#ifndef _WIN32
#error "src/gui/win32 is Windows-only - see the namebreak-gui target in CMakeLists.txt"
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

#endif // NAMEBREAK_GUI_WIN32_WIN32_H
