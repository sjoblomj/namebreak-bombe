#ifndef NAMEBREAK_GUI_WIN32_HARDWARE_DIALOG_H
#define NAMEBREAK_GUI_WIN32_HARDWARE_DIALOG_H

#include <atomic>
#include <string>

#include "gui/win32/win32.h"

namespace gui {

// The setup dialog's "Hardware..." button: config.conf's [hardware] section
// (see buildHardwareConfig, common/config.h) behind a window of its own. It
// shows the CPU, with its threads, and the GPUs of the backend that
// searches - `backend`, the config's, or with that empty the first that has
// any - each with a checkbox: the CPU searches instead of them, never as
// well. An integrated GPU next to a discrete one starts unchecked, as
// `gpus = auto` leaves it, and says why. For the OpenCL and Metal backends,
// each GPU's launch settings are shown too, to be changed by hand, each with
// its default beside it - in another color while it isn't the default - and
// its Optimize for hardware button times them (optimizeForHardware,
// optimize_dialog.h) on the GPUs checked. Under the GPUs is how much memory
// a coordinator run gives meet-in-the-middle work ([hardware]'s
// meet_in_the_middle_table_memory - see table_memory.h).
//
// OK writes what was changed - and only that - to `configPath` straight
// away, whether or not the setup dialog's own OK is pressed, as Optimize for
// hardware does, and makes the launch settings the process's
// (setHardwareConfig). The rest is only read when a search starts.
// `runningMode` and `searchPause` are as for optimizeForHardware.
void showHardwareDialog(HWND owner, const std::string& configPath, const std::string& backend, const std::string& runningMode,
                        std::atomic<bool>* searchPause);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_HARDWARE_DIALOG_H
