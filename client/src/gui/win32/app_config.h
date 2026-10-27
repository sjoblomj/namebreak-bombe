#ifndef NAMEBREAK_GUI_WIN32_APP_CONFIG_H
#define NAMEBREAK_GUI_WIN32_APP_CONFIG_H

#include <string>

#include "engine/dictionary_search.h"
#include "engine/meet_in_the_middle_search.h"
#include "engine/search.h"
#include "gui/win32/win32.h"
#include "net/coordinator_runner.h"

namespace gui {

// What the rest of the GUI needs to actually run, whichever mode was
// configured - exactly one of coordinatorArgs/searchRequest/
// dictionaryRequest/meetInTheMiddleRequest is meaningful, selected by `mode`
// (see workerThreadMain).
struct AppConfig {
    std::string mode;
    // config.conf's `backend` - empty for the first one that can run here.
    std::string backend;
    CoordinatorArgs coordinatorArgs;
    SearchRequest searchRequest;
    DictionaryRequest dictionaryRequest;
    MeetInTheMiddleRequest meetInTheMiddleRequest;
};

// Loads `configPath`, running the setup dialog first if it's missing or
// doesn't have everything its own mode needs (see isConfigReady). Returns
// false (having already reported why, via MessageBox) if the user cancelled
// setup or a config error couldn't be resolved.
bool prepareConfig(HINSTANCE hInstance, const std::string& configPath, AppConfig& outConfig);

// Shows that same form again for a config that's already running - the tray
// menu's Settings... (main_window.cpp) - pre-filled from `configPath` as it
// now stands, and writes whatever the user accepts back over it: every
// value of the tab they used, not only the keys that happened to be missing
// (see updateConfigForMode, common/config.h). True once the file has been
// changed, false if they cancelled or it couldn't be read or written -
// having already said why, via MessageBox owned by `owner`. The running
// search is left alone; nothing reads the file again until the next start.
// `runningMode` is the mode of the search running, for Optimize for
// hardware (see optimizeForHardware).
bool editSettings(HINSTANCE hInstance, const std::string& configPath, HWND owner, const std::string& runningMode);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_APP_CONFIG_H
