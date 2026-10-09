#ifndef NAMEBREAK_GUI_WIN32_APP_CONFIG_H
#define NAMEBREAK_GUI_WIN32_APP_CONFIG_H

#include <string>

#include "engine/dictionary_search.h"
#include "engine/search.h"
#include "gui/win32/win32.h"
#include "net/coordinator_runner.h"

namespace gui {

// What the rest of the GUI needs to actually run, whichever mode was
// configured - exactly one of coordinatorArgs/searchRequest/
// dictionaryRequest is meaningful, selected by `mode` (see
// workerThreadMain).
struct AppConfig {
    std::string mode;
    // config.conf's `backend` - empty for the first one that can run here.
    std::string backend;
    CoordinatorArgs coordinatorArgs;
    SearchRequest searchRequest;
    DictionaryRequest dictionaryRequest;
};

// Loads `configPath`, running the setup dialog first if it's missing or
// doesn't have everything its own mode needs (see isConfigReady). Returns
// false (having already reported why, via MessageBox) if the user cancelled
// setup or a config error couldn't be resolved.
bool prepareConfig(HINSTANCE hInstance, const std::string& configPath, AppConfig& outConfig);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_APP_CONFIG_H
