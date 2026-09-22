#ifndef NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H
#define NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H

#include <string>

#include "gui/win32/win32.h"

namespace gui {

// Tab indices in the setup dialog's SysTabControl32 - also doubles as which
// group of controls (coordinator vs. search) is shown/enabled at a time.
constexpr int kTabCoordinator = 0;
constexpr int kTabLocalSearch = 1;

struct SetupDialogFields {
    // Which tab/radio to preselect - from an existing (even if incomplete)
    // config.conf's own mode, so re-opening setup on a partially-filled-in
    // file resumes where it left off instead of always defaulting to
    // Coordinator.
    int initialTab = kTabCoordinator;
    bool initialContinuous = false;

    // Coordinator tab - see CoordinatorArgs/buildCoordinatorArgs.
    std::string username;
    std::string hostname;
    std::string serverUrl;
    std::string pollIntervalSecs = "30";

    // Local Search tab - see SearchRequest/buildSearchRequest.
    std::string alphabet;
    std::string maxBackslashCount = "0";
    std::string prefix;
    std::string suffix;
    std::string startCandidate;
    std::string lowerBound;
    std::string upperBound;
    std::string hashA;
    std::string hashB;
    bool pruneSymbolRuns = false;
    bool pruneUnopenedBrackets = false;

    // Result: "coordinator" | "bounded" | "continuous", set only if OK was
    // pressed (see showSetupDialog's return value).
    std::string mode;
};

// Shows the setup form pre-filled from `fields`; on OK, fills `fields` with
// whatever tab was active and its values (plus `fields.mode`) and returns
// true. Returns false (leaving `fields` untouched) on Cancel/close.
bool showSetupDialog(HINSTANCE hInstance, SetupDialogFields& fields);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H
