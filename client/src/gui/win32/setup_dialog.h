#ifndef NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H
#define NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H

#include <string>
#include <utility>
#include <vector>

#include "gui/win32/win32.h"

namespace gui {

// Tab indices in the setup dialog's SysTabControl32 - also doubles as which
// group of controls (coordinator, search or dictionary) is shown at a time.
constexpr int kTabCoordinator = 0;
constexpr int kTabLocalSearch = 1;
constexpr int kTabDictionary = 2;

// The Local Dictionary tab - see DictionaryRequest/buildDictionaryRequest. Each
// as the user typed it: the lists (dictionaries, separators, tails) in
// config.conf's own list syntax, everything else as plain text.
struct DictionaryFields {
    bool builtinDictionary = true; // english-1, or none
    std::string dictionaries;
    std::string minWords = "1";
    std::string maxWords = "2";
    std::string separators = "\"\"";
    std::string tails;
    std::string prefix;
    std::string suffix;
    std::string lowerBound;
    std::string upperBound;
    std::string hashA;
    std::string hashB;
    std::string encryptionKey;
    bool recordBasenames = true; // only written with an encryption key
    bool recordHashAMatches = false;
    std::string matchesName;
    bool resume = true;
};

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
    std::string minBackslashCount = "0";
    std::string prefix;
    std::string suffix;
    std::string startCandidate;
    std::string lowerBound;
    std::string upperBound;
    std::string hashA;
    std::string hashB;
    bool pruneSymbolRuns = false;
    bool pruneUnopenedBrackets = false;
    bool pruneWholeCandidate = false;
    bool pruneAdjacentBackslashes = false;
    std::string insertFromStart; // "text, position", or empty
    std::string insertFromEnd;

    DictionaryFields dictionary;
    // The config's matches_dir - where a dictionary search's progress file
    // is, which OK checks a resumed search against.
    std::string matchesDir;

    // Result: "coordinator" | "bounded" | "continuous" | "dictionary", set
    // only if OK was pressed (see showSetupDialog's return value).
    std::string mode;
};

// The [dictionary] section's keys for `fields`, each value as config.conf
// has it (see quoteDictionaryValue) - empty for a key to leave out.
std::vector<std::pair<std::string, std::string>> dictionaryConfigKeys(const DictionaryFields& fields);

// Shows the setup form pre-filled from `fields`; on OK, fills `fields` with
// whatever tab was active and its values (plus `fields.mode`) and returns
// true. Returns false (leaving `fields` untouched) on Cancel/close.
bool showSetupDialog(HINSTANCE hInstance, SetupDialogFields& fields);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H
