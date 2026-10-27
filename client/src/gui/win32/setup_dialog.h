#ifndef NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H
#define NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H

#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "gui/win32/win32.h"

namespace gui {

// Tab indices in the setup dialog's SysTabControl32 - also doubles as which
// group of controls (coordinator, search, dictionary or meet-in-the-middle)
// is shown at a time.
constexpr int kTabCoordinator = 0;
constexpr int kTabLocalSearch = 1;
constexpr int kTabDictionary = 2;
constexpr int kTabMeetInTheMiddle = 3;

// Where the file's entry is in its archive's hash table - the
// hash_table_size, hash_table_first_start_slot and hash_table_last_start_slot
// keys (see MatchCheck, engine/match_check.h), as the user typed them: all
// three or none.
struct HashTableFields {
    std::string tableSize;
    std::string firstStartSlot;
    std::string lastStartSlot;
};

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
    HashTableFields hashTable;
    std::string matchesName;
    bool resume = true;
};

// The Meet in the Middle tab - see MeetInTheMiddleRequest/
// buildMeetInTheMiddleRequest. Each as the user typed it; the lengths and
// the memory as plain text, parsed the same way config.conf's own values
// are.
struct MeetInTheMiddleFields {
    std::string alphabet;
    std::string prefix;
    std::string suffix;
    // The meeting point - the hash state after the directory - in hex:
    // hashA's two seeds, then hashB's.
    std::string meetingPointASeed1;
    std::string meetingPointASeed2;
    std::string meetingPointBSeed1;
    std::string meetingPointBSeed2;
    std::string hashA;
    std::string hashB;
    std::string minLength = "1";
    std::string maxLength;
    std::string tableLength; // blank: as long as tableMemory allows
    std::string tableMemory = "1G";
    std::string maxBackslashCount = "0";
    std::string minBackslashCount = "0";
    bool pruneSymbolRuns = false;
    bool pruneUnopenedBrackets = false;
    bool pruneAdjacentBackslashes = false;
    std::string encryptionKey;
    HashTableFields hashTable;
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
    // [hardware]'s meet_in_the_middle_table_memory: how much memory this
    // client will give a meet-in-the-middle target's table, as config.conf
    // has it ("4G") -
    // "0" for none, and empty to propose a size for this machine (see
    // proposedTableMemory, table_memory.h). Not on any tab: the Hardware
    // dialog sets it, and first-run setup only fills it in where the file
    // has none (see writeConfigForFields).
    std::string meetInTheMiddleTableMemory;

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
    std::string encryptionKey;
    HashTableFields hashTable;

    DictionaryFields dictionary;
    MeetInTheMiddleFields meetInTheMiddle;
    // The config's matches_dir - where a dictionary or meet-in-the-middle
    // search's progress file is, which OK checks a resumed search against.
    std::string matchesDir;

    // For the Hardware... button (showHardwareDialog, hardware_dialog.h),
    // which saves its settings straight away, whether or not OK is pressed: the config file, its `backend` (empty for the
    // first that can run), and - shown over a running search, from
    // Settings... - that search's pause flag.
    std::string configPath;
    std::string backend;
    std::atomic<bool>* searchPause = nullptr;
    // The mode of the search that's running - empty at first run, when none
    // has started yet.
    std::string runningMode;

    // Result: "coordinator" | "bounded" | "continuous" | "dictionary" |
    // "meet-in-the-middle", set only if OK was pressed (see
    // showSetupDialog's return value).
    std::string mode;
};

// The [search] section's keys for the Local Search tab's `fields` - empty for
// a key to leave out.
std::vector<std::pair<std::string, std::string>> searchConfigKeys(const SetupDialogFields& fields);

// The [dictionary] section's keys for `fields`, each value as config.conf
// has it (see quoteDictionaryValue) - empty for a key to leave out.
std::vector<std::pair<std::string, std::string>> dictionaryConfigKeys(const DictionaryFields& fields);

// The same for the [meet-in-the-middle] section's keys.
std::vector<std::pair<std::string, std::string>> meetInTheMiddleConfigKeys(const MeetInTheMiddleFields& fields);

// Shows the setup form pre-filled from `fields`; on OK, fills `fields` with
// whatever tab was active and its values (plus `fields.mode`) and returns
// true. Returns false (leaving `fields` untouched) on Cancel/close.
// `reopened` is for the same form shown again later from the tray menu's
// Settings... rather than at first run (see editSettings, app_config.h): it
// only changes the window's title and the paragraph at the top, which would
// otherwise call itself first-run setup.
bool showSetupDialog(HINSTANCE hInstance, SetupDialogFields& fields, bool reopened = false);

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_SETUP_DIALOG_H
