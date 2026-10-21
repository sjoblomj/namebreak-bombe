#include "gui/win32/app_config.h"

#include <map>

#include "common/config.h"
#include "common/platform.h"
#include "gui/win32/setup_dialog.h"

namespace {

using gui::SetupDialogFields;
using gui::kTabCoordinator;
using gui::kTabDictionary;
using gui::kTabLocalSearch;
using gui::kTabMeetInTheMiddle;

bool configHasServerUrl(const ConfigFile& config) {
    auto it = config.coordinator.find("server_url");
    return it != config.coordinator.end() && !it->second.empty();
}

// True if `config` has everything its own `mode` needs to actually run
// (mirrors buildCoordinatorArgs'/buildSearchRequest's own required-key
// lists, config.h) - an unset or unrecognized mode is never "ready", same as
// the CLI's own main() would refuse it.
bool isConfigReady(const ConfigFile& config) {
    if (config.mode == "coordinator")
        return configHasServerUrl(config);
    if (config.mode == "bounded" || config.mode == "continuous") {
        SearchRequest probe;
        std::string error;
        return buildSearchRequest(config, config.mode == "continuous", probe, error);
    }
    // Only its required keys: anything else wrong with it (a word list gone,
    // a progress file from another search) is said in a message box rather
    // than by showing the setup again, which can't change a key already set
    // (see ensureConfigForMode).
    if (config.mode == "dictionary") {
        for (const char* key : {"max_words", "prefix", "suffix", "hash_a", "hash_b"}) {
            auto it = config.dictionary.find(key);
            if (it == config.dictionary.end() || it->second.empty())
                return false;
        }
        return true;
    }
    // Likewise only its required keys - hashA's meeting point among them,
    // which is what the search walks (buildMeetInTheMiddleRequest,
    // config.cpp); hashB's is optional, so it isn't one.
    if (config.mode == "meet-in-the-middle") {
        for (const char* key : {"alphabet", "suffix", "meeting_point_a_seed1", "meeting_point_a_seed2", "hash_a", "hash_b",
                                "max_length"}) {
            auto it = config.meetInTheMiddle.find(key);
            if (it == config.meetInTheMiddle.end() || it->second.empty())
                return false;
        }
        return true;
    }
    return false;
}

// A [dictionary] value as written, without the quotes quoteDictionaryValue
// adds.
std::string unquoteDictionaryValue(const std::string& value) {
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        return value.substr(1, value.size() - 2);
    return value;
}

// Everything the setup form shows, taken from `config` - or, for a key it
// hasn't got (`loaded` false for no readable file at all), the auto-detected
// username/hostname and the same defaults a hand-written config.conf would
// be read with. Same spirit as the CLI's own resolveMissingIdentity
// (coordinator_runner.cpp), for every section rather than two keys.
SetupDialogFields fieldsFromConfig(const ConfigFile& config, bool loaded) {
    SetupDialogFields fields;
    auto get = [&](const std::map<std::string, std::string>& section, const std::string& key, const std::string& fallback) {
        if (!loaded)
            return fallback;
        auto it = section.find(key);
        return it != section.end() ? it->second : fallback;
    };
    fields.username = get(config.coordinator, "username", resolveUsername());
    fields.hostname = get(config.coordinator, "hostname", resolveHostname());
    fields.serverUrl = get(config.coordinator, "server_url", kDefaultServerUrl);
    fields.pollIntervalSecs = get(config.coordinator, "poll_interval_secs", "30");
    // Empty: the dialog proposes a size for this machine instead.
    fields.meetInTheMiddleTableMemory = get(config.coordinator, "meet_in_the_middle_table_memory", "");
    fields.alphabet = get(config.search, "alphabet", "");
    fields.maxBackslashCount = get(config.search, "max_backslash_count", "0");
    fields.minBackslashCount = get(config.search, "min_backslash_count", "0");
    fields.prefix = get(config.search, "prefix", "");
    fields.suffix = get(config.search, "suffix", "");
    fields.startCandidate = get(config.search, "start_candidate", "");
    fields.lowerBound = get(config.search, "lower_bound", "");
    fields.upperBound = get(config.search, "upper_bound", "");
    fields.hashA = get(config.search, "hash_a", "");
    fields.hashB = get(config.search, "hash_b", "");
    fields.pruneSymbolRuns = get(config.search, "prune_symbol_runs", "false") == "true";
    fields.pruneUnopenedBrackets = get(config.search, "prune_unopened_brackets", "false") == "true";
    fields.pruneWholeCandidate = get(config.search, "prune_whole_candidate", "false") == "true";
    fields.pruneAdjacentBackslashes = get(config.search, "prune_adjacent_backslashes", "false") == "true";
    fields.insertFromStart = get(config.search, "insert_from_start", "");
    fields.insertFromEnd = get(config.search, "insert_from_end", "");
    // The lists as written, list syntax and all; the rest without quotes.
    auto getDict = [&](const std::string& key, const std::string& fallback) {
        const std::string value = get(config.dictionary, key, fallback);
        return value == fallback ? value : unquoteDictionaryValue(value);
    };
    gui::DictionaryFields& dict = fields.dictionary;
    dict.builtinDictionary = getDict("builtin_dictionary", "english-1") != "none";
    dict.dictionaries = get(config.dictionary, "dictionaries", dict.dictionaries);
    dict.minWords = getDict("min_words", dict.minWords);
    dict.maxWords = getDict("max_words", dict.maxWords);
    dict.separators = get(config.dictionary, "separators", dict.separators);
    dict.tails = get(config.dictionary, "tails", dict.tails);
    dict.prefix = getDict("prefix", "");
    dict.suffix = getDict("suffix", "");
    dict.lowerBound = getDict("lower_bound", "");
    dict.upperBound = getDict("upper_bound", "");
    dict.hashA = getDict("hash_a", "");
    dict.hashB = getDict("hash_b", "");
    dict.encryptionKey = getDict("encryption_key", "");
    dict.recordBasenames = getDict("record_basenames", "true") == "true";
    dict.recordHashAMatches = getDict("record_hasha_matches", "false") == "true";
    dict.matchesName = getDict("matches_name", "");
    dict.resume = getDict("resume_from_last_candidate", "true") == "true";
    // [meet-in-the-middle]'s values are unquoted on the way in, as every
    // section's but [dictionary]'s are, so they go in as they come.
    gui::MeetInTheMiddleFields& mitm = fields.meetInTheMiddle;
    mitm.alphabet = get(config.meetInTheMiddle, "alphabet", "");
    mitm.prefix = get(config.meetInTheMiddle, "prefix", "");
    mitm.suffix = get(config.meetInTheMiddle, "suffix", "");
    mitm.meetingPointASeed1 = get(config.meetInTheMiddle, "meeting_point_a_seed1", "");
    mitm.meetingPointASeed2 = get(config.meetInTheMiddle, "meeting_point_a_seed2", "");
    mitm.meetingPointBSeed1 = get(config.meetInTheMiddle, "meeting_point_b_seed1", "");
    mitm.meetingPointBSeed2 = get(config.meetInTheMiddle, "meeting_point_b_seed2", "");
    mitm.hashA = get(config.meetInTheMiddle, "hash_a", "");
    mitm.hashB = get(config.meetInTheMiddle, "hash_b", "");
    mitm.minLength = get(config.meetInTheMiddle, "min_length", mitm.minLength);
    mitm.maxLength = get(config.meetInTheMiddle, "max_length", "");
    mitm.tableLength = get(config.meetInTheMiddle, "table_length", "");
    mitm.tableMemory = get(config.meetInTheMiddle, "table_memory", mitm.tableMemory);
    mitm.maxBackslashCount = get(config.meetInTheMiddle, "max_backslash_count", mitm.maxBackslashCount);
    mitm.minBackslashCount = get(config.meetInTheMiddle, "min_backslash_count", mitm.minBackslashCount);
    mitm.pruneSymbolRuns = get(config.meetInTheMiddle, "prune_symbol_runs", "false") == "true";
    mitm.pruneUnopenedBrackets = get(config.meetInTheMiddle, "prune_unopened_brackets", "false") == "true";
    mitm.pruneAdjacentBackslashes = get(config.meetInTheMiddle, "prune_adjacent_backslashes", "false") == "true";
    mitm.matchesName = get(config.meetInTheMiddle, "matches_name", "");
    mitm.resume = get(config.meetInTheMiddle, "resume_from_last_candidate", "true") == "true";
    fields.matchesDir = loaded ? config.matchesDir : kDefaultMatchesDir;

    fields.initialTab = !loaded                                                      ? kTabCoordinator
                        : config.mode == "bounded" || config.mode == "continuous" ? kTabLocalSearch
                        : config.mode == "dictionary"                             ? kTabDictionary
                        : config.mode == "meet-in-the-middle"                     ? kTabMeetInTheMiddle
                                                                                  : kTabCoordinator;
    fields.initialContinuous = loaded && config.mode == "continuous";

    return fields;
}

// What the form's `fields` mean for the file at `path`: its mode, and the
// section of whichever tab was used. `overwrite` is the settings dialog's -
// every value as the user now wants it; without it only the keys the file
// is missing are filled in, which is what first-run setup wants on a
// half-written file (see ensureConfigForMode).
bool writeConfigForFields(const std::string& path, const SetupDialogFields& fields, bool overwrite, std::string& error) {
    auto write = [&](const std::string& mode, const std::string& section,
                     const std::vector<std::pair<std::string, std::string>>& keys) {
        return overwrite ? updateConfigForMode(path, mode, section, keys, error)
                         : ensureConfigForMode(path, mode, section, keys, error);
    };
    if (fields.mode == "coordinator")
        return write("coordinator", "coordinator",
                     {{"server_url", fields.serverUrl},
                      {"username", fields.username},
                      {"hostname", fields.hostname},
                      {"poll_interval_secs", fields.pollIntervalSecs},
                      {"meet_in_the_middle_table_memory", fields.meetInTheMiddleTableMemory}});
    if (fields.mode == "dictionary")
        return write("dictionary", "dictionary", gui::dictionaryConfigKeys(fields.dictionary));
    if (fields.mode == "meet-in-the-middle")
        return write("meet-in-the-middle", "meet-in-the-middle", gui::meetInTheMiddleConfigKeys(fields.meetInTheMiddle));
    return write(fields.mode, "search",
                 {{"alphabet", fields.alphabet},
                  {"max_backslash_count", fields.maxBackslashCount},
                  {"min_backslash_count", fields.minBackslashCount},
                  {"prefix", fields.prefix},
                  {"suffix", fields.suffix},
                  {"start_candidate", fields.startCandidate},
                  {"lower_bound", fields.lowerBound},
                  {"upper_bound", fields.upperBound},
                  {"hash_a", fields.hashA},
                  {"hash_b", fields.hashB},
                  {"prune_symbol_runs", fields.pruneSymbolRuns ? "true" : "false"},
                  {"prune_unopened_brackets", fields.pruneUnopenedBrackets ? "true" : "false"},
                  {"prune_whole_candidate", fields.pruneWholeCandidate ? "true" : "false"},
                  {"prune_adjacent_backslashes", fields.pruneAdjacentBackslashes ? "true" : "false"},
                  {"insert_from_start", fields.insertFromStart},
                  {"insert_from_end", fields.insertFromEnd}});
}

} // namespace

namespace gui {

bool prepareConfig(HINSTANCE hInstance, const std::string& configPath, AppConfig& outConfig) {
    ConfigFile config;
    std::string error;
    bool loaded = loadConfigFile(configPath, config, error);
    bool ready = loaded && isConfigReady(config);

    if (!ready) {
        // Pre-filled from whatever's already in the file, even if it's
        // only some of one section's keys.
        SetupDialogFields fields = fieldsFromConfig(config, loaded);
        if (!showSetupDialog(hInstance, fields))
            return false; // user cancelled - nothing to report

        std::string writeError;
        // Only what the file is missing - a value it already has was shown
        // in the form, but changing it is what the tray menu's Settings...
        // is for (see editSettings).
        const bool writeOk = writeConfigForFields(configPath, fields, false, writeError);
        if (!writeOk) {
            MessageBoxA(nullptr, ("Failed to write " + configPath + ": " + writeError).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
        if (!loadConfigFile(configPath, config, error)) {
            MessageBoxA(nullptr, ("Failed to reload " + configPath + " after setup: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
    }

    outConfig.mode = config.mode;
    outConfig.backend = config.backend;
    if (config.mode == "coordinator") {
        if (!buildCoordinatorArgs(config, outConfig.coordinatorArgs, error)) {
            MessageBoxA(nullptr, (configPath + " [coordinator]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
        outConfig.coordinatorArgs.configPath = configPath;
    } else if (config.mode == "bounded" || config.mode == "continuous") {
        if (!buildSearchRequest(config, config.mode == "continuous", outConfig.searchRequest, error)) {
            MessageBoxA(nullptr, (configPath + " [search]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
    } else if (config.mode == "dictionary") {
        std::vector<std::string> warnings;
        const bool built = buildDictionaryRequest(config, outConfig.dictionaryRequest, warnings, error);
        if (!built) {
            MessageBoxA(nullptr, (configPath + " [dictionary]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
        // Lines of the word lists skipped - the search goes ahead without them.
        if (!warnings.empty()) {
            constexpr size_t kMaxWarnings = 10;
            std::string text = "Some lines of the word lists are skipped:\r\n";
            for (size_t i = 0; i < warnings.size() && i < kMaxWarnings; ++i)
                text += "\r\n" + warnings[i];
            if (warnings.size() > kMaxWarnings)
                text += "\r\n... and " + std::to_string(warnings.size() - kMaxWarnings) + " more";
            MessageBoxA(nullptr, text.c_str(), "namebreak", MB_OK | MB_ICONWARNING);
        }
    } else if (config.mode == "meet-in-the-middle") {
        if (!buildMeetInTheMiddleRequest(config, outConfig.meetInTheMiddleRequest, error)) {
            MessageBoxA(nullptr, (configPath + " [meet-in-the-middle]: " + error).c_str(), "namebreak", MB_OK | MB_ICONERROR);
            return false;
        }
    } else {
        MessageBoxA(nullptr, ("Unknown or missing mode in " + configPath).c_str(), "namebreak", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

bool editSettings(HINSTANCE hInstance, const std::string& configPath, HWND owner) {
    ConfigFile config;
    std::string error;
    if (!loadConfigFile(configPath, config, error)) {
        MessageBoxA(owner, ("Cannot read " + configPath + ": " + error).c_str(), "namebreak settings", MB_OK | MB_ICONERROR);
        return false;
    }
    SetupDialogFields fields = fieldsFromConfig(config, true);
    if (!showSetupDialog(hInstance, fields, true))
        return false; // cancelled - nothing to write, nothing to report
    std::string writeError;
    if (!writeConfigForFields(configPath, fields, true, writeError)) {
        MessageBoxA(owner, ("Failed to write " + configPath + ": " + writeError).c_str(), "namebreak settings", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

} // namespace gui
