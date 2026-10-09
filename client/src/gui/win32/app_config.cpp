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
    return false;
}

// A [dictionary] value as written, without the quotes quoteDictionaryValue
// adds.
std::string unquoteDictionaryValue(const std::string& value) {
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        return value.substr(1, value.size() - 2);
    return value;
}

} // namespace

namespace gui {

bool prepareConfig(HINSTANCE hInstance, const std::string& configPath, AppConfig& outConfig) {
    ConfigFile config;
    std::string error;
    bool loaded = loadConfigFile(configPath, config, error);
    bool ready = loaded && isConfigReady(config);

    if (!ready) {
        SetupDialogFields fields;
        // Prefill from whatever's already in the file (even if incomplete,
        // e.g. only some [search] keys were ever set), falling back to
        // auto-detected username/hostname and sensible defaults for
        // everything else - same spirit as the CLI's own
        // resolveMissingIdentity (coordinator_runner.cpp).
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
        fields.matchesDir = loaded ? config.matchesDir : kDefaultMatchesDir;

        fields.initialTab = !loaded                                                      ? kTabCoordinator
                            : config.mode == "bounded" || config.mode == "continuous" ? kTabLocalSearch
                            : config.mode == "dictionary"                             ? kTabDictionary
                                                                                      : kTabCoordinator;
        fields.initialContinuous = loaded && config.mode == "continuous";

        if (!showSetupDialog(hInstance, fields))
            return false; // user cancelled - nothing to report

        std::string writeError;
        bool writeOk;
        if (fields.mode == "coordinator") {
            writeOk = ensureConfigForMode(configPath, "coordinator", "coordinator",
                                           {{"server_url", fields.serverUrl},
                                            {"username", fields.username},
                                            {"hostname", fields.hostname},
                                            {"poll_interval_secs", fields.pollIntervalSecs}},
                                           writeError);
        } else if (fields.mode == "dictionary") {
            writeOk = ensureConfigForMode(configPath, "dictionary", "dictionary", gui::dictionaryConfigKeys(fields.dictionary), writeError);
        } else {
            writeOk = ensureConfigForMode(configPath, fields.mode, "search",
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
                                            {"insert_from_end", fields.insertFromEnd}},
                                           writeError);
        }
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
    } else {
        MessageBoxA(nullptr, ("Unknown or missing mode in " + configPath).c_str(), "namebreak", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

} // namespace gui
