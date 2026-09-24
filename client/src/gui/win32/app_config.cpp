#include "gui/win32/app_config.h"

#include <map>

#include "common/config.h"
#include "common/platform.h"
#include "gui/win32/setup_dialog.h"

namespace {

using gui::SetupDialogFields;
using gui::kTabCoordinator;
using gui::kTabLocalSearch;

// Prefilled into the setup dialog's Server URL field when the config doesn't
// already have one.
constexpr const char* kDefaultServerUrl = "https://namebreak-coordinator.fly.dev";

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
    return false;
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
        fields.prefix = get(config.search, "prefix", "");
        fields.suffix = get(config.search, "suffix", "");
        fields.startCandidate = get(config.search, "start_candidate", "");
        fields.lowerBound = get(config.search, "lower_bound", "");
        fields.upperBound = get(config.search, "upper_bound", "");
        fields.hashA = get(config.search, "hash_a", "");
        fields.hashB = get(config.search, "hash_b", "");
        fields.pruneSymbolRuns = get(config.search, "prune_symbol_runs", "false") == "true";
        fields.pruneUnopenedBrackets = get(config.search, "prune_unopened_brackets", "false") == "true";
        fields.initialTab = (loaded && (config.mode == "bounded" || config.mode == "continuous")) ? kTabLocalSearch : kTabCoordinator;
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
        } else {
            writeOk = ensureConfigForMode(configPath, fields.mode, "search",
                                           {{"alphabet", fields.alphabet},
                                            {"max_backslash_count", fields.maxBackslashCount},
                                            {"prefix", fields.prefix},
                                            {"suffix", fields.suffix},
                                            {"start_candidate", fields.startCandidate},
                                            {"lower_bound", fields.lowerBound},
                                            {"upper_bound", fields.upperBound},
                                            {"hash_a", fields.hashA},
                                            {"hash_b", fields.hashB},
                                            {"prune_symbol_runs", fields.pruneSymbolRuns ? "true" : "false"},
                                            {"prune_unopened_brackets", fields.pruneUnopenedBrackets ? "true" : "false"}},
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
    } else {
        MessageBoxA(nullptr, ("Unknown or missing mode in " + configPath).c_str(), "namebreak", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

} // namespace gui
