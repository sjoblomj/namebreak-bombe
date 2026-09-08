#ifndef NAMEBREAK_CUDA_CONFIG_H
#define NAMEBREAK_CUDA_CONFIG_H

#include <map>
#include <set>
#include <string>

#include "search.h"

// The one, fixed location namebreak reads its config from - see ConfigFile
// below. Shared so every file that needs to reference it (for error
// messages, or to write a discovered value back into it) agrees on it.
inline constexpr const char* kConfigPath = "config.conf";

// namebreak's on-disk config.conf - always read from "./config.conf" (the
// current working directory), never pointed to via a flag. A flat key=value
// file with exactly two possible [section] headers:
//   [search]      - bounded/continuous parameters (mode picks which of the
//                   two actually runs; both take the same parameters)
//   [coordinator] - coordinator-mode parameters (see coordinator_runner.h)
// plus one key allowed before any section: `mode = continuous|bounded|
// coordinator` (overridable by passing a mode as the program's first
// argument - see main()). '#'-led lines and blank lines are ignored; a
// value may optionally be wrapped in "..." if it needs meaningful leading/
// trailing whitespace (no other escaping - a candidate/prefix/suffix
// containing a literal backslash needs none).
struct ConfigFile {
    std::string mode;
    std::map<std::string, std::string> search;
    std::map<std::string, std::string> coordinator;
};

// Returns false with a human-readable `error` (including path:line) on any
// malformed line or unknown section/top-level key.
bool loadConfigFile(const std::string& path, ConfigFile& out, std::string& error);

// Builds a SearchRequest from a config.conf [search] section - shared by
// both bounded and continuous mode (`continuous` selects which). Returns
// false with `error` set if a required key is missing or a value doesn't
// parse.
bool buildSearchRequest(const std::map<std::string, std::string>& section, bool continuous, SearchRequest& out, std::string& error);

// Inserts `key = value` right after `[sectionName]`'s header line in the
// file at `path`, leaving every other line untouched - used to persist an
// interactively-confirmed value
// (see coordinator_runner.cpp's identity prompt) without disturbing the
// rest of a hand-edited config.conf. Returns false (nothing written) if the
// file can't be read/written or the section doesn't exist in it.
bool appendKeyToConfigSection(const std::string& path, const std::string& sectionName, const std::string& key, const std::string& value);

// Small helper for reading a config section by known key names while
// catching typos: tracks which keys were actually looked up, so a caller
// can report any left over as "unknown key" once done. Shared with
// coordinator_runner.cpp's own section (a different set of known keys).
class ConfigSectionReader {
public:
    explicit ConfigSectionReader(const std::map<std::string, std::string>& section) : section_(section) {}

    bool getRequired(const std::string& key, std::string& out, std::string& error) {
        used_.insert(key);
        auto it = section_.find(key);
        if (it == section_.end()) {
            error = "missing required key '" + key + "'";
            return false;
        }
        out = it->second;
        return true;
    }

    std::string getOptional(const std::string& key, const std::string& fallback) {
        used_.insert(key);
        auto it = section_.find(key);
        return it == section_.end() ? fallback : it->second;
    }

    // The first key present in the section but never looked up above (a
    // likely typo), or "" if every key was recognized.
    std::string firstUnknownKey() const {
        for (const auto& kv : section_) {
            if (!used_.count(kv.first)) return kv.first;
        }
        return "";
    }

private:
    const std::map<std::string, std::string>& section_;
    std::set<std::string> used_;
};

#endif // NAMEBREAK_CUDA_CONFIG_H
