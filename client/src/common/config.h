#ifndef NAMEBREAK_COMMON_CONFIG_H
#define NAMEBREAK_COMMON_CONFIG_H

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/matches_file.h"
#include "engine/search.h"

// Where namebreak reads its config from absent a --config <file> argument -
// see ConfigFile below. Shared so every file that needs to reference it (for
// usage/error messages) agrees on it.
inline constexpr const char* kDefaultConfigPath = "config.conf";

// namebreak's on-disk config.conf - read from "./config.conf" (the current
// working directory) unless overridden via --config <file>. A flat key=value
// file with exactly two possible [section] headers:
//   [search]      - bounded/continuous parameters (mode picks which of the
//                   two actually runs; both take the same parameters)
//   [coordinator] - coordinator-mode parameters (see coordinator_runner.h)
// plus three keys allowed before any section: `mode = continuous|bounded|
// coordinator` (overridable by passing --mode <mode> - see main()),
// `matches_dir = <directory>` (optional - see matches_file.h), and
// `backend = <name>` (optional, overridable by --backend <name> - see
// backends/backends.h; unset, the first one that can run here). '#'-led lines
// and blank lines are ignored; a value may optionally be wrapped in "..." if
// it needs meaningful leading/trailing whitespace (no other escaping - a
// candidate/prefix/suffix containing a literal backslash needs none).
struct ConfigFile {
    std::string mode;
    std::string matchesDir = kDefaultMatchesDir;
    std::string backend;
    std::map<std::string, std::string> search;
    std::map<std::string, std::string> coordinator;
};

// Returns false with a human-readable `error` (including path:line) on any
// malformed line or unknown section/top-level key.
bool loadConfigFile(const std::string& path, ConfigFile& out, std::string& error);

// Builds a SearchRequest from `config`'s [search] section (and matches_dir) -
// shared by both bounded and continuous mode (`continuous` selects which).
// Returns false with `error` set if a required key is missing or a value
// doesn't parse.
bool buildSearchRequest(const ConfigFile& config, bool continuous, SearchRequest& out, std::string& error);

// Inserts `key = value` right after `[sectionName]`'s header line in the
// file at `path`, leaving every other line untouched - used to persist an
// interactively-confirmed value
// (see coordinator_runner.cpp's identity prompt) without disturbing the
// rest of a hand-edited config.conf. Returns false (nothing written) if the
// file can't be read/written or the section doesn't exist in it.
bool appendKeyToConfigSection(const std::string& path, const std::string& sectionName, const std::string& key, const std::string& value);

// Makes sure `path` is set up to run in `mode`: its `mode = ...` line is set
// (see setModeKey), its [sectionName] section exists, and every non-empty
// `keys` entry is present in that section - creating the file from scratch
// if it doesn't exist yet at all. Values already set in the file (that the
// caller left untouched, e.g. re-running setup after only partially filling
// it in before) are never overwritten - same "fill in only what's missing"
// behavior the CLI's own resolveMissingIdentity (coordinator_runner.cpp)
// already has for username/hostname, just generalized to any section/mode.
bool ensureConfigForMode(const std::string& path, const std::string& mode, const std::string& sectionName,
                         const std::vector<std::pair<std::string, std::string>>& keys, std::string& error);

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
            if (!used_.count(kv.first)) {
                return kv.first;
            }
        }
        return "";
    }

private:
    const std::map<std::string, std::string>& section_;
    std::set<std::string> used_;
};

#endif // NAMEBREAK_COMMON_CONFIG_H
