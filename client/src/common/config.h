#ifndef NAMEBREAK_COMMON_CONFIG_H
#define NAMEBREAK_COMMON_CONFIG_H

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/matches_file.h"
#include "engine/dictionary_search.h"
#include "engine/search.h"

// Where namebreak reads its config from absent a --config <file> argument -
// see ConfigFile below. Shared so every file that needs to reference it (for
// usage/error messages) agrees on it.
inline constexpr const char* kDefaultConfigPath = "config.conf";

// The public coordinator server - what the Windows GUI's setup prefills, and
// the CLI suggests when there's no config file yet.
inline constexpr const char* kDefaultServerUrl = "https://namebreak-coordinator.fly.dev";

// namebreak's on-disk config.conf - read from "./config.conf" (the current
// working directory) unless overridden via --config <file>. A flat key=value
// file with three possible [section] headers:
//   [search]      - bounded/continuous parameters (mode picks which of the
//                   two actually runs; both take the same parameters)
//   [coordinator] - coordinator-mode parameters (see coordinator_runner.h)
//   [dictionary]  - dictionary-mode parameters (see buildDictionaryRequest)
// plus four keys allowed before any section: `mode = continuous|bounded|
// coordinator|dictionary` (overridable by passing --mode <mode> - see main()),
// `matches_dir = <directory>` (optional - see matches_file.h),
// `backend = <name>` (optional, overridable by --backend <name> - see
// backends/backends.h; unset, the first one that can run here), and
// `check_for_updates = true|false` (optional, default true - whether to look
// for a newer release at startup; see net/update_check.h). '#'-led lines
// and blank lines are ignored; a value may optionally be wrapped in "..." if
// it needs meaningful leading/trailing whitespace (no other escaping - a
// candidate/prefix/suffix containing a literal backslash needs none).
struct ConfigFile {
    std::string mode;
    std::string matchesDir = kDefaultMatchesDir;
    std::string backend;
    bool checkForUpdates = true;
    std::map<std::string, std::string> search;
    std::map<std::string, std::string> coordinator;
    // Unlike the other sections, its values are as written - not unquoted -
    // since a list value like `separators = "", "_"` quotes each item, and
    // so starts and ends with a '"' of its own. buildDictionaryRequest
    // unquotes the values that aren't lists.
    std::map<std::string, std::string> dictionary;
};

// Returns false with a human-readable `error` (including path:line) on any
// malformed line or unknown section/top-level key.
bool loadConfigFile(const std::string& path, ConfigFile& out, std::string& error);

// Builds a SearchRequest from `config`'s [search] section (and matches_dir) -
// shared by both bounded and continuous mode (`continuous` selects which).
// Returns false with `error` set if a required key is missing or a value
// doesn't parse.
bool buildSearchRequest(const ConfigFile& config, bool continuous, SearchRequest& out, std::string& error);

// Builds a DictionaryRequest from `config`'s [dictionary] section (and
// matches_dir), reading its word lists - any line one of them skips is
// described in `warnings`. Returns false with `error` set if a required key
// is missing, a value doesn't parse, a word list can't be read, or - with
// resume_from_last_candidate - the progress file is from a different search.
//
// Keys (see README.md's "Dictionary mode"):
//   builtin_dictionary  english-1 (the default) or none
//   dictionaries        more word lists, a list of file paths
//   min_words           default 1
//   max_words           required
//   separators          a list, default "" (words written together)
//   tails               a list of tail elements (see expandDictionaryTails,
//                       dictionary.h), default none
//   prefix, suffix      required
//   lower_bound, upper_bound   whole filenames, optional
//   hash_a, hash_b      required
//   encryption_key      optional, the raw key: hash type 3 of the basename
//   record_basenames    default true with an encryption_key, else false
//   record_hasha_matches   default false
//   matches_name        names the files: matches-<name>.txt, ...
//   resume_from_last_candidate   default false
// A list is items separated by commas, each either "quoted" - kept as it
// is, spaces and commas included - or not, and then trimmed.
bool buildDictionaryRequest(const ConfigFile& config, DictionaryRequest& out, std::vector<std::string>& warnings, std::string& error);

// Reads a list value as described above buildDictionaryRequest. False, with
// `error` set, on an empty unquoted item, an unterminated quote, or text
// after a quoted item before the next comma. An empty value is no items.
bool parseConfigList(const std::string& value, std::vector<std::string>& out, std::string& error);

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
