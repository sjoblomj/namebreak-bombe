#include "common/config.h"

#include <cctype>
#include <fstream>
#include <utility>
#include <vector>

#include "common/string_util.h"
#include "engine/candidate.h"

namespace {

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

bool parseBool(const std::string& s, bool& out) {
    std::string lower = s;
    for (char& c : lower)
        c = (char) std::tolower((unsigned char) c);
    if (lower == "true"  || lower == "1") { out = true;  return true; }
    if (lower == "false" || lower == "0") { out = false; return true; }
    return false;
}

// Inverse of unquote() above: wraps `s` in "..." only if writing it plain
// would lose meaningful leading/trailing whitespace when re-read.
std::string quoteIfNeeded(const std::string& s) {
    if (!s.empty() && (std::isspace((unsigned char) s.front()) || std::isspace((unsigned char) s.back()))) {
        return "\"" + s + "\"";
    }
    return s;
}

// config.conf is strictly one key=value per line, with no escaping for a
// literal newline - an embedded '\n' or '\r' in a value (e.g. an unusual
// auto-detected username/hostname pulled from an env var) would otherwise
// split into what looks like a second, unparseable line the next time the
// file is loaded. Quoting alone doesn't guard against this (it only
// preserves whitespace at the ends), so strip these out before a value is
// ever written back to the file.
std::string sanitizeForConfigLine(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        if (c != '\n' && c != '\r') {
            result += c;
        }
    }
    return result;
}

// appendKeyToConfigSection requires the file AND the target
// [sectionName] header to already exist - true for a config.conf a CLI run
// already created (as long as that run used the same section), but not for
// one being created from scratch (by the Windows GUI's setup dialog), and
// not for one that so far only
// ever had the *other* section (e.g. a hand-written [search]-only file, and
// the setup dialog's Coordinator tab was used). Appends a bare
// "[sectionName]" header to the end of the file if one isn't already
// present; a no-op (returns true) if it already is.
bool ensureSectionExists(const std::string& path, const std::string& sectionName, std::string& error) {
    {
        std::ifstream in(path);
        if (!in) {
            error = "cannot open " + path;
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (trim(line) == "[" + sectionName + "]")
                return true;
        }
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        error = "cannot append to " + path;
        return false;
    }
    out << "\n[" << sectionName << "]\n";
    return true;
}

// Sets (or inserts) config.conf's single top-level `mode = ...` line - the
// one key that lives before any [section] header (config.h) - to `mode`.
// Unlike ensureSectionExists/appendKeyToConfigSection, this can
// *replace* an existing value: switching which tab of the setup dialog was
// used (e.g. Coordinator -> Local Search on a config.conf that already had a
// mode from a previous run) needs the old mode value gone, not just another
// line added alongside it - loadConfigFile would otherwise just take
// whichever `mode = ...` line comes last, silently ignoring the new choice
// if it happened to land above the stale one.
bool setModeKey(const std::string& path, const std::string& mode, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    in.close();

    int firstSectionLine = -1;
    int modeLine = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::string trimmed = trim(lines[i]);
        if (!trimmed.empty() && trimmed.front() == '[') {
            firstSectionLine = (int) i;
            break;
        }
        if (trimmed.rfind("mode", 0) == 0) {
            size_t eq = trimmed.find('=');
            if (eq != std::string::npos && trim(trimmed.substr(0, eq)) == "mode") {
                modeLine = (int) i;
                break;
            }
        }
    }

    if (modeLine >= 0) {
        lines[modeLine] = "mode = " + mode;
    } else if (firstSectionLine >= 0) {
        lines.insert(lines.begin() + firstSectionLine, "mode = " + mode);
    } else {
        lines.insert(lines.begin(), "mode = " + mode);
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        error = "cannot rewrite " + path;
        return false;
    }
    for (const auto& l : lines)
        out << l << "\n";
    return true;
}

} // namespace

bool loadConfigFile(const std::string& path, ConfigFile& out, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }

    std::map<std::string, std::string>* currentSection = nullptr;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        lineNo++;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;

        if (trimmed.front() == '[') {
            if (trimmed.back() != ']') {
                error = path + ":" + std::to_string(lineNo) + ": malformed section header (missing ']')";
                return false;
            }
            std::string name = trimmed.substr(1, trimmed.size() - 2);
            if (name == "search") {
                currentSection = &out.search;
            } else if (name == "coordinator") {
                currentSection = &out.coordinator;
            } else {
                error = path + ":" + std::to_string(lineNo) + ": unknown section [" + name + "] (expected [search] or [coordinator])";
                return false;
            }
            continue;
        }

        size_t eq = trimmed.find('=');
        if (eq == std::string::npos) {
            error = path + ":" + std::to_string(lineNo) + ": expected 'key = value', got: " + trimmed;
            return false;
        }
        std::string key = trim(trimmed.substr(0, eq));
        std::string value = unquote(trim(trimmed.substr(eq + 1)));
        if (key.empty()) {
            error = path + ":" + std::to_string(lineNo) + ": empty key";
            return false;
        }

        if (currentSection == nullptr) {
            if (key == "mode") {
                out.mode = value;
            } else if (key == "matches_dir") {
                out.matchesDir = value;
            } else {
                error = path + ":" + std::to_string(lineNo) + ": '" + key +
                        "' must be inside a [search] or [coordinator] section (only 'mode' and 'matches_dir' are allowed before any section)";
                return false;
            }
        } else {
            (*currentSection)[key] = value;
        }
    }
    return true;
}

bool buildSearchRequest(const ConfigFile& config, bool continuous, SearchRequest& out, std::string& error) {
    ConfigSectionReader r(config.search);
    std::string prefix, suffix, startFilename, lowerFilename, upperFilename, hashAHex, hashBHex, maxBackslashStr;

    if (!r.getRequired("alphabet", out.alphabet, error)) return false;
    if (!r.getRequired("max_backslash_count", maxBackslashStr, error)) return false;
    if (!r.getRequired("prefix", prefix, error)) return false;
    if (!r.getRequired("suffix", suffix, error)) return false;
    if (!r.getRequired("start_candidate", startFilename, error)) return false;
    if (!r.getRequired("lower_bound", lowerFilename, error)) return false;
    if (!r.getRequired("upper_bound", upperFilename, error)) return false;
    if (!r.getRequired("hash_a", hashAHex, error)) return false;
    if (!r.getRequired("hash_b", hashBHex, error)) return false;
    std::string pruneStr = r.getOptional("prune_symbol_runs", "false");

    std::string unknown = r.firstUnknownKey();
    if (!unknown.empty()) {
        error = "unknown key '" + unknown + "' in [search] section";
        return false;
    }

    try {
        out.maxBackslashCount = std::stoi(maxBackslashStr);
    } catch (const std::exception&) {
        error = "invalid max_backslash_count: " + maxBackslashStr;
        return false;
    }
    if (!parseBool(pruneStr, out.pruneSymbolRuns)) {
        error = "invalid prune_symbol_runs: '" + pruneStr + "' (expected true/false)";
        return false;
    }
    if (!hexToU32(hashAHex, out.targetHashA)) {
        error = "invalid hash_a: " + hashAHex;
        return false;
    }
    if (!hexToU32(hashBHex, out.targetHashB)) {
        error = "invalid hash_b: " + hashBHex;
        return false;
    }

    out.prefix = prefix;
    out.suffix = suffix;
    if (!getStartCandidate(startFilename, prefix, suffix, out.startCandidate, error)) {
        return false;
    }
    out.lowerBound = removePrefixAndSuffix(lowerFilename, prefix, suffix);
    out.upperBound = removePrefixAndSuffix(upperFilename, prefix, suffix);
    out.continuous = continuous;
    out.outputFilePath = matchesFilePath(config.matchesDir, "");
    return true;
}

bool appendKeyToConfigSection(const std::string& path, const std::string& sectionName, const std::string& key, const std::string& value) {
    std::ifstream in(path);
    if (!in)
        return false;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    in.close();

    int sectionLine = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (trim(lines[i]) == "[" + sectionName + "]") {
            sectionLine = (int) i;
            break;
        }
    }
    if (sectionLine < 0)
        return false;

    lines.insert(lines.begin() + sectionLine + 1, key + " = " + quoteIfNeeded(sanitizeForConfigLine(value)));

    std::ofstream out(path, std::ios::trunc);
    if (!out)
        return false;
    for (const auto& l : lines)
        out << l << "\n";
    return true;
}

bool ensureConfigForMode(const std::string& path, const std::string& mode, const std::string& sectionName,
                          const std::vector<std::pair<std::string, std::string>>& keys, std::string& error) {
    std::ifstream probe(path);
    bool fileExists = probe.good();
    probe.close();

    if (!fileExists) {
        std::ofstream out(path, std::ios::trunc);
        if (!out) {
            error = "cannot create " + path;
            return false;
        }
        out << "mode = " << mode << "\n\n[" << sectionName << "]\n";
        for (const auto& kv : keys) {
            if (!kv.second.empty())
                out << kv.first << " = " << kv.second << "\n";
        }
        return true;
    }

    if (!setModeKey(path, mode, error))
        return false;
    if (!ensureSectionExists(path, sectionName, error))
        return false;

    ConfigFile existing;
    std::string loadErr;
    if (!loadConfigFile(path, existing, loadErr)) {
        error = loadErr;
        return false;
    }
    const std::map<std::string, std::string>& existingSection = (sectionName == "coordinator") ? existing.coordinator : existing.search;

    for (const auto& kv : keys) {
        if (kv.second.empty())
            continue;
        auto it = existingSection.find(kv.first);
        bool hasNonEmpty = it != existingSection.end() && !it->second.empty();
        if (hasNonEmpty)
            continue;
        if (!appendKeyToConfigSection(path, sectionName, kv.first, kv.second)) {
            error = "failed to write " + kv.first + " to " + path;
            return false;
        }
    }
    return true;
}
