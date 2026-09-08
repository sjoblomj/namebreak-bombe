#include "config.h"

#include <cctype>
#include <fstream>
#include <vector>

#include "cpu-utils.h"

namespace {

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
    return s;
}

bool parseBool(const std::string& s, bool& out) {
    std::string lower = s;
    for (char& c : lower) c = (char) std::tolower((unsigned char) c);
    if (lower == "true" || lower == "1") { out = true; return true; }
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
        if (trimmed.empty() || trimmed[0] == '#') continue;

        if (trimmed.front() == '[') {
            if (trimmed.back() != ']') {
                error = path + ":" + std::to_string(lineNo) + ": malformed section header (missing ']')";
                return false;
            }
            std::string name = trimmed.substr(1, trimmed.size() - 2);
            if (name == "search") currentSection = &out.search;
            else if (name == "coordinator") currentSection = &out.coordinator;
            else {
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
            if (key != "mode") {
                error = path + ":" + std::to_string(lineNo) + ": '" + key +
                        "' must be inside a [search] or [coordinator] section (only 'mode' is allowed before any section)";
                return false;
            }
            out.mode = value;
        } else {
            (*currentSection)[key] = value;
        }
    }
    return true;
}

bool buildSearchRequest(const std::map<std::string, std::string>& section, bool continuous, SearchRequest& out, std::string& error) {
    ConfigSectionReader r(section);
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
    out.startCandidate = getStartCandidate(startFilename, prefix, suffix);
    out.lowerBound = remove_prefix_and_suffix(lowerFilename, prefix, suffix);
    out.upperBound = remove_prefix_and_suffix(upperFilename, prefix, suffix);
    out.continuous = continuous;
    return true;
}

bool appendKeyToConfigSection(const std::string& path, const std::string& sectionName, const std::string& key, const std::string& value) {
    std::ifstream in(path);
    if (!in) return false;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    in.close();

    int sectionLine = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (trim(lines[i]) == "[" + sectionName + "]") {
            sectionLine = (int) i;
            break;
        }
    }
    if (sectionLine < 0) return false;

    lines.insert(lines.begin() + sectionLine + 1, {
        "# Auto-detected - remove this line to be asked again next time.",
        key + " = " + quoteIfNeeded(value),
    });

    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    for (const auto& l : lines) out << l << "\n";
    return true;
}
