#include "common/config.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

#include "common/string_util.h"
#include "engine/candidate.h"
#include "engine/wordlist.h"

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

// An insertion as `insert_from_start`/`insert_from_end` give it: its text,
// a comma, and its position - "\, 3", or with the text quoted when it has
// leading or trailing spaces: "" A ", 3". The last comma is the separator,
// so the text may have commas of its own. Empty: none.
bool parseInsertion(const std::string& s, Insertion& out) {
    out = Insertion();
    if (trim(s).empty())
        return true;
    const size_t comma = s.rfind(',');
    if (comma == std::string::npos)
        return false;
    const std::string text = unquote(trim(s.substr(0, comma)));
    const std::string position = trim(s.substr(comma + 1));
    if (text.empty() || position.empty() || position.find_first_not_of("0123456789") != std::string::npos || position.size() > 4)
        return false;
    out.text = text;
    out.position = std::stoi(position);
    return true;
}

// parseConfigBytes (config.h), which [meet-in-the-middle]'s `table_memory` and
// [coordinator]'s `meet_in_the_middle_table_memory` both read.
bool parseBytesImpl(const std::string& s, uint64_t& out) {
    const std::string text = trim(s);
    if (text.empty())
        return false;
    size_t digits = text.find_first_not_of("0123456789");
    if (digits == 0)
        return false;
    uint64_t multiplier = 1;
    if (digits != std::string::npos) {
        if (digits + 1 != text.size())
            return false;
        switch (std::tolower((unsigned char) text[digits])) {
        case 'k': multiplier = 1024; break;
        case 'm': multiplier = 1024 * 1024; break;
        case 'g': multiplier = 1024 * 1024 * 1024; break;
        default: return false;
        }
    } else {
        digits = text.size();
    }
    const std::string number = text.substr(0, digits);
    // Far more than any machine has, rather than an overflow.
    if (number.size() > 19)
        return false;
    try {
        const uint64_t value = std::stoull(number);
        if (value > UINT64_MAX / multiplier)
            return false;
        out = value * multiplier;
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

// What a section says is known about the file beyond its hashes, as written
// - see readMatchCheckKeys.
struct MatchCheckKeys {
    std::string key;
    std::string tableSize, firstStartSlot, lastStartSlot;
};

// The keys a match of both hashes is checked against (see MatchCheck,
// engine/match_check.h): `encryption_key`, unless `withKey` is false (the
// [dictionary] section reads its own), and `hash_table_size`,
// `hash_table_first_start_slot` and `hash_table_last_start_slot`. Read here,
// before the section's unknown keys are looked for; parsed by
// parseMatchCheck once they have been. `quoted`: the section keeps its
// values as written (see ConfigFile::dictionary).
MatchCheckKeys readMatchCheckKeys(ConfigSectionReader& r, bool withKey, bool quoted) {
    auto get = [&](const char* name) {
        const std::string value = r.getOptional(name, "");
        return quoted ? unquote(value) : value;
    };
    MatchCheckKeys keys;
    if (withKey)
        keys.key = get("encryption_key");
    keys.tableSize = get("hash_table_size");
    keys.firstStartSlot = get("hash_table_first_start_slot");
    keys.lastStartSlot = get("hash_table_last_start_slot");
    return keys;
}

// `keys` into `check`: the key in hex like the hashes, and the hash table
// position all three or none - a table size that's a power of two, up to
// 2^31, as an archive's is, and slots of it. False, with `error` set, on
// anything else.
bool parseMatchCheck(const MatchCheckKeys& keys, MatchCheck& check, std::string& error) {
    check = MatchCheck();
    if (!keys.key.empty()) {
        if (!hexToU32(keys.key, check.key)) {
            error = "invalid encryption_key: " + keys.key;
            return false;
        }
        check.checkKey = true;
    }
    const bool any = !keys.tableSize.empty() || !keys.firstStartSlot.empty() || !keys.lastStartSlot.empty();
    if (!any)
        return true;
    if (keys.tableSize.empty() || keys.firstStartSlot.empty() || keys.lastStartSlot.empty()) {
        error = "hash_table_size, hash_table_first_start_slot and hash_table_last_start_slot go together: give all three, or none";
        return false;
    }
    auto number = [&](const char* name, const std::string& text, uint64_t& out) {
        if (text.size() > 10 || text.find_first_not_of("0123456789") != std::string::npos) {
            error = std::string("invalid ") + name + ": '" + text + "' (expected a number)";
            return false;
        }
        out = std::stoull(text);
        return true;
    };
    uint64_t size = 0, first = 0, last = 0;
    if (!number("hash_table_size", keys.tableSize, size) || !number("hash_table_first_start_slot", keys.firstStartSlot, first) ||
        !number("hash_table_last_start_slot", keys.lastStartSlot, last))
        return false;
    if (size == 0 || size > (1ull << 31) || (size & (size - 1)) != 0) {
        error = "invalid hash_table_size: " + keys.tableSize + " (an archive's hash table's size is a power of two, up to 2^31)";
        return false;
    }
    if (first >= size || last >= size) {
        error = "hash_table_first_start_slot and hash_table_last_start_slot must be slots of the table, 0 to " + std::to_string(size - 1);
        return false;
    }
    check.checkSlot = true;
    check.hashTableSize = (uint32_t) size;
    check.firstStartSlot = (uint32_t) first;
    check.lastStartSlot = (uint32_t) last;
    return true;
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

// The candidate on the last line of `req`'s matches file, if it has one that
// belongs to this search (this prefix and suffix, only this alphabet's
// characters) - a matches file is shared by every local search, so its last
// line may be another search's.
bool lastUsableMatch(const SearchRequest& req, std::string& out) {
    std::vector<std::string> lines = readLastLines(req.outputFilePath, 1);
    if (lines.empty())
        return false;
    std::string candidate, error;
    if (!candidateOfFilename(trim(lines.back()), req.prefix, req.suffix, req.insertFromStart, req.insertFromEnd, candidate, error) ||
        candidate.empty() ||
        candidate.find_first_not_of(req.alphabet) != std::string::npos) {
        fprintf(stderr, "resume_from_last_candidate: ignoring the last line of %s ('%s'), which isn't a candidate of this search\n",
                req.outputFilePath.c_str(), lines.back().c_str());
        return false;
    }
    out = candidate;
    return true;
}

// Whether the search reaches candidate `a` after `b`: it goes through the
// lengths shortest first, and each length in `alphabet`'s order.
bool isLaterInSearchOrder(const std::string& a, const std::string& b, const std::string& alphabet) {
    if (a.size() != b.size())
        return a.size() > b.size();
    bool bIsBefore = false;
    std::string error;
    return isBeforeInAlphabet(b, a, alphabet, bIsBefore, error) && bIsBefore;
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
            } else if (name == "dictionary") {
                currentSection = &out.dictionary;
            } else if (name == "meet-in-the-middle") {
                currentSection = &out.meetInTheMiddle;
            } else if (name == "hardware") {
                currentSection = &out.hardware;
            } else {
                error = path + ":" + std::to_string(lineNo) + ": unknown section [" + name +
                        "] (expected [search], [coordinator], [dictionary], [meet-in-the-middle] or [hardware])";
                if (name == "opencl" || name == "metal")
                    error += " - the OpenCL and Metal backends' launch settings are in [hardware] now";
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
        std::string rawValue = trim(trimmed.substr(eq + 1));
        std::string value = unquote(rawValue);
        if (key.empty()) {
            error = path + ":" + std::to_string(lineNo) + ": empty key";
            return false;
        }

        if (currentSection == nullptr) {
            if (key == "mode") {
                out.mode = value;
            } else if (key == "matches_dir") {
                out.matchesDir = value;
            } else if (key == "backend") {
                out.backend = value;
            } else if (key == "check_for_updates") {
                if (value != "true" && value != "false") {
                    error = path + ":" + std::to_string(lineNo) + ": check_for_updates must be true or false, got '" + value + "'";
                    return false;
                }
                out.checkForUpdates = value == "true";
            } else {
                error = path + ":" + std::to_string(lineNo) + ": '" + key +
                        "' must be inside a [search], [coordinator], [dictionary], [meet-in-the-middle] or [hardware] section (only 'mode', 'matches_dir', "
                        "'backend' and 'check_for_updates' are allowed before any section)";
                return false;
            }
        } else {
            (*currentSection)[key] = currentSection == &out.dictionary ? rawValue : value;
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
    startFilename = r.getOptional("start_candidate", "");
    if (!r.getRequired("lower_bound", lowerFilename, error)) return false;
    if (!r.getRequired("upper_bound", upperFilename, error)) return false;
    if (!r.getRequired("hash_a", hashAHex, error)) return false;
    if (!r.getRequired("hash_b", hashBHex, error)) return false;
    std::string pruneStr = r.getOptional("prune_symbol_runs", "false");
    std::string pruneBracketsStr = r.getOptional("prune_unopened_brackets", "false");
    std::string pruneWholeStr = r.getOptional("prune_whole_candidate", "false");
    std::string minBackslashStr = r.getOptional("min_backslash_count", "0");
    std::string pruneAdjacentStr = r.getOptional("prune_adjacent_backslashes", "false");
    std::string insertFromStartStr = r.getOptional("insert_from_start", "");
    std::string insertFromEndStr = r.getOptional("insert_from_end", "");
    std::string resumeStr = r.getOptional("resume_from_last_candidate", "false");
    std::string matchesName = r.getOptional("matches_name", "");
    const MatchCheckKeys matchCheckKeys = readMatchCheckKeys(r, true, false);

    std::string unknown = r.firstUnknownKey();
    if (!unknown.empty()) {
        error = "unknown key '" + unknown + "' in [search] section";
        return false;
    }
    if (!parseMatchCheck(matchCheckKeys, out.matchCheck, error))
        return false;

    try {
        out.maxBackslashCount = std::stoi(maxBackslashStr);
    } catch (const std::exception&) {
        error = "invalid max_backslash_count: " + maxBackslashStr;
        return false;
    }
    try {
        out.minBackslashCount = std::stoi(minBackslashStr);
    } catch (const std::exception&) {
        error = "invalid min_backslash_count: " + minBackslashStr;
        return false;
    }
    if (!parseBool(pruneStr, out.pruneSymbolRuns)) {
        error = "invalid prune_symbol_runs: '" + pruneStr + "' (expected true/false)";
        return false;
    }
    if (!parseBool(pruneBracketsStr, out.pruneUnopenedBrackets)) {
        error = "invalid prune_unopened_brackets: '" + pruneBracketsStr + "' (expected true/false)";
        return false;
    }
    if (!parseBool(pruneWholeStr, out.pruneWholeCandidate)) {
        error = "invalid prune_whole_candidate: '" + pruneWholeStr + "' (expected true/false)";
        return false;
    }
    if (!parseBool(pruneAdjacentStr, out.pruneAdjacentBackslashes)) {
        error = "invalid prune_adjacent_backslashes: '" + pruneAdjacentStr + "' (expected true/false)";
        return false;
    }
    if (!parseInsertion(insertFromStartStr, out.insertFromStart)) {
        error = "invalid insert_from_start: '" + insertFromStartStr + "' (expected text, position)";
        return false;
    }
    if (!parseInsertion(insertFromEndStr, out.insertFromEnd)) {
        error = "invalid insert_from_end: '" + insertFromEndStr + "' (expected text, position)";
        return false;
    }
    bool resume = false;
    if (!parseBool(resumeStr, resume)) {
        error = "invalid resume_from_last_candidate: '" + resumeStr + "' (expected true/false)";
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
    out.lowerBound = removePrefixAndSuffix(lowerFilename, prefix, suffix);
    out.upperBound = removePrefixAndSuffix(upperFilename, prefix, suffix);
    out.continuous = continuous;
    out.outputFilePath = namedFilePath(config.matchesDir, "matches", matchesName);

    // Without a start_candidate, from the beginning: the shortest candidates
    // in continuous mode (an empty start candidate, see SearchRequest), the
    // lower bound in bounded mode - which searches its start candidate's
    // length only, so an empty one would leave nothing to search.
    if (startFilename.empty()) {
        out.startCandidate = continuous ? "" : out.lowerBound;
    } else if (!candidateOfFilename(startFilename, prefix, suffix, out.insertFromStart, out.insertFromEnd, out.startCandidate, error)) {
        error = "invalid start_candidate: " + error;
        return false;
    }

    if (resume) {
        std::string lastMatch;
        if (lastUsableMatch(out, lastMatch) && isLaterInSearchOrder(lastMatch, out.startCandidate, out.alphabet)) {
            printf("Resuming from the last match in %s: %s%s%s\n", out.outputFilePath.c_str(), prefix.c_str(),
                   insertIntoCandidate(lastMatch, out.insertFromStart, out.insertFromEnd).c_str(), suffix.c_str());
            out.startCandidate = lastMatch;
        }
    }
    return true;
}

bool buildMeetInTheMiddleRequest(const ConfigFile& config, MeetInTheMiddleRequest& out, std::string& error) {
    out = MeetInTheMiddleRequest();
    ConfigSectionReader r(config.meetInTheMiddle);
    std::string suffix, maxLengthStr, hashAHex, hashBHex;
    std::string meetingPointHex[4];
    static const char* kMeetInTheMiddleKeys[4] = {"meeting_point_a_seed1", "meeting_point_a_seed2", "meeting_point_b_seed1", "meeting_point_b_seed2"};

    if (!r.getRequired("alphabet", out.alphabet, error)) return false;
    out.prefix = r.getOptional("prefix", "");
    if (!r.getRequired("suffix", suffix, error)) return false;
    // hashA's meeting point is the search itself; hashB's only checks it
    // against hash_b before the search starts, so it may be left out - but
    // half of a state means nothing, so it's both seeds or neither.
    for (int i = 0; i < 2; ++i) {
        if (!r.getRequired(kMeetInTheMiddleKeys[i], meetingPointHex[i], error))
            return false;
    }
    for (int i = 2; i < 4; ++i)
        meetingPointHex[i] = r.getOptional(kMeetInTheMiddleKeys[i], "");
    if (meetingPointHex[2].empty() != meetingPointHex[3].empty()) {
        error = "meeting_point_b_seed1 and meeting_point_b_seed2 go together: give both, or neither (hashB's meeting point is "
                "optional - the search walks hashA's state, and hashB's is only used to check that the meeting point really is the "
                "state where the suffix begins)";
        return false;
    }
    out.hasMeetingPointB = !meetingPointHex[2].empty();
    if (!r.getRequired("hash_a", hashAHex, error)) return false;
    if (!r.getRequired("hash_b", hashBHex, error)) return false;
    const std::string minLengthStr = r.getOptional("min_length", "1");
    if (!r.getRequired("max_length", maxLengthStr, error)) return false;
    const std::string tableLengthStr = r.getOptional("table_length", "");
    const std::string tableMemoryStr = r.getOptional("table_memory", "1G");
    const std::string pruneStr = r.getOptional("prune_symbol_runs", "false");
    const std::string pruneBracketsStr = r.getOptional("prune_unopened_brackets", "false");
    const std::string pruneAdjacentStr = r.getOptional("prune_adjacent_backslashes", "false");
    const std::string maxBackslashStr = r.getOptional("max_backslash_count", "0");
    const std::string minBackslashStr = r.getOptional("min_backslash_count", "0");
    const std::string matchesName = r.getOptional("matches_name", "");
    const std::string resumeStr = r.getOptional("resume_from_last_candidate", "false");
    const MatchCheckKeys matchCheckKeys = readMatchCheckKeys(r, true, false);

    const std::string unknown = r.firstUnknownKey();
    if (!unknown.empty()) {
        error = "unknown key '" + unknown + "' in [meet-in-the-middle] section";
        return false;
    }
    if (!parseMatchCheck(matchCheckKeys, out.matchCheck, error))
        return false;

    uint32_t seeds[4] = {0, 0, 0, 0};
    for (int i = 0; i < (out.hasMeetingPointB ? 4 : 2); ++i) {
        if (!hexToU32(meetingPointHex[i], seeds[i])) {
            error = std::string("invalid ") + kMeetInTheMiddleKeys[i] + ": " + meetingPointHex[i];
            return false;
        }
    }
    out.meetingPointA = {seeds[0], seeds[1]};
    out.meetingPointB = {seeds[2], seeds[3]};
    if (!hexToU32(hashAHex, out.targetHashA)) {
        error = "invalid hash_a: " + hashAHex;
        return false;
    }
    if (!hexToU32(hashBHex, out.targetHashB)) {
        error = "invalid hash_b: " + hashBHex;
        return false;
    }

    auto parseCount = [&](const std::string& key, const std::string& value, int& count) {
        if (value.empty() || value.size() > 3 || value.find_first_not_of("0123456789") != std::string::npos) {
            error = "invalid " + key + ": '" + value + "' (expected a number)";
            return false;
        }
        count = std::stoi(value);
        return true;
    };
    if (!parseCount("min_length", minLengthStr, out.minLength) || !parseCount("max_length", maxLengthStr, out.maxLength))
        return false;
    if (!parseCount("max_backslash_count", maxBackslashStr, out.rules.maxBackslashCount) ||
        !parseCount("min_backslash_count", minBackslashStr, out.rules.minBackslashCount))
        return false;
    if (!parseBool(pruneStr, out.rules.symbolRuns)) {
        error = "invalid prune_symbol_runs: '" + pruneStr + "' (expected true/false)";
        return false;
    }
    if (!parseBool(pruneBracketsStr, out.rules.unopenedBrackets)) {
        error = "invalid prune_unopened_brackets: '" + pruneBracketsStr + "' (expected true/false)";
        return false;
    }
    if (!parseBool(pruneAdjacentStr, out.rules.adjacentBackslashes)) {
        error = "invalid prune_adjacent_backslashes: '" + pruneAdjacentStr + "' (expected true/false)";
        return false;
    }
    bool resume = false;
    if (!parseBool(resumeStr, resume)) {
        error = "invalid resume_from_last_candidate: '" + resumeStr + "' (expected true/false)";
        return false;
    }
    if (!parseConfigBytes(tableMemoryStr, out.tableBytes)) {
        error = "invalid table_memory: '" + tableMemoryStr + "' (expected a number of bytes, or one with a K, M or G after it)";
        return false;
    }

    out.suffix = suffix;
    // The table's length is settled here rather than left to the search, so
    // that it's part of what the progress file's fingerprint identifies: it
    // decides where the walk's nodes are cut, and a `next` from a search
    // that split the directory elsewhere means something else entirely.
    if (tableLengthStr.empty()) {
        out.tableLength = meetInTheMiddleTableLengthForBytes((int) out.alphabet.size(), out.maxLength, out.tableBytes);
    } else if (!parseCount("table_length", tableLengthStr, out.tableLength)) {
        return false;
    } else if (out.tableLength > out.maxLength) {
        error = "table_length " + tableLengthStr + " is more than max_length " + maxLengthStr;
        return false;
    }
    out.outputFilePath = namedFilePath(config.matchesDir, "matches", matchesName);
    out.progressFilePath = namedFilePath(config.matchesDir, "meet-in-the-middle", matchesName);

    if (resume) {
        MeetInTheMiddleProgress progress;
        bool exists = false;
        if (!readMeetInTheMiddleProgress(out.progressFilePath, progress, exists, error))
            return false;
        if (!exists) {
            printf("resume_from_last_candidate: no progress file %s yet - starting from the beginning\n", out.progressFilePath.c_str());
        } else if (progress.fingerprint != meetInTheMiddleFingerprint(out)) {
            error = "resume_from_last_candidate: " + out.progressFilePath +
                    " is from a search with different settings (alphabet, prefix, suffix, meeting point, hashes, lengths, table length or "
                    "rules) - remove it, or set resume_from_last_candidate = false, to start this one from the beginning";
            return false;
        } else {
            out.startNumber = progress.next;
            printf("Resuming from node %llu, from %s\n", (unsigned long long) progress.next, out.progressFilePath.c_str());
        }
    }
    return true;
}

bool parseConfigBytes(const std::string& text, uint64_t& out) {
    return parseBytesImpl(text, out);
}

bool parseConfigList(const std::string& value, std::vector<std::string>& out, std::string& error) {
    out.clear();
    if (trim(value).empty())
        return true;
    size_t i = 0;
    for (;;) {
        // One item: up to the next comma, or a quoted one.
        while (i < value.size() && (value[i] == ' ' || value[i] == '\t'))
            ++i;
        if (i < value.size() && value[i] == '"') {
            const size_t close = value.find('"', i + 1);
            if (close == std::string::npos) {
                error = "unterminated quote in '" + value + "'";
                return false;
            }
            out.push_back(value.substr(i + 1, close - i - 1));
            i = close + 1;
            while (i < value.size() && (value[i] == ' ' || value[i] == '\t'))
                ++i;
            if (i < value.size() && value[i] != ',') {
                error = "expected a comma after the quoted item \"" + out.back() + "\" in '" + value + "'";
                return false;
            }
        } else {
            const size_t comma = std::min(value.find(',', i), value.size());
            const std::string item = trim(value.substr(i, comma - i));
            if (item.empty()) {
                error = "empty item in '" + value + "' - quote it (\"\") if it's meant to be empty";
                return false;
            }
            out.push_back(item);
            i = comma;
        }
        if (i >= value.size())
            return true;
        ++i; // the comma
    }
}

bool buildDictionaryRequest(const ConfigFile& config, DictionaryRequest& out, std::vector<std::string>& warnings, std::string& error) {
    out = DictionaryRequest();
    ConfigSectionReader r(config.dictionary);
    std::string maxWordsStr, prefix, suffix, hashAHex, hashBHex;
    std::string builtin = unquote(r.getOptional("builtin_dictionary", kEnglish1));
    std::string dictionariesStr = r.getOptional("dictionaries", "");
    std::string minWordsStr = unquote(r.getOptional("min_words", "1"));
    if (!r.getRequired("max_words", maxWordsStr, error)) return false;
    std::string separatorsStr = r.getOptional("separators", "\"\"");
    std::string tailsStr = r.getOptional("tails", "");
    if (!r.getRequired("prefix", prefix, error)) return false;
    if (!r.getRequired("suffix", suffix, error)) return false;
    const bool hasLower = config.dictionary.count("lower_bound") > 0, hasUpper = config.dictionary.count("upper_bound") > 0;
    std::string lowerStr = unquote(r.getOptional("lower_bound", ""));
    std::string upperStr = unquote(r.getOptional("upper_bound", ""));
    if (!r.getRequired("hash_a", hashAHex, error)) return false;
    if (!r.getRequired("hash_b", hashBHex, error)) return false;
    std::string keyHex = unquote(r.getOptional("encryption_key", ""));
    std::string recordStr = unquote(r.getOptional("record_basenames", ""));
    std::string recordHashAStr = unquote(r.getOptional("record_hasha_matches", "false"));
    std::string matchesName = unquote(r.getOptional("matches_name", ""));
    std::string resumeStr = unquote(r.getOptional("resume_from_last_candidate", "false"));
    // Its encryption_key is read above: it picks the candidates to hash, and
    // isn't checked again - with record_hasha_matches, so that a wrong key
    // can't hide the file.
    const MatchCheckKeys matchCheckKeys = readMatchCheckKeys(r, false, true);

    std::string unknown = r.firstUnknownKey();
    if (!unknown.empty()) {
        error = "unknown key '" + unknown + "' in [dictionary] section";
        return false;
    }
    if (!parseMatchCheck(matchCheckKeys, out.matchCheck, error))
        return false;
    maxWordsStr = unquote(maxWordsStr);
    prefix = unquote(prefix);
    suffix = unquote(suffix);
    hashAHex = unquote(hashAHex);
    hashBHex = unquote(hashBHex);

    auto parseCount = [&](const std::string& key, const std::string& value, int& count) {
        if (value.empty() || value.size() > 3 || value.find_first_not_of("0123456789") != std::string::npos) {
            error = "invalid " + key + ": '" + value + "' (expected a number)";
            return false;
        }
        count = std::stoi(value);
        return true;
    };
    if (!parseCount("min_words", minWordsStr, out.pattern.minWords) || !parseCount("max_words", maxWordsStr, out.pattern.maxWords))
        return false;
    if (!hexToU32(hashAHex, out.targetHashA)) {
        error = "invalid hash_a: " + hashAHex;
        return false;
    }
    if (!hexToU32(hashBHex, out.targetHashB)) {
        error = "invalid hash_b: " + hashBHex;
        return false;
    }
    const bool hasKey = !keyHex.empty();
    if (hasKey && !hexToU32(keyHex, out.basenameKey)) {
        error = "invalid encryption_key: " + keyHex;
        return false;
    }
    out.checkBasename = hasKey;
    out.recordBasenames = hasKey;
    if (!recordStr.empty() && !parseBool(recordStr, out.recordBasenames)) {
        error = "invalid record_basenames: '" + recordStr + "' (expected true/false)";
        return false;
    }
    if (out.recordBasenames && !hasKey) {
        error = "record_basenames needs an encryption_key to compare the basenames to";
        return false;
    }
    if (!parseBool(recordHashAStr, out.recordHashAMatches)) {
        error = "invalid record_hasha_matches: '" + recordHashAStr + "' (expected true/false)";
        return false;
    }
    bool resume = false;
    if (!parseBool(resumeStr, resume)) {
        error = "invalid resume_from_last_candidate: '" + resumeStr + "' (expected true/false)";
        return false;
    }

    std::vector<std::string> separators, dictionaries;
    if (!parseConfigList(separatorsStr, separators, error)) {
        error = "invalid separators: " + error;
        return false;
    }
    if (!parseConfigList(dictionariesStr, dictionaries, error)) {
        error = "invalid dictionaries: " + error;
        return false;
    }
    for (std::string& separator : separators)
        out.pattern.separators.push_back(normalizeMpqName(separator));
    std::vector<std::string> tailElements;
    if (!parseConfigList(tailsStr, tailElements, error) || !expandDictionaryTails(tailElements, out.pattern.tails, error)) {
        error = "invalid tails: " + error;
        return false;
    }

    // The words: the built-in dictionary's and every list's, together.
    std::vector<std::string> words;
    if (builtin != "none") {
        if (!isBuiltinWordList(builtin)) {
            error = "unknown builtin_dictionary '" + builtin + "' (expected " + kEnglish1 + " or none)";
            return false;
        }
        words = builtinWordList(builtin);
        out.wordSource = builtin;
    }
    for (const std::string& path : dictionaries) {
        if (!loadWordListFile(path, words, warnings, error))
            return false;
        out.wordSource += (out.wordSource.empty() ? "" : " + ") + path;
    }
    out.pattern.words = sortedUniqueWords(std::move(words));
    if (out.pattern.words.empty()) {
        error = "no words to search - builtin_dictionary is none, and the dictionaries have none";
        return false;
    }
    DictionarySpace space;
    if (!DictionarySpace::create(out.pattern, space, error))
        return false;

    out.prefix = normalizeMpqName(prefix);
    out.suffix = normalizeMpqName(suffix);
    out.bounds.hasLower = hasLower;
    out.bounds.lower = normalizeMpqName(lowerStr);
    out.bounds.hasUpper = hasUpper;
    out.bounds.upper = normalizeMpqName(upperStr);
    if (hasLower && hasUpper && out.bounds.upper < out.bounds.lower) {
        error = "lower_bound '" + lowerStr + "' sorts after upper_bound '" + upperStr + "'";
        return false;
    }

    out.outputFilePath = namedFilePath(config.matchesDir, "matches", matchesName);
    out.basenamesFilePath = namedFilePath(config.matchesDir, "basenames", matchesName);
    out.progressFilePath = namedFilePath(config.matchesDir, "wordnumber", matchesName);

    if (resume) {
        DictionaryProgress progress;
        bool exists = false;
        if (!readDictionaryProgress(out.progressFilePath, progress, exists, error))
            return false;
        if (!exists) {
            printf("resume_from_last_candidate: no progress file %s yet - starting from the beginning\n", out.progressFilePath.c_str());
        } else if (const std::vector<std::string> resumable = dictionaryResumableFingerprints(out);
                   std::find(resumable.begin(), resumable.end(), progress.fingerprint) == resumable.end()) {
            error = "resume_from_last_candidate: " + out.progressFilePath +
                    " is from a search with different settings (words, separators, word counts, prefix, suffix, bounds, hashes or "
                    "encryption key), or one that did less than this one (compared fewer candidates to the hashes - "
                    "record_hasha_matches - or recorded no basenames) - remove it, or set resume_from_last_candidate = false, to start "
                    "this one from the beginning";
            return false;
        } else {
            out.startNumber = progress.next;
            printf("Resuming from number %llu, from %s%s\n", (unsigned long long) progress.next, out.progressFilePath.c_str(),
                   progress.fingerprint == dictionaryFingerprint(out) ? "" : " (written by a search that did at least as much as this one)");
        }
    }
    return true;
}

std::string quoteDictionaryValue(const std::string& text) {
    if (text.empty() || std::isspace((unsigned char) text.front()) || std::isspace((unsigned char) text.back()) || unquote(text) != text)
        return "\"" + text + "\"";
    return text;
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
    const std::map<std::string, std::string>& existingSection = sectionName == "coordinator"        ? existing.coordinator
                                                               : sectionName == "dictionary"        ? existing.dictionary
                                                               : sectionName == "meet-in-the-middle" ? existing.meetInTheMiddle
                                                                                                     : existing.search;

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

bool updateConfigForMode(const std::string& path, const std::string& mode, const std::string& sectionName,
                          const std::vector<std::pair<std::string, std::string>>& keys, std::string& error) {
    std::ifstream probe(path);
    const bool fileExists = probe.good();
    probe.close();
    // Nothing to replace in a file that isn't there at all - writing one
    // from scratch is the same job either way.
    if (!fileExists)
        return ensureConfigForMode(path, mode, sectionName, keys, error);

    if (!setModeKey(path, mode, error))
        return false;
    return updateConfigSection(path, sectionName, keys, error);
}

bool updateConfigSection(const std::string& path, const std::string& sectionName, const std::vector<std::pair<std::string, std::string>>& keys,
                         std::string& error) {
    {
        // ensureSectionExists appends to the file, which needs to be there.
        std::ofstream create(path, std::ios::app);
        if (!create) {
            error = "cannot write " + path;
            return false;
        }
    }
    if (!ensureSectionExists(path, sectionName, error))
        return false;

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

    // The section's own lines: everything between its header and the next
    // section's (or the end of the file).
    size_t header = lines.size();
    for (size_t i = 0; i < lines.size(); ++i) {
        if (trim(lines[i]) == "[" + sectionName + "]") {
            header = i;
            break;
        }
    }
    if (header == lines.size()) { // ensureSectionExists just put one there
        error = "no [" + sectionName + "] section in " + path;
        return false;
    }
    size_t end = lines.size();
    for (size_t i = header + 1; i < lines.size(); ++i) {
        const std::string trimmed = trim(lines[i]);
        if (!trimmed.empty() && trimmed.front() == '[') {
            end = i;
            break;
        }
    }
    std::vector<std::string> body(lines.begin() + header + 1, lines.begin() + end);

    // Which of the section's own lines sets `key` - a commented-out one
    // doesn't count, and is left alone as the comment it is.
    auto lineSetting = [&body](const std::string& key) {
        for (size_t i = 0; i < body.size(); ++i) {
            const std::string trimmed = trim(body[i]);
            if (trimmed.empty() || trimmed.front() == '#')
                continue;
            const size_t eq = trimmed.find('=');
            if (eq != std::string::npos && trim(trimmed.substr(0, eq)) == key)
                return i;
        }
        return body.size();
    };

    for (const auto& kv : keys) {
        const size_t at = lineSetting(kv.first);
        if (kv.second.empty()) {
            if (at != body.size())
                body.erase(body.begin() + at);
            continue;
        }
        const std::string written = kv.first + " = " + quoteIfNeeded(sanitizeForConfigLine(kv.second));
        if (at != body.size()) {
            body[at] = written;
            continue;
        }
        // A key the section didn't have goes after its last line that holds
        // anything, so it joins the other keys instead of landing past the
        // blank line that separates this section from the next.
        size_t after = 0;
        for (size_t i = 0; i < body.size(); ++i)
            if (!trim(body[i]).empty())
                after = i + 1;
        body.insert(body.begin() + after, written);
    }

    std::vector<std::string> rewritten(lines.begin(), lines.begin() + header + 1);
    rewritten.insert(rewritten.end(), body.begin(), body.end());
    rewritten.insert(rewritten.end(), lines.begin() + end, lines.end());

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        error = "cannot rewrite " + path;
        return false;
    }
    for (const auto& l : rewritten)
        out << l << "\n";
    return true;
}

namespace {

// A [hardware] number that's `min` to `max` - false, with `error` set, if
// `value` isn't one.
bool parseHardwareNumber(const std::string& key, const std::string& value, int min, int max, int& out, std::string& error) {
    const std::string range = std::to_string(min) + " to " + std::to_string(max);
    if (value.empty() || value.size() > 4 || value.find_first_not_of("0123456789") != std::string::npos) {
        error = "invalid " + key + ": '" + value + "' (expected a number, " + range + ")";
        return false;
    }
    out = std::stoi(value);
    if (out < min || out > max) {
        error = key + " must be " + range + ", got " + value;
        return false;
    }
    return true;
}

// `gpus`: auto, all, none, or GPU numbers separated by commas.
bool parseGpus(const std::string& value, HardwareConfig& out, std::string& error) {
    out.gpus.clear();
    if (value == "auto" || value.empty()) {
        out.gpuChoice = HardwareConfig::GpuChoice::Auto;
        return true;
    }
    if (value == "all") {
        out.gpuChoice = HardwareConfig::GpuChoice::All;
        return true;
    }
    out.gpuChoice = HardwareConfig::GpuChoice::Listed;
    if (value == "none")
        return true;
    size_t at = 0;
    while (at <= value.size()) {
        const size_t comma = std::min(value.find(',', at), value.size());
        int number = 0;
        if (!parseHardwareNumber("gpus", trim(value.substr(at, comma - at)), 0, 63, number, error)) {
            error += " - gpus is auto, all, none, or GPU numbers as `namebreak --list-gpus` shows them, e.g. 0, 2";
            return false;
        }
        if (std::find(out.gpus.begin(), out.gpus.end(), number) == out.gpus.end())
            out.gpus.push_back(number);
        at = comma + 1;
    }
    return true;
}

// The GPU `key` (a `gpu<N>.<name>` key) is about, and the rest of it - false
// if it isn't one.
bool splitGpuKey(const std::string& key, int& gpu, std::string& rest) {
    const size_t dot = key.find('.');
    if (key.rfind("gpu", 0) != 0 || dot == std::string::npos || dot == 3 || dot > 5)
        return false;
    const std::string number = key.substr(3, dot - 3);
    if (number.find_first_not_of("0123456789") != std::string::npos)
        return false;
    gpu = std::stoi(number);
    rest = key.substr(dot + 1);
    return true;
}

} // namespace

bool buildHardwareConfig(const ConfigFile& config, HardwareConfig& out, std::string& error) {
    out = HardwareConfig();
    for (const auto& [key, value] : config.hardware) {
        int gpu = 0;
        std::string name = key;
        const bool ofOneGpu = splitGpuKey(key, gpu, name);
        if (!ofOneGpu && key == "gpus") {
            if (!parseGpus(value, out, error))
                break;
            continue;
        }
        if (!ofOneGpu && key == "cpu_threads") {
            if (!parseHardwareNumber(key, value, 0, 4096, out.cpuThreads, error))
                break;
            continue;
        }
        if (ofOneGpu && name == "name") {
            out.gpuNames[gpu] = value;
            continue;
        }
        const std::vector<GpuTuningKey>& keys = allGpuTuningKeys();
        const auto known = std::find_if(keys.begin(), keys.end(), [&](const GpuTuningKey& k) { return name == k.name; });
        if (known == keys.end()) {
            error = "unknown key '" + key + "'";
            break;
        }
        int number = 0;
        if (!parseHardwareNumber(key, value, known->min, known->max, number, error))
            break;
        GpuTuning& tuning = ofOneGpu ? out.gpuTunings[gpu] : out.tuning;
        tuning.*known->field = number;
    }
    if (!error.empty()) {
        error = "[hardware]: " + error;
        return false;
    }
    return true;
}
