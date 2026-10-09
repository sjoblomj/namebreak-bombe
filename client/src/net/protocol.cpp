// Minimal hand-rolled JSON encode/decode for exactly the flat request/
// response shapes in protocol.h - no general-purpose JSON library, since the
// whole wire surface this client needs is a handful of fixed, flat objects
// (string/number/bool/null fields and arrays of strings only, no nesting) -
// in keeping with the rest of this codebase (hand-rolled MPQ hashing,
// alphabet indexing, etc.) rather than pulling in an external dependency for
// this small a job.
#include "net/protocol.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>

namespace {

std::string escapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    out += '"';
    return out;
}

// One value out of a flat JSON object - only the shapes this protocol's
// responses actually use (string, number, true/false, null, and an array of
// strings - a dictionary claim's word lists and separators). No nested
// objects: none of the structs in protocol.h need them.
struct JsonValue {
    enum class Kind { String, Number, True, False, Null, Array } kind = Kind::Null;
    std::string text; // raw (already-unescaped) string, or the number's literal text
    std::vector<std::string> items; // an array's strings

    bool isNull() const { return kind == Kind::Null; }
    bool asBool() const { return kind == Kind::True; }
    int64_t asInt64() const { return kind == Kind::Number ? std::stoll(text) : 0; }
    double asDouble() const { return kind == Kind::Number ? std::stod(text) : 0.0; }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    // Parses a top-level `{...}` object whose values are all flat (string/
    // number/bool/null) into `out`. Returns false on any malformed input.
    bool parseFlatObject(std::map<std::string, JsonValue>& out) {
        skipWs();
        if (!consume('{')) return false;
        skipWs();
        if (consume('}')) return true; // empty object
        while (true) {
            skipWs();
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (!consume(':')) return false;
            skipWs();
            JsonValue value;
            if (!parseValue(value)) return false;
            out[key] = value;
            skipWs();
            if (consume(',')) continue;
            if (consume('}')) break;
            return false;
        }
        return true;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    void skipWs() {
        while (i_ < s_.size() && std::isspace((unsigned char) s_[i_])) i_++;
    }

    bool consume(char c) {
        if (i_ < s_.size() && s_[i_] == c) {
            i_++;
            return true;
        }
        return false;
    }

    bool consumeLiteral(const char* lit) {
        size_t n = strlen(lit);
        if (s_.compare(i_, n, lit) == 0) {
            i_ += n;
            return true;
        }
        return false;
    }

    bool parseString(std::string& out) {
        if (!consume('"')) return false;
        out.clear();
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c == '\\') {
                if (i_ >= s_.size()) return false;
                char esc = s_[i_++];
                switch (esc) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u': {
                        // Only handles the BMP (\uXXXX -> UTF-8), which is all
                        // any of this protocol's actual field values need
                        // (usernames/hostnames/filenames); no surrogate-pair
                        // decoding.
                        if (i_ + 4 > s_.size()) return false;
                        unsigned code = (unsigned) std::stoul(s_.substr(i_, 4), nullptr, 16);
                        i_ += 4;
                        if (code < 0x80) {
                            out += (char) code;
                        } else if (code < 0x800) {
                            out += (char) (0xC0 | (code >> 6));
                            out += (char) (0x80 | (code & 0x3F));
                        } else {
                            out += (char) (0xE0 | (code >> 12));
                            out += (char) (0x80 | ((code >> 6) & 0x3F));
                            out += (char) (0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: return false;
                }
            } else {
                out += c;
            }
        }
        return consume('"');
    }

    bool parseValue(JsonValue& out) {
        if (i_ >= s_.size()) return false;
        char c = s_[i_];
        if (c == '"') {
            out.kind = JsonValue::Kind::String;
            return parseString(out.text);
        }
        if (consume('[')) {
            out.kind = JsonValue::Kind::Array;
            skipWs();
            if (consume(']'))
                return true;
            while (true) {
                skipWs();
                std::string item;
                if (!parseString(item)) return false; // only arrays of strings
                out.items.push_back(std::move(item));
                skipWs();
                if (consume(',')) continue;
                return consume(']');
            }
        }
        if (consumeLiteral("true"))  { out.kind = JsonValue::Kind::True;  return true; }
        if (consumeLiteral("false")) { out.kind = JsonValue::Kind::False; return true; }
        if (consumeLiteral("null"))  { out.kind = JsonValue::Kind::Null;  return true; }
        if (c == '-' || std::isdigit((unsigned char) c)) {
            size_t start = i_;
            if (c == '-') i_++;
            while (i_ < s_.size() && (std::isdigit((unsigned char) s_[i_]) || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) i_++;
            if (i_ == start) return false;
            out.kind = JsonValue::Kind::Number;
            out.text = s_.substr(start, i_ - start);
            return true;
        }
        return false; // an object value - not needed by any struct here
    }
};

bool getString(const std::map<std::string, JsonValue>& obj, const std::string& key, std::string& out) {
    auto it = obj.find(key);
    if (it == obj.end() || it->second.kind != JsonValue::Kind::String) return false;
    out = it->second.text;
    return true;
}

bool getInt64(const std::map<std::string, JsonValue>& obj, const std::string& key, int64_t& out) {
    auto it = obj.find(key);
    if (it == obj.end() || it->second.kind != JsonValue::Kind::Number) return false;
    out = it->second.asInt64();
    return true;
}

bool getBool(const std::map<std::string, JsonValue>& obj, const std::string& key, bool& out) {
    auto it = obj.find(key);
    if (it == obj.end() || (it->second.kind != JsonValue::Kind::True && it->second.kind != JsonValue::Kind::False)) return false;
    out = it->second.asBool();
    return true;
}

// Like getBool, but a missing key leaves `out` untouched instead of failing.
bool getOptionalBool(const std::map<std::string, JsonValue>& obj, const std::string& key, bool& out) {
    return obj.find(key) == obj.end() || getBool(obj, key, out);
}

// Like getInt64, but a missing key leaves `out` untouched instead of failing.
bool getOptionalInt64(const std::map<std::string, JsonValue>& obj, const std::string& key, int64_t& out) {
    return obj.find(key) == obj.end() || getInt64(obj, key, out);
}

// Like getString, but a missing key (or null) leaves `out` untouched.
bool getOptionalString(const std::map<std::string, JsonValue>& obj, const std::string& key, std::string& out) {
    auto it = obj.find(key);
    return it == obj.end() || it->second.isNull() || getString(obj, key, out);
}

// Like getOptionalString, into an optional: empty if missing or null.
bool getOptionalString(const std::map<std::string, JsonValue>& obj, const std::string& key, std::optional<std::string>& out) {
    out.reset();
    auto it = obj.find(key);
    if (it == obj.end() || it->second.isNull())
        return true;
    std::string value;
    if (!getString(obj, key, value))
        return false;
    out = value;
    return true;
}

// An array of strings - false if missing, or anything else.
bool getStringArray(const std::map<std::string, JsonValue>& obj, const std::string& key, std::vector<std::string>& out) {
    auto it = obj.find(key);
    if (it == obj.end() || it->second.kind != JsonValue::Kind::Array) return false;
    out = it->second.items;
    return true;
}

// Like getStringArray, but a missing key (or null) is no items.
bool getOptionalStringArray(const std::map<std::string, JsonValue>& obj, const std::string& key, std::vector<std::string>& out) {
    out.clear();
    auto it = obj.find(key);
    return it == obj.end() || it->second.isNull() || getStringArray(obj, key, out);
}

// A dictionary claim's fields (see ClaimResponse::dictionary) - every one
// of them needed once `dictionary` is true, none of them before.
bool getDictionaryClaim(const std::map<std::string, JsonValue>& obj, ClaimResponse& out) {
    if (!getOptionalBool(obj, "dictionary", out.dictionary)) return false;
    if (!out.dictionary) return true;
    return getStringArray(obj, "word_lists", out.wordLists) &&
           getStringArray(obj, "word_list_checksums", out.wordListChecksums) &&
           out.wordListChecksums.size() == out.wordLists.size() &&
           getString(obj, "words_checksum", out.wordsChecksum) &&
           getStringArray(obj, "separators", out.separators) &&
           getInt64(obj, "min_words", out.minWords) &&
           getInt64(obj, "max_words", out.maxWords) &&
           getOptionalStringArray(obj, "tails", out.tails) &&
           getOptionalString(obj, "tails_checksum", out.tailsChecksum) && out.tails.empty() == out.tailsChecksum.empty() &&
           getInt64(obj, "first_candidate_number", out.firstCandidateNumber) &&
           getInt64(obj, "end_candidate_number", out.endCandidateNumber) &&
           out.firstCandidateNumber >= 0 && out.endCandidateNumber >= out.firstCandidateNumber &&
           getOptionalString(obj, "filename_lower_bound", out.filenameLowerBound) &&
           getOptionalString(obj, "filename_upper_bound", out.filenameUpperBound) &&
           getOptionalBool(obj, "send_basenames", out.sendBasenames) &&
           getOptionalString(obj, "encryption_key_hex", out.encryptionKeyHex) &&
           (!out.sendBasenames || !out.encryptionKeyHex.empty());
}

// A JSON array of `items`.
std::string jsonStringArray(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += ",";
        out += escapeJsonString(items[i]);
    }
    return out + "]";
}

// The fields a heartbeat, quit and completion share beyond the first:
// `next_candidate_number` if there is one, and `basenames` if any - left out
// otherwise, so that a report is what it was before protocol 1.5.
std::string progressAndBasenames(const std::optional<int64_t>& nextCandidateNumber, const std::vector<std::string>& basenames) {
    std::string out;
    if (nextCandidateNumber)
        out += ",\"next_candidate_number\":" + std::to_string(*nextCandidateNumber);
    if (!basenames.empty())
        out += ",\"basenames\":" + jsonStringArray(basenames);
    return out;
}

// An insertion, as `name`_text and `name`_position - nothing inserted if
// the text is missing or empty.
bool getOptionalInsertion(const std::map<std::string, JsonValue>& obj, const std::string& name, Insertion& out) {
    out = Insertion();
    auto text = obj.find(name + "_text");
    if (text == obj.end() || text->second.isNull())
        return true;
    int64_t position = 0;
    if (!getString(obj, name + "_text", out.text) || !getOptionalInt64(obj, name + "_position", position) || position < 0 || position > 9999)
        return false;
    out.position = (int) position;
    return true;
}

} // namespace

std::string toJson(const RegisterRequest& req) {
    return "{\"username\":" + escapeJsonString(req.username) + ",\"hostname\":" + escapeJsonString(req.hostname) +
           ",\"backend\":" + escapeJsonString(req.backend) + ",\"protocol_version\":" + escapeJsonString(req.protocolVersion) +
           ",\"client_release\":" + escapeJsonString(req.clientRelease) + "}";
}

std::string toJson(const HeartbeatRequest& req) {
    std::string out = "{\"last_hash_a_match_filename\":";
    out += req.lastHashAMatchFilename ? escapeJsonString(*req.lastHashAMatchFilename) : "null";
    out += progressAndBasenames(req.nextCandidateNumber, req.basenames);
    out += "}";
    return out;
}

std::string toJson(const QuitRequest& req) {
    std::string out = "{\"last_hash_a_match_filename\":";
    out += req.lastHashAMatchFilename ? escapeJsonString(*req.lastHashAMatchFilename) : "null";
    out += progressAndBasenames(req.nextCandidateNumber, req.basenames);
    out += "}";
    return out;
}

std::string toJson(const CompleteRequest& req) {
    std::string out = "{";
    out += "\"found\":";
    out += req.found ? "true" : "false";
    out += ",\"filename\":";
    out += req.filename ? escapeJsonString(*req.filename) : "null";
    out += ",\"elapsed_seconds\":" + std::to_string(req.elapsedSeconds);
    out += ",\"candidates_processed\":" + std::to_string(req.candidatesProcessed);
    out += progressAndBasenames(std::nullopt, req.basenames);
    out += "}";
    return out;
}

bool parseRegisterResponse(const std::string& body, RegisterResponse& out) {
    std::map<std::string, JsonValue> obj;
    if (!JsonParser(body).parseFlatObject(obj)) return false;
    return getInt64(obj, "user_id", out.userId) && getString(obj, "token", out.token) &&
           getString(obj, "server_protocol_version", out.serverProtocolVersion);
}

bool parseClaimResponse(const std::string& body, ClaimResponse& out) {
    std::map<std::string, JsonValue> obj;
    if (!JsonParser(body).parseFlatObject(obj)) return false;
    return getInt64( obj, "range_id", out.rangeId) &&
           getInt64( obj, "target_id", out.targetId) &&
           getString(obj, "target_name", out.targetName) &&
           getString(obj, "prefix", out.prefix) &&
           getString(obj, "suffix", out.suffix) &&
           getString(obj, "hash_a_hex", out.hashAHex) &&
           getString(obj, "hash_b_hex", out.hashBHex) &&
           getBool(  obj, "prune_symbol_runs", out.pruneSymbolRuns) &&
           getOptionalBool(obj, "prune_unopened_brackets", out.pruneUnopenedBrackets) &&
           getOptionalBool(obj, "prune_whole_candidate", out.pruneWholeCandidate) &&
           getInt64( obj, "max_backslash_count", out.maxBackslashCount) &&
           getOptionalInt64(obj, "min_backslash_count", out.minBackslashCount) &&
           getOptionalBool(obj, "prune_adjacent_backslashes", out.pruneAdjacentBackslashes) &&
           getOptionalInsertion(obj, "insert_from_start", out.insertFromStart) &&
           getOptionalInsertion(obj, "insert_from_end", out.insertFromEnd) &&
           getString(obj, "lower_bound_filename", out.lowerBoundFilename) &&
           getString(obj, "upper_bound_filename", out.upperBoundFilename) &&
           getString(obj, "alphabet", out.alphabet) &&
           getInt64( obj, "candidate_count", out.candidateCount) &&
           getInt64( obj, "lease_seconds", out.leaseSeconds) &&
           getDictionaryClaim(obj, out);
}

bool parseHeartbeatResponse(const std::string& body, HeartbeatResponse& out) {
    std::map<std::string, JsonValue> obj;
    if (!JsonParser(body).parseFlatObject(obj)) return false;
    return getInt64(obj, "lease_seconds", out.leaseSeconds) && getBool(obj, "range_released", out.rangeReleased);
}

std::string parseErrorMessage(const std::string& body) {
    std::map<std::string, JsonValue> obj;
    if (!JsonParser(body).parseFlatObject(obj)) return "";
    std::string msg;
    getString(obj, "error", msg);
    return msg;
}
