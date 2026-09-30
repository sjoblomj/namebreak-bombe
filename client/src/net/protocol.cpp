// Minimal hand-rolled JSON encode/decode for exactly the flat request/
// response shapes in protocol.h - no general-purpose JSON library, since the
// whole wire surface this client needs is a handful of fixed, flat objects
// (string/number/bool/null fields only, no nesting) - in keeping with the
// rest of this codebase (hand-rolled MPQ hashing, alphabet indexing, etc.)
// rather than pulling in an external dependency for this small a job.
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
// responses actually use (string, number, true/false, null). No arrays or
// nested objects: none of the structs in protocol.h need them.
struct JsonValue {
    enum class Kind { String, Number, True, False, Null } kind = Kind::Null;
    std::string text; // raw (already-unescaped) string, or the number's literal text

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
        return false; // an object/array value - not needed by any struct here
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

} // namespace

std::string toJson(const RegisterRequest& req) {
    return "{\"username\":" + escapeJsonString(req.username) + ",\"hostname\":" + escapeJsonString(req.hostname) +
           ",\"backend\":" + escapeJsonString(req.backend) + ",\"protocol_version\":" + escapeJsonString(req.protocolVersion) +
           ",\"client_release\":" + escapeJsonString(req.clientRelease) + "}";
}

std::string toJson(const HeartbeatRequest& req) {
    std::string out = "{\"last_hash_a_match_filename\":";
    out += req.lastHashAMatchFilename ? escapeJsonString(*req.lastHashAMatchFilename) : "null";
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
           getString(obj, "lower_bound_filename", out.lowerBoundFilename) &&
           getString(obj, "upper_bound_filename", out.upperBoundFilename) &&
           getString(obj, "alphabet", out.alphabet) &&
           getInt64( obj, "candidate_count", out.candidateCount) &&
           getInt64( obj, "lease_seconds", out.leaseSeconds);
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
