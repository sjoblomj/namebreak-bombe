#ifndef NAMEBREAK_CUDA_PROTOCOL_H
#define NAMEBREAK_CUDA_PROTOCOL_H

#include <cstdint>
#include <optional>
#include <string>

// Wire types for the coordinator's HTTP API - field-for-field the same shape
// as coordinator/protocol/src/lib.rs, since this talks to that same,
// unmodified Rust server. Only the request/response shapes the client
// actually uses are included (not the admin/status/alphabets endpoints).

struct RegisterRequest {
    std::string username;
    std::string hostname;
};

struct RegisterResponse {
    int64_t userId = 0;
    std::string token;
};

// A contiguous, ready-to-run slice of one target's search space.
// lowerBoundFilename/upperBoundFilename are both inclusive, full filenames
// (prefix+candidate+suffix) - directly usable as SearchRequest's
// startCandidate/lowerBound/upperBound once stripped of prefix/suffix (see
// remove_prefix_and_suffix in cpu-utils.h).
struct ClaimResponse {
    int64_t rangeId = 0;
    int64_t targetId = 0;
    std::string targetName;
    std::string prefix;
    std::string suffix;
    std::string hashAHex;
    std::string hashBHex;
    bool pruneSymbolRuns = false;
    int64_t maxBackslashCount = 0;
    std::string lowerBoundFilename;
    std::string upperBoundFilename;
    std::string alphabet;
    int64_t candidateCount = 0;
    int64_t leaseSeconds = 0;
};

struct HeartbeatRequest {
    // The most recent Hash-A-only match's full filename for the range this
    // heartbeat is for, if any - see runSearch's onPartialMatch callback.
    std::optional<std::string> lastHashAMatchFilename;
};

struct HeartbeatResponse {
    int64_t leaseSeconds = 0;
    // True once this range's target has been solved via a different range -
    // the caller should abort its current search rather than keep going.
    bool targetSolved = false;
};

struct CompleteRequest {
    bool found = false;
    std::optional<std::string> filename;
    double elapsedSeconds = 0;
    int64_t candidatesProcessed = 0;
};

std::string toJson(const RegisterRequest&  req);
std::string toJson(const HeartbeatRequest& req);
std::string toJson(const CompleteRequest&  req);

// Each returns false (contents of `out` unspecified) if `body` isn't valid
// JSON or is missing a required field - callers should treat that as a
// malformed/unexpected server response.
bool parseRegisterResponse (const std::string& body, RegisterResponse&  out);
bool parseClaimResponse    (const std::string& body, ClaimResponse&     out);
bool parseHeartbeatResponse(const std::string& body, HeartbeatResponse& out);

// Best-effort extraction of {"error": "..."} from a server error body (see
// server/src/error.rs) - empty string if the body doesn't parse or has no
// such field.
std::string parseErrorMessage(const std::string& body);

#endif // NAMEBREAK_CUDA_PROTOCOL_H
