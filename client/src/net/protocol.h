#ifndef NAMEBREAK_NET_PROTOCOL_H
#define NAMEBREAK_NET_PROTOCOL_H

#include <cstdint>
#include <optional>
#include <string>

// Wire types for the coordinator's HTTP API - field-for-field the same shape
// as coordinator/protocol/src/lib.rs, since this talks to that same,
// unmodified Rust server. Only the request/response shapes the client
// actually uses are included (not the admin/status/alphabets endpoints).

// This client's own protocol version - see coordinator/protocol/src/lib.rs's
// PROTOCOL_VERSION for what MAJOR/MINOR/PATCH each mean for this pair. Sent
// with every RegisterRequest so the server can gate what it offers this
// client to only what a version this old can actually make sense of (e.g. a
// predefined alphabet introduced in a MINOR version newer than this one
// simply never gets handed to it - see ranges::claim_range on the server).
// Bump this whenever this client starts depending on something the protocol
// only guarantees from a newer version onward.
constexpr const char* kProtocolVersion = "1.0.0";

struct RegisterRequest {
    std::string username;
    std::string hostname;
    // The backend this client searches with (SearchBackend::name(), e.g.
    // "cuda") - informational only; the server just records it.
    std::string backend;
    std::string protocolVersion = kProtocolVersion;
};

struct RegisterResponse {
    int64_t userId = 0;
    std::string token;
    // The server's own protocol version - purely informational. A MAJOR
    // version mismatch is already rejected by the server before a response
    // like this one is ever returned (see parseErrorMessage's use in
    // CoordinatorClient::registerClient), so a successful registration
    // implies MAJOR already matches; this is just worth logging so it's
    // visible which MINOR feature set (e.g. which predefined alphabets) the
    // server might use that this client predates.
    std::string serverProtocolVersion;
};

// A contiguous, ready-to-run slice of one target's search space.
// lowerBoundFilename/upperBoundFilename are both inclusive, full filenames
// (prefix+candidate+suffix) - directly usable as SearchRequest's
// startCandidate/lowerBound/upperBound once stripped of prefix/suffix (see
// removePrefixAndSuffix in candidate.h).
struct ClaimResponse {
    int64_t rangeId = 0;
    int64_t targetId = 0;
    std::string targetName;
    std::string prefix;
    std::string suffix;
    std::string hashAHex;
    std::string hashBHex;
    bool pruneSymbolRuns = false;
    // Optional in the response (false when absent).
    bool pruneUnopenedBrackets = false;
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
    // Also doubles as this client's only liveness-of-*progress* signal (as
    // opposed to liveness of the heartbeat itself, which every call already
    // proves) - reporting the same filename heartbeat after heartbeat, as a
    // paused client necessarily would, is what lets the server notice and
    // eventually release a stalled range (see HeartbeatResponse::rangeReleased).
    std::optional<std::string> lastHashAMatchFilename;
};

struct HeartbeatResponse {
    int64_t leaseSeconds = 0;
    // True once this range should be abandoned - either its target was
    // solved (by this range or a different one), or the range went too long
    // without any reported progress and the server released it back to
    // pending for someone else. Either way the caller should abort its
    // current search and won't be reporting completion for this range; the
    // two reasons need no distinguishing here - a caller that's locally
    // paused already knows to stay paused and idle rather than claim a new
    // range regardless of which one this was (see runCoordinator's claim loop).
    bool rangeReleased = false;
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

#endif // NAMEBREAK_NET_PROTOCOL_H
