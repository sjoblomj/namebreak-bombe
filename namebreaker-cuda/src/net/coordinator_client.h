#ifndef NAMEBREAK_CUDA_COORDINATOR_CLIENT_H
#define NAMEBREAK_CUDA_COORDINATOR_CLIENT_H

#include <cstdint>
#include <optional>
#include <string>

#include "net/http_client.h"
#include "net/protocol.h"

// Talks to the coordinator server's HTTP API
// Not thread-safe: each thread that needs to make requests (the
// coordinator loop and its heartbeat thread both do) should use its
// own instance, sharing the token via `setToken`/`token`.
class CoordinatorClient {
public:
    explicit CoordinatorClient(std::string baseUrl) : baseUrl_(std::move(baseUrl)) {}

    // Registers with the server, sending this client's own protocolVersion
    // (see protocol.h's kProtocolVersion) so the server can gate what it
    // offers to only what this client understands. On success, stores the
    // returned token for subsequent calls (also retrievable via token())
    // and returns true, with outServerProtocolVersion set to the server's
    // own protocol version - purely informational, worth logging.
    bool registerClient(const std::string& username, const std::string& hostname, int64_t& outUserId, std::string& outServerProtocolVersion, std::string& error);

    void setToken(std::string token) { token_ = std::move(token); }
    const std::string& token() const { return token_; }

    // std::nullopt means "no work available" (server returned 204 No
    // Content) - not an error.
    bool claim(std::optional<ClaimResponse>& out, std::string& error);

    // Distinguishes a 409 (this range's ownership already moved on - e.g. a
    // network or sleep/hibernate outage during heartbeating outlasted the
    // lease and the server already released or reassigned it) from other
    // failures, mirroring api.rs's `err.status() == CONFLICT` check. On
    // Conflict, `out`/`error` are unset - there's nothing more to read, the
    // caller already knows what happened.
    enum class HeartbeatOutcome { Ok, Conflict, Error };
    HeartbeatOutcome heartbeat(int64_t rangeId, const std::optional<std::string>& lastHashAMatchFilename, HeartbeatResponse& out, std::string& error);

    // Distinguishes a 409 (this range's ownership already moved on - e.g. a
    // network outage during heartbeating outlasted the lease and the server
    // already reassigned it) from other failures, mirroring api.rs's
    // `err.status() == CONFLICT` check.
    enum class CompleteOutcome { Ok, Conflict, Error };
    CompleteOutcome complete(int64_t rangeId, const CompleteRequest& req, std::string& error);

private:
    std::string baseUrl_;
    std::string token_;
    HttpClient http_;
};

#endif // NAMEBREAK_CUDA_COORDINATOR_CLIENT_H
