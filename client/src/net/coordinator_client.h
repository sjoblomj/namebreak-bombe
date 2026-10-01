#ifndef NAMEBREAK_NET_COORDINATOR_CLIENT_H
#define NAMEBREAK_NET_COORDINATOR_CLIENT_H

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
    // offers to only what this client understands, its release (see
    // common/version.h) and the name of the backend it searches with. On
    // success, stores the returned token for subsequent calls (also
    // retrievable via token()) and returns true, with `out` holding the
    // server's response - notably its protocol version (purely
    // informational, worth logging).
    bool registerClient(const std::string& username, const std::string& hostname, const std::string& backend, RegisterResponse& out,
                        std::string& error);

    void setToken(std::string token) { token_ = std::move(token); }
    const std::string& token() const { return token_; }

    // std::nullopt means "no work available" (server returned 204 No
    // Content) - not an error.
    bool claim(std::optional<ClaimResponse>& out, std::string& error);

    // Whether the last registerClient() or claim() failed because the
    // server refuses this release as too old (HTTP 426 Upgrade Required) -
    // retrying can't help, so the caller should show `error` (the server's
    // message, saying what to get) and quit.
    bool upgradeRequired() const { return upgradeRequired_; }

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
    // `err.status() == CONFLICT` check. Failures are further split into
    // TransientError (the request never got a real answer - transport
    // failure, 5xx, 429 - so sending it again may well succeed) and Error
    // (any other rejection, which resending would only repeat).
    enum class CompleteOutcome { Ok, Conflict, TransientError, Error };
    CompleteOutcome complete(int64_t rangeId, const CompleteRequest& req, std::string& error);

    // Tells the server this client is quitting with range `rangeId` in hand,
    // and how far it got (see QuitRequest). Conflict (409) means the range
    // wasn't ours any more anyway - see heartbeat().
    enum class QuitOutcome { Ok, Conflict, Error };
    QuitOutcome quit(int64_t rangeId, const std::optional<std::string>& lastHashAMatchFilename, std::string& error);

private:
    std::string baseUrl_;
    std::string token_;
    bool upgradeRequired_ = false;
    HttpClient http_;
};

#endif // NAMEBREAK_NET_COORDINATOR_CLIENT_H
