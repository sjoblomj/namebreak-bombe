#include "net/coordinator_client.h"

#include "common/version.h"

namespace {
std::string authHeader(const std::string& token) {
    return "Authorization: Bearer " + token;
}
} // namespace

bool CoordinatorClient::registerClient(const std::string& username, const std::string& hostname, const std::string& backend, RegisterResponse& out,
                                       std::string& error) {
    RegisterRequest req;
    req.username = username;
    req.hostname = hostname;
    req.backend = backend;
    req.clientRelease = namebreakVersion();
    HttpResponse resp = http_.post(baseUrl_ + "/api/v1/register", {}, toJson(req));
    upgradeRequired_ = resp.status == 426;
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return false;
    }
    if (upgradeRequired_) {
        error = parseErrorMessage(resp.body);
        return false;
    }
    if (!resp.ok()) {
        // A protocol version mismatch (see protocol.h's kProtocolVersion)
        // surfaces here as an ordinary 400 - the server's own message
        // (already in parseErrorMessage's output) explains why.
        error = "register failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return false;
    }
    if (!parseRegisterResponse(resp.body, out)) {
        error = "malformed register response: " + resp.body;
        return false;
    }
    token_ = out.token;
    return true;
}

bool CoordinatorClient::claim(std::optional<ClaimResponse>& out, std::string& error) {
    HttpResponse resp = http_.post(baseUrl_ + "/api/v1/claim", {authHeader(token_)}, "");
    upgradeRequired_ = resp.status == 426;
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return false;
    }
    if (upgradeRequired_) {
        error = parseErrorMessage(resp.body);
        return false;
    }
    if (resp.status == 204) {
        out = std::nullopt;
        return true;
    }
    if (!resp.ok()) {
        error = "claim failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return false;
    }
    ClaimResponse claimed;
    if (!parseClaimResponse(resp.body, claimed)) {
        error = "malformed claim response: " + resp.body;
        return false;
    }
    out = claimed;
    return true;
}

CoordinatorClient::HeartbeatOutcome CoordinatorClient::heartbeat(int64_t rangeId, const std::optional<std::string>& lastHashAMatchFilename, HeartbeatResponse& out, std::string& error) {
    HeartbeatRequest req{lastHashAMatchFilename};
    std::string url = baseUrl_ + "/api/v1/ranges/" + std::to_string(rangeId) + "/heartbeat";
    HttpResponse resp = http_.post(url, {authHeader(token_)}, toJson(req));
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return HeartbeatOutcome::Error;
    }
    if (resp.status == 409) {
        return HeartbeatOutcome::Conflict;
    }
    if (!resp.ok()) {
        error = "heartbeat failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return HeartbeatOutcome::Error;
    }
    if (!parseHeartbeatResponse(resp.body, out)) {
        error = "malformed heartbeat response: " + resp.body;
        return HeartbeatOutcome::Error;
    }
    return HeartbeatOutcome::Ok;
}

CoordinatorClient::CompleteOutcome CoordinatorClient::complete(int64_t rangeId, const CompleteRequest& req, std::string& error) {
    std::string url = baseUrl_ + "/api/v1/ranges/" + std::to_string(rangeId) + "/complete";
    HttpResponse resp = http_.post(url, {authHeader(token_)}, toJson(req));
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return CompleteOutcome::TransientError;
    }
    if (resp.status == 409) {
        return CompleteOutcome::Conflict;
    }
    if (!resp.ok()) {
        error = "complete failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return (resp.status >= 500 || resp.status == 429) ? CompleteOutcome::TransientError : CompleteOutcome::Error;
    }
    return CompleteOutcome::Ok;
}
