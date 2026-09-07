#include "coordinator_client.h"

namespace {
std::string authHeader(const std::string& token) {
    return "Authorization: Bearer " + token;
}
} // namespace

bool CoordinatorClient::registerClient(const std::string& username, const std::string& hostname, int64_t& outUserId, std::string& error) {
    RegisterRequest req{username, hostname};
    HttpResponse resp = http_.post(baseUrl_ + "/api/v1/register", {}, toJson(req));
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return false;
    }
    if (!resp.ok()) {
        error = "register failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return false;
    }
    RegisterResponse out;
    if (!parseRegisterResponse(resp.body, out)) {
        error = "malformed register response: " + resp.body;
        return false;
    }
    token_ = out.token;
    outUserId = out.userId;
    return true;
}

bool CoordinatorClient::claim(std::optional<ClaimResponse>& out, std::string& error) {
    HttpResponse resp = http_.post(baseUrl_ + "/api/v1/claim", {authHeader(token_)}, "");
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
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

bool CoordinatorClient::heartbeat(int64_t rangeId, const std::optional<std::string>& lastHashAMatchFilename, HeartbeatResponse& out, std::string& error) {
    HeartbeatRequest req{lastHashAMatchFilename};
    std::string url = baseUrl_ + "/api/v1/ranges/" + std::to_string(rangeId) + "/heartbeat";
    HttpResponse resp = http_.post(url, {authHeader(token_)}, toJson(req));
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return false;
    }
    if (!resp.ok()) {
        error = "heartbeat failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return false;
    }
    if (!parseHeartbeatResponse(resp.body, out)) {
        error = "malformed heartbeat response: " + resp.body;
        return false;
    }
    return true;
}

CoordinatorClient::CompleteOutcome CoordinatorClient::complete(int64_t rangeId, const CompleteRequest& req, std::string& error) {
    std::string url = baseUrl_ + "/api/v1/ranges/" + std::to_string(rangeId) + "/complete";
    HttpResponse resp = http_.post(url, {authHeader(token_)}, toJson(req));
    if (resp.status == 0) {
        error = "request failed: " + resp.error;
        return CompleteOutcome::Error;
    }
    if (resp.status == 409) {
        return CompleteOutcome::Conflict;
    }
    if (!resp.ok()) {
        error = "complete failed (HTTP " + std::to_string(resp.status) + "): " + parseErrorMessage(resp.body);
        return CompleteOutcome::Error;
    }
    return CompleteOutcome::Ok;
}
