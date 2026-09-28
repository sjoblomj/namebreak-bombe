#ifndef NAMEBREAK_NET_HTTP_CLIENT_H
#define NAMEBREAK_NET_HTTP_CLIENT_H

#include <string>
#include <vector>

// Thin libcurl wrapper - generic HTTP, no knowledge of the coordinator's
// protocol (see coordinator_client.h for that). Kept separate so protocol.*
// stays testable without ever touching the network.
struct HttpResponse {
    // 0 means the request itself failed (DNS/connect/timeout/TLS - see
    // `error`), not a fetched HTTP status. Any 0-599 status curl reports is
    // otherwise passed straight through, including 4xx/5xx.
    long status = 0;
    std::string body;
    // libcurl's error string, only set when status == 0.
    std::string error;
    // The URL the response actually came from, after following redirects -
    // only set by head().
    std::string effectiveUrl;

    bool ok() const { return status >= 200 && status < 300; }
};

class HttpClient {
public:
    HttpClient();
    ~HttpClient();
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // Well under the coordinator client's 60s heartbeat interval, so a
    // request that hangs (rather than failing fast) can't stall the
    // heartbeat loop past its next scheduled tick.
    static constexpr long kDefaultTimeoutSeconds = 20;

    HttpResponse post(const std::string& url, const std::vector<std::string>& headers, const std::string& jsonBody);

    // A body-less GET, following redirects - for where a URL redirects to
    // (see HttpResponse::effectiveUrl), as update_check.cpp needs.
    HttpResponse head(const std::string& url, long timeoutSeconds = kDefaultTimeoutSeconds);

private:
    void* curl_; // CURL*, opaque here so this header doesn't need <curl/curl.h>
};

#endif // NAMEBREAK_NET_HTTP_CLIENT_H
