#include "net/http_client.h"

#include <curl/curl.h>

namespace {
size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}
} // namespace

HttpClient::HttpClient() {
    curl_ = curl_easy_init();
}

HttpClient::~HttpClient() {
    if (curl_) curl_easy_cleanup(static_cast<CURL*>(curl_));
}

HttpResponse HttpClient::post(const std::string& url, const std::vector<std::string>& headers, const std::string& jsonBody) {
    HttpResponse resp;
    auto* curl = static_cast<CURL*>(curl_);
    if (!curl) {
        resp.error = "failed to initialize libcurl";
        return resp;
    }

    curl_slist* headerList = nullptr;
    headerList = curl_slist_append(headerList, "Content-Type: application/json");
    for (const auto& h : headers) headerList = curl_slist_append(headerList, h.c_str());

    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonBody.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long) jsonBody.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, kDefaultTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    // curl_easy_reset() (above) restores every *option* to its default, but
    // it does NOT touch this handle's cached connection - by default curl
    // keeps a request's TCP connection open and reuses it on the handle's
    // next perform(). If a network glitch (interface drop, VPN reconnect,
    // NAT rebind) kills that connection without a clean FIN/RST, the cached
    // socket looks fine to curl but is actually dead: every subsequent
    // request silently tries to reuse it, hangs until CURLOPT_TIMEOUT, and
    // fails again - forever, even once connectivity is back - since nothing
    // ever prompts curl to open a fresh connection. A CoordinatorClient (and
    // the CURL handle it owns) lives for an entire range's worth of
    // heartbeats, or the whole process for claim/register, so this isn't a
    // one-off: one glitch permanently wedges every future request on that
    // handle. Forcing the connection closed after every transfer trades a
    // fresh TCP(+TLS) handshake per request - negligible next to a 60s
    // heartbeat/30s poll cadence - for never getting stuck on a dead one.
    curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1L);

    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headerList);

    if (rc != CURLE_OK) {
        resp.error = curl_easy_strerror(rc);
        return resp;
    }

    long httpStatus = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpStatus);
    resp.status = httpStatus;
    return resp;
}

HttpResponse HttpClient::get(const std::string& url, const std::vector<std::string>& headers, long timeoutSeconds) {
    HttpResponse resp;
    auto* curl = static_cast<CURL*>(curl_);
    if (!curl) {
        resp.error = "failed to initialize libcurl";
        return resp;
    }

    curl_slist* headerList = nullptr;
    for (const auto& h : headers) headerList = curl_slist_append(headerList, h.c_str());

    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1L); // see post()
    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headerList);

    if (rc != CURLE_OK) {
        resp.error = curl_easy_strerror(rc);
        return resp;
    }

    long httpStatus = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpStatus);
    resp.status = httpStatus;
    return resp;
}

HttpResponse HttpClient::head(const std::string& url, long timeoutSeconds) {
    HttpResponse resp;
    auto* curl = static_cast<CURL*>(curl_);
    if (!curl) {
        resp.error = "failed to initialize libcurl";
        return resp;
    }

    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "namebreak");
    curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1L); // see post()

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        resp.error = curl_easy_strerror(rc);
        return resp;
    }

    long httpStatus = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpStatus);
    resp.status = httpStatus;
    char* effective = nullptr;
    if (curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &effective) == CURLE_OK && effective)
        resp.effectiveUrl = effective;
    return resp;
}
