#include "net/update_check.h"

#include "common/version.h"
#include "net/http_client.h"

namespace {
// Short, since it runs at startup before any work starts, and is only ever
// a nicety.
constexpr long kTimeoutSeconds = 5;
} // namespace

std::optional<std::string> releaseTagFromUrl(const std::string& url) {
    static const std::string kMarker = "/releases/tag/";
    size_t pos = url.rfind(kMarker);
    if (pos == std::string::npos)
        return std::nullopt;
    std::string tag = url.substr(pos + kMarker.size());
    size_t end = tag.find_first_of("/?#");
    if (end != std::string::npos)
        tag.resize(end);
    if (tag.empty())
        return std::nullopt;
    return tag;
}

std::optional<std::string> newerReleaseMessage(const std::string& currentVersion, const std::string& latestTag) {
    std::optional<ReleaseVersion> current = parseReleaseVersion(currentVersion);
    std::optional<ReleaseVersion> latest = parseReleaseVersion(latestTag);
    if (!current || !latest || !(*current < *latest))
        return std::nullopt;
    return "A newer namebreak is available: " + latestTag + " (you have " + currentVersion + "). Download it from " + kReleasesUrl +
           "/tag/" + latestTag;
}

std::optional<std::string> checkForNewerRelease() {
    std::string current = namebreakVersion();
    if (!parseReleaseVersion(current))
        return std::nullopt; // a dev build - nothing to compare against
    HttpClient http;
    HttpResponse resp = http.head(std::string(kReleasesUrl) + "/latest", kTimeoutSeconds);
    if (!resp.ok())
        return std::nullopt;
    std::optional<std::string> latest = releaseTagFromUrl(resp.effectiveUrl);
    if (!latest)
        return std::nullopt;
    return newerReleaseMessage(current, *latest);
}
