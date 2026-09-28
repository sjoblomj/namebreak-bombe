// Release versions (common/version.h) and, in a build with networking, the
// pure parts of the startup update check (net/update_check.h) and the
// release sent when registering. No network access.

#include <cstdio>
#include <string>

#include "common/version.h"
#ifdef NAMEBREAK_WITH_NETWORK
#include "net/protocol.h"
#include "net/update_check.h"
#endif

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static bool older(const std::string& a, const std::string& b) {
    return *parseReleaseVersion(a) < *parseReleaseVersion(b);
}

int main() {
    for (const char* tag : {"v2026-09-26", "v2026-09-26.1", "v2025-12-31.12"})
        check(parseReleaseVersion(tag).has_value(), std::string("parses ") + tag);
    for (const char* tag : {"dev", "", "2026-09-26", "v2026-9-26", "v2026-13-01", "v2026-09-00", "v2026-09-26.", "v2026-09-26.x", "v2026-09-26x"})
        check(!parseReleaseVersion(tag).has_value(), std::string("rejects '") + tag + "'");

    check(older("v2026-09-26", "v2026-09-27"), "a later date is newer");
    check(older("v2026-09-26", "v2026-09-26.1"), "a same-day .N is newer");
    check(older("v2026-09-26.9", "v2026-10-01"), "the date comes first");
    check(!older("v2026-09-26", "v2026-09-26"), "equal isn't older");

#ifdef NAMEBREAK_WITH_NETWORK
    check(releaseTagFromUrl("https://github.com/sjoblomj/namebreak-bombe/releases/tag/v2026-09-26") == std::string("v2026-09-26"),
          "tag from a release URL");
    check(!releaseTagFromUrl("https://github.com/sjoblomj/namebreak-bombe/releases").has_value(), "no tag without /releases/tag/");

    check(newerReleaseMessage("v2026-09-01", "v2026-09-26").has_value(), "a newer release is announced");
    check(!newerReleaseMessage("v2026-09-26", "v2026-09-26").has_value(), "the same release isn't");
    check(!newerReleaseMessage("v2026-10-01", "v2026-09-26").has_value(), "an older one isn't");
    check(!newerReleaseMessage("dev", "v2026-09-26").has_value(), "a dev build never is");

    RegisterRequest req;
    req.clientRelease = "v2026-09-26";
    check(toJson(req).find("\"client_release\":\"v2026-09-26\"") != std::string::npos, "the release is sent when registering");
#endif

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
