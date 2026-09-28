#ifndef NAMEBREAK_NET_UPDATE_CHECK_H
#define NAMEBREAK_NET_UPDATE_CHECK_H

#include <optional>
#include <string>

// Looks up the latest release on GitHub (where kReleasesUrl's /latest
// redirects to - no API token or JSON needed) and, if it's newer than this
// build (namebreakVersion()), returns a message telling the user so. Never
// downloads anything. std::nullopt when there's nothing to say: a dev build,
// no newer release, or the lookup failed (offline, say) - which is never
// worth more than silence.
std::optional<std::string> checkForNewerRelease();

// The pure parts of checkForNewerRelease, for tests:
// The release tag at the end of a ".../releases/tag/<tag>" URL.
std::optional<std::string> releaseTagFromUrl(const std::string& url);
// The message for `latestTag` being newer than `currentVersion`, if it is.
std::optional<std::string> newerReleaseMessage(const std::string& currentVersion, const std::string& latestTag);

#endif // NAMEBREAK_NET_UPDATE_CHECK_H
