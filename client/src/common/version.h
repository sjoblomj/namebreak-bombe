#ifndef NAMEBREAK_COMMON_VERSION_H
#define NAMEBREAK_COMMON_VERSION_H

#include <optional>
#include <string>

// This build's release - the GitHub release tag it was built for
// ("v2026-09-26"), set with CMake's NAMEBREAK_VERSION (the release workflow
// passes its tag). "dev" for any other build, which never looks for or gets
// told about updates.
const char* namebreakVersion();

inline constexpr const char* kDevVersion = "dev";

// Where releases are published - see update_check.h.
inline constexpr const char* kReleasesUrl = "https://github.com/sjoblomj/namebreak-bombe/releases";

// A release tag, `vYYYY-MM-DD` optionally followed by `.N` for a later
// release the same day, ordered by date and then that number. Mirrors the
// coordinator's client_release.rs.
struct ReleaseVersion {
    int year = 0;
    int month = 0;
    int day = 0;
    int number = 0;

    bool operator<(const ReleaseVersion& other) const;
};

// std::nullopt for anything that isn't a release tag (including "dev").
std::optional<ReleaseVersion> parseReleaseVersion(const std::string& tag);

#endif // NAMEBREAK_COMMON_VERSION_H
