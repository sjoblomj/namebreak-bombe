#include "common/version.h"

#include <cctype>
#include <tuple>

// Defined for this one file only (see CMakeLists.txt), so changing the
// version rebuilds just this.
#ifndef NAMEBREAK_VERSION
#define NAMEBREAK_VERSION "dev"
#endif

const char* namebreakVersion() {
    return NAMEBREAK_VERSION;
}

bool ReleaseVersion::operator<(const ReleaseVersion& other) const {
    return std::tie(year, month, day, number) < std::tie(other.year, other.month, other.day, other.number);
}

namespace {
// Exactly `digits` decimal digits at `s[pos]`, advancing `pos` past them.
bool readDigits(const std::string& s, size_t& pos, size_t digits, int& out) {
    if (pos + digits > s.size())
        return false;
    out = 0;
    for (size_t i = 0; i < digits; ++i) {
        if (!std::isdigit((unsigned char) s[pos + i]))
            return false;
        out = out * 10 + (s[pos + i] - '0');
    }
    pos += digits;
    return true;
}
} // namespace

std::optional<ReleaseVersion> parseReleaseVersion(const std::string& tag) {
    ReleaseVersion v;
    size_t pos = 0;
    if (tag.empty() || tag[pos++] != 'v')
        return std::nullopt;
    if (!readDigits(tag, pos, 4, v.year) || pos >= tag.size() || tag[pos++] != '-' || !readDigits(tag, pos, 2, v.month) || pos >= tag.size() ||
        tag[pos++] != '-' || !readDigits(tag, pos, 2, v.day))
        return std::nullopt;
    if (v.month < 1 || v.month > 12 || v.day < 1 || v.day > 31)
        return std::nullopt;
    if (pos < tag.size()) {
        if (tag[pos++] != '.' || pos >= tag.size() || tag.size() - pos > 6 || !readDigits(tag, pos, tag.size() - pos, v.number))
            return std::nullopt;
    }
    return v;
}
