#ifndef NAMEBREAK_ENGINE_DURATION_H
#define NAMEBREAK_ENGINE_DURATION_H

#include <cstdint>
#include <cstdio>
#include <string>

// `seconds` for a human: "6s", "4m05s", "1h02m03s", "2d03h04m" - or "?" if
// it's negative or absurdly long. What every mode's progress line says is
// left.
inline std::string formatDuration(double seconds) {
    if (seconds < 0 || seconds > 1e12)
        return "?";
    uint64_t s = (uint64_t) (seconds + 0.5);
    char buf[64];
    if (s >= 86400)
        snprintf(buf, sizeof buf, "%llud%02lluh%02llum", (unsigned long long) (s / 86400), (unsigned long long) (s / 3600 % 24),
                 (unsigned long long) (s / 60 % 60));
    else if (s >= 3600)
        snprintf(buf, sizeof buf, "%lluh%02llum%02llus", (unsigned long long) (s / 3600), (unsigned long long) (s / 60 % 60),
                 (unsigned long long) (s % 60));
    else if (s >= 60)
        snprintf(buf, sizeof buf, "%llum%02llus", (unsigned long long) (s / 60), (unsigned long long) (s % 60));
    else
        snprintf(buf, sizeof buf, "%llus", (unsigned long long) s);
    return buf;
}

#endif // NAMEBREAK_ENGINE_DURATION_H
