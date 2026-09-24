#ifndef NAMEBREAK_COMMON_STRING_UTIL_H
#define NAMEBREAK_COMMON_STRING_UTIL_H

#include <cstdint>
#include <exception>
#include <string>

// `s` without leading/trailing spaces, tabs, CRs and LFs.
inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Parses a hex string into a uint32_t - tolerates (but doesn't require) a
// leading "0x"/"0X", matching how target hashes are written everywhere in
// this project (CLI args, config.conf, the coordinator's JSON). Returns
// false (rather than throwing) on anything that doesn't parse.
inline bool hexToU32(const std::string& s, uint32_t& out) {
    try {
        size_t consumed = 0;
        unsigned long value = std::stoul(s, &consumed, 16);
        if (consumed != s.size())
            return false; // trailing garbage
        out = (uint32_t) value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

#endif // NAMEBREAK_COMMON_STRING_UTIL_H
