#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>
#include "constants.h"

// Terminology:
// * Candidate = The part of the name that we are brute-forcing
// * Filename  = The Prefix + Candidate + Suffix


std::pair<uint32_t, uint32_t > mpqHashWithPrefixCache_CPU(const char* str, const uint32_t* cryptTable) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    // unsigned so a byte >= 0x80 zero-extends into the crypt-table index/seed
    // arithmetic below instead of sign-extending to a negative value - must
    // match namebreak.cu's device-side hash functions exactly, or a match
    // found on one side would never reproduce on the other.
    unsigned char ch;

    while ((ch = *str++) != '\0') {
        seed1 = cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return {seed1, seed2};
}

void prepareCryptTable(uint32_t* table) {
    uint32_t seed = 0x00100001;
    for (int index1 = 0; index1 < 0x100; ++index1) {
        for (int i = 0; i < 5; ++i) {
            int index2 = i * 0x100 + index1;
            seed = (seed * 125 + 3) % 0x2AAAAB;
            uint32_t temp1 = (seed & 0xFFFF) << 0x10;
            seed = (seed * 125 + 3) % 0x2AAAAB;
            uint32_t temp2 = (seed & 0xFFFF);
            table[index2] = temp1 | temp2;
        }
    }
}


bool stringToIndex(const std::string& str, const std::string& alphabet, uint64_t& out, std::string& error) {
    uint64_t index = 0;
    for (char c : str) {
        size_t pos = alphabet.find(c);
        if (pos == std::string::npos) {
            error = std::string("invalid character in string: '") + c + "'";
            return false;
        }
        index = index * alphabet.size() + pos;
    }
    out = index;
    return true;
}


std::string indexToString(uint64_t index, int len, const std::string& alphabet) {
    std::string result(len, alphabet[0]);
    for (int i = len - 1; i >= 0; --i) {
        result[i] = alphabet[index % alphabet.size()];
        index /= alphabet.size();
    }
    return result;
}

// Compares a and b by their characters' positions in `alphabet`, treating a shorter
// string as if followed by the smallest alphabet character - matching how bounds are
// extended elsewhere (e.g. make_bound_string). Deliberately does this character-by-
// character rather than via stringToIndex: converting a long string to a single index
// overflows uint64_t well before MAX_CANDIDATE_LEN characters (e.g. 50^16), so this
// avoids that entirely regardless of string length.
bool isBeforeInAlphabet(const std::string& a, const std::string& b, const std::string& alphabet, bool& outIsBefore, std::string& error) {
    size_t n = std::max(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        char ca = i < a.size() ? a[i] : alphabet[0];
        char cb = i < b.size() ? b[i] : alphabet[0];
        if (ca == cb)
            continue;
        size_t pa = alphabet.find(ca);
        size_t pb = alphabet.find(cb);
        if (pa == std::string::npos || pb == std::string::npos) {
            error = "invalid character in string during comparison";
            return false;
        }
        outIsBefore = pa < pb;
        return true;
    }
    outIsBefore = false; // equal
    return true;
}

bool getStartCandidate(const std::string& path, const std::string& prefix, const std::string& suffix, std::string& out, std::string& error) {
    // Remove prefix and suffix
    if (path.rfind(prefix, 0) != 0 || path.size() <= prefix.size() + suffix.size()) {
        error = "invalid start filename format";
        return false;
    }

    out = path.substr(prefix.size(), path.size() - prefix.size() - suffix.size());
    return true;
}

std::string make_bound_string(std::string input, int candidateLen) {
    // Return a copy of the given input string. Append the last character until the result string has length candidateLen
    std::string result = input;
    for (int i = input.size(); i < candidateLen; ++i) {
        result += result.back();
    }
    return result.substr(0, candidateLen);
}

std::string remove_prefix_and_suffix(std::string base, std::string prefix, std::string suffix) {
    std::string str = base;
    if (str.find(prefix) == 0) {
        str = str.substr(prefix.length());
    }
    if (str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0) {
        str = str.substr(0, str.size() - suffix.size());
    }
    return str;
}

// Computes the predecessor of `input` (zero-extended to MAX_CANDIDATE_LEN with alphabet's
// min character), i.e. the largest full-length candidate strictly less than `input`. This
// needs a cascading borrow rather than just decrementing the last character: e.g. if the
// last character is already the alphabet minimum, that position wraps to the max and the
// borrow propagates to the character before it (standard "120 - 1 = 119" style borrow).
// If every character in `input` is already the alphabet minimum, `input` itself is already
// the absolute minimum candidate and has no predecessor - saturate to that minimum instead
// of erroring, since callers use this as a starting point for the search.
bool getLowerBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error) {
    if (input.empty()) {
        out = std::string(MAX_CANDIDATE_LEN, alphabet.front());
        return true;
    }

    std::string result = input;
    int i = (int) result.size() - 1;
    for (; i >= 0; --i) {
        auto pos = alphabet.find(result[i]);
        if (pos == std::string::npos) {
            error = std::string("invalid character in string: '") + result[i] + "'";
            return false;
        }
        if (pos == 0) {
            result[i] = alphabet.back(); // borrow: this digit wraps to max, keep borrowing left
            continue;
        }
        result[i] = alphabet[pos - 1];
        break;
    }
    if (i < 0) {
        out = std::string(MAX_CANDIDATE_LEN, alphabet.front());
        return true;
    }

    // Positions beyond input's length are implicitly the min character, which the borrow
    // above already maxes out - so pad the rest with the max character too.
    result.resize(MAX_CANDIDATE_LEN, alphabet.back());
    out = result;
    return true;
}

// Symmetric to getLowerBound: computes the successor of `input` (max-extended to
// MAX_CANDIDATE_LEN), cascading a carry through the string, saturating to the absolute
// maximum candidate if `input` is already all max characters.
bool getUpperBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error) {
    if (input.empty()) {
        out = std::string(MAX_CANDIDATE_LEN, alphabet.front());
        return true;
    }

    std::string result = input;
    int i = (int) result.size() - 1;
    for (; i >= 0; --i) {
        auto pos = alphabet.find(result[i]);
        if (pos == std::string::npos) {
            error = std::string("invalid character in string: '") + result[i] + "'";
            return false;
        }
        if (pos + 1 >= alphabet.size()) {
            result[i] = alphabet.front(); // carry: this digit wraps to min, keep carrying left
            continue;
        }
        result[i] = alphabet[pos + 1];
        break;
    }
    if (i < 0) {
        out = std::string(MAX_CANDIDATE_LEN, alphabet.back());
        return true;
    }

    // Positions beyond input's length are implicitly the max character, which the carry
    // above already wraps to min - so pad the rest with the min character too.
    result.resize(MAX_CANDIDATE_LEN, alphabet.front());
    out = result;
    return true;
}

bool hexToU32(const std::string& s, uint32_t& out) {
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
