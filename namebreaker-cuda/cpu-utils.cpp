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
    char ch;

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


uint64_t stringToIndex(const std::string& str, std::string alphabet) {
    uint64_t index = 0;
    for (char c : str) {
        size_t pos = alphabet.find(c);
        if (pos == std::string::npos) {
            fprintf(stderr, "Invalid character in string: '%c'\n", c);
            exit(1);
        }
        index = index * alphabet.size() + pos;
    }
    return index;
}


std::string getStartCandidate(std::string path, std::string prefix, std::string suffix) {
    std::string full = path;

    // Remove prefix and suffix
    if (full.rfind(prefix, 0) != 0 || full.size() <= prefix.size() + suffix.size()) {
        fprintf(stderr, "Invalid start filename format.\n");
        return nullptr;
    }

    return full.substr(prefix.size(), full.size() - prefix.size() - suffix.size());
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

std::string getLowerBound(const std::string& input, std::string alphabet) {
    std::string result = input;
    if (result.empty()) return std::string(16, ' ');

    // Find index of the last character
    char& lastChar = result.back();
    auto pos = alphabet.find(lastChar);
    if (pos == std::string::npos || pos - 1 < 0) {
        fprintf(stderr, "Cannot bump last character or character not in alphabet\n");
    }
    lastChar = alphabet[pos - 1];

    // Pad with underscores to length 16
    result.resize(16, '_');
    return result;
}

std::string getUpperBound(const std::string& input, std::string alphabet) {
    std::string result = input;
    if (result.empty()) return std::string(16, ' ');

    // Find index of the last character
    char& lastChar = result.back();
    auto pos = alphabet.find(lastChar);
    if (pos == std::string::npos || pos + 1 >= alphabet.size()) {
        fprintf(stderr, "Cannot bump last character or character not in alphabet\n");
    }
    lastChar = alphabet[pos + 1];

    // Pad with spaces to length 16
    result.resize(16, ' ');
    return result;
}
