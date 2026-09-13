//
// Created by sjoblomj on 2025-06-17.
//

#ifndef NAMEBREAK_CUDA_CPU_UTILS_H
#define NAMEBREAK_CUDA_CPU_UTILS_H

#include <string>

std::pair<uint32_t, uint32_t > mpqHashWithPrefixCache_CPU(const char* str, const uint32_t* cryptTable);

// Below: every function that can encounter a character outside `alphabet`
// reports that as a plain false/error-string return instead of exit()ing the
// process - this code runs from a long-lived coordinator daemon (once per
// claimed range), where a single malformed range must not take the whole
// process down with it. Callers that only ever run once at startup (config
// parsing) still get the same fail-fast behavior; they just do it via their
// own existing `return false` error path instead.

// Returns false (with `error` set) if `str` contains a character not in
// `alphabet`; otherwise fills `out` with str's numeric value in that
// alphabet's base and returns true.
bool stringToIndex(const std::string& str, const std::string& alphabet, uint64_t& out, std::string& error);

std::string indexToString(uint64_t index, int len, const std::string& alphabet);

// Returns false (with `error` set) if `a` or `b` contains a character not in
// `alphabet`; otherwise fills `outIsBefore` with whether `a` sorts before `b`
// and returns true.
bool isBeforeInAlphabet(const std::string& a, const std::string& b, const std::string& alphabet, bool& outIsBefore, std::string& error);

void prepareCryptTable(uint32_t* table);

// Returns false (with `error` set) if `path` doesn't start with `prefix` or
// isn't long enough to also hold `suffix`; otherwise fills `out` with the
// candidate portion and returns true.
bool getStartCandidate(const std::string& path, const std::string& prefix, const std::string& suffix, std::string& out, std::string& error);

std::string make_bound_string(std::string input, int candidateLen);
std::string remove_prefix_and_suffix(std::string base, std::string prefix, std::string suffix);

// Returns false (with `error` set) if `input` contains a character not in
// `alphabet`; otherwise fills `out` per getLowerBound/getUpperBound's own
// doc comments (in cpu-utils.cpp) and returns true.
bool getLowerBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error);
bool getUpperBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error);

// Parses a hex string into a uint32_t - tolerates (but doesn't require) a
// leading "0x"/"0X", matching how target hashes are written everywhere in
// this project (CLI args, config.conf, the coordinator's JSON). Returns
// false (rather than throwing) on anything that doesn't parse.
bool hexToU32(const std::string& s, uint32_t& out);

#endif //NAMEBREAK_CUDA_CPU_UTILS_H
