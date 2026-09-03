//
// Created by sjoblomj on 2025-06-17.
//

#ifndef NAMEBREAK_CUDA_CPU_UTILS_H
#define NAMEBREAK_CUDA_CPU_UTILS_H

#include <string>

std::pair<uint32_t, uint32_t > mpqHashWithPrefixCache_CPU(const char* str, const uint32_t* cryptTable);
uint64_t stringToIndex(const std::string& str, std::string alphabet);
void prepareCryptTable(uint32_t* table);
std::string getStartCandidate(std::string path, std::string prefix, std::string suffix);
std::string make_bound_string(std::string input, int candidateLen);
std::string remove_prefix_and_suffix(std::string base, std::string prefix, std::string suffix);
std::string getLowerBound(const std::string& input, std::string alphabet);
std::string getUpperBound(const std::string& input, std::string alphabet);

#endif //NAMEBREAK_CUDA_CPU_UTILS_H
