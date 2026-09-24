#include "backends/common/row_batch.h"

#include <cstdio>

#include "engine/candidate.h"

RowRange rowRangeFor(uint64_t start, uint64_t count, int alphabetSize) {
    const uint64_t end = start + count;
    RowRange range;
    range.firstRow = start / alphabetSize;
    const uint64_t lastRow = (end - 1) / alphabetSize;
    range.rowCount = lastRow - range.firstRow + 1;
    range.firstRowStartK = (int) (start - range.firstRow * alphabetSize);
    range.lastRowEndK = (int) (end - lastRow * alphabetSize);
    return range;
}

void HitVerifier::begin(const SearchConstants& constants) {
    alphabet_ = constants.alphabet;
    suffix_ = constants.suffix;
    cryptTable_.assign(constants.cryptTable, constants.cryptTable + 0x500);
    targetA_ = constants.targetHashA;
    targetB_ = constants.targetHashB;
}

uint32_t HitVerifier::hashFromScratch(const std::string& filename, int tableOffset) const {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : filename) {
        seed1 = cryptTable_[tableOffset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}

void HitVerifier::addHits(const std::vector<uint64_t>& trailingIndices, int trailingLen, const BatchParams& params,
                          BatchOutcome& outcome) const {
    const std::string prefix(params.prefix, params.prefixSize);
    for (uint64_t index : trailingIndices) {
        std::string filename = prefix + indexToString(index, trailingLen, alphabet_) + suffix_;
        uint32_t hashA = hashFromScratch(filename, 0x100);
        if (hashA != targetA_) {
            printf("WARNING: the backend reported a hashA hit for '%s' but hashing it from scratch gives 0x%08X, not the target 0x%08X\n",
                   filename.c_str(), hashA, targetA_);
        }
        if (!outcome.found && hashFromScratch(filename, 0x200) == targetB_) {
            outcome.found = true;
            outcome.foundFilename = filename;
        }
        outcome.hits.push_back(filename);
    }
}
