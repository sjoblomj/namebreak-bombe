#include "backends/common/lowbits_filter.h"

#include <algorithm>
#include <cstdio>
#include <random>

namespace {

inline void mpqStep(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = ord + seed1 + seed2 + (seed2 << 5) + 3;
}

} // namespace

std::vector<uint64_t> buildLowBitsFilterTable(const SearchConstants& constants) {
    std::vector<uint64_t> table(kLowBitsFilterEntries, 0);
    const uint32_t* cryptTable = constants.cryptTable;
    std::vector<uint32_t> suffixKey, suffixOrd;
    for (unsigned char ch : constants.suffix) {
        suffixOrd.push_back(ch);
        suffixKey.push_back(cryptTable[0x100 + ch]);
    }

    // A block of entries at a time, one hash step across the whole block per
    // loop, so the compiler can vectorize it.
    constexpr uint32_t kBlock = 256;
    uint32_t seed1[kBlock], seed2[kBlock];
    for (size_t k = 0; k < constants.alphabet.size(); ++k) {
        const uint32_t ord = (unsigned char) constants.alphabet[k];
        const uint32_t key = cryptTable[0x100 + ord];
        for (uint32_t base = 0; base < kLowBitsFilterEntries; base += kBlock) {
            const uint32_t count = std::min(kBlock, kLowBitsFilterEntries - base);
            for (uint32_t b = 0; b < count; ++b) {
                // The entry's state, with every high bit 0 - they can't
                // affect the result's low bits.
                seed1[b] = (base + b) & kLowBitsFilterStateMask;
                seed2[b] = (base + b) >> kLowBitsFilterBits;
                mpqStep(seed1[b], seed2[b], key, ord);
            }
            for (size_t i = 0; i < suffixKey.size(); ++i) {
                for (uint32_t b = 0; b < count; ++b)
                    mpqStep(seed1[b], seed2[b], suffixKey[i], suffixOrd[i]);
            }
            for (uint32_t b = 0; b < count; ++b) {
                if (((seed1[b] ^ constants.targetHashA) & kLowBitsFilterHashMask) == 0)
                    table[base + b] |= uint64_t(1) << k;
            }
        }
    }
    return table;
}

bool checkLowBitsFilterTable(const std::vector<uint64_t>& table, const SearchConstants& constants, uint32_t entriesToCheck,
                             int highBitRounds, uint64_t seed, std::string& error) {
    char message[512];
    if (table.size() != kLowBitsFilterEntries) {
        snprintf(message, sizeof(message), "the table has %zu entries, not %u", table.size(), kLowBitsFilterEntries);
        error = message;
        return false;
    }
    const uint32_t* cryptTable = constants.cryptTable;
    const int alphabetSize = (int) constants.alphabet.size();
    std::mt19937_64 rng(seed);
    const uint32_t checks = entriesToCheck == 0 ? kLowBitsFilterEntries : entriesToCheck;
    for (uint32_t c = 0; c < checks; ++c) {
        const uint32_t entry = entriesToCheck == 0 ? c : (uint32_t) (rng() % kLowBitsFilterEntries);
        for (int round = 0; round < highBitRounds; ++round) {
            // A state lowBitsFilterIndex maps to `entry`, with random high bits.
            const uint32_t seed1 = ((uint32_t) rng() & ~kLowBitsFilterStateMask) | (entry & kLowBitsFilterStateMask);
            const uint32_t seed2 = ((uint32_t) rng() & ~kLowBitsFilterStateMask) | (entry >> kLowBitsFilterBits);
            if (lowBitsFilterIndex(seed1, seed2) != entry) {
                snprintf(message, sizeof(message), "lowBitsFilterIndex(0x%08X, 0x%08X) is %u, not entry %u", seed1, seed2,
                         lowBitsFilterIndex(seed1, seed2), entry);
                error = message;
                return false;
            }
            for (int k = 0; k < 64; ++k) {
                const bool bit = (table[entry] >> k) & 1;
                bool expected = false;
                if (k < alphabetSize) {
                    uint32_t a = seed1, b = seed2;
                    const unsigned char ch = (unsigned char) constants.alphabet[k];
                    mpqStep(a, b, cryptTable[0x100 + ch], ch);
                    for (unsigned char s : constants.suffix)
                        mpqStep(a, b, cryptTable[0x100 + s], s);
                    expected = ((a ^ constants.targetHashA) & kLowBitsFilterHashMask) == 0;
                }
                if (bit != expected) {
                    snprintf(message, sizeof(message),
                             "entry %u, character %d: bit is %d, but from state (0x%08X, 0x%08X) the hashA's low bits %s the target's (0x%08X)",
                             entry, k, (int) bit, seed1, seed2, expected ? "match" : "don't match", constants.targetHashA);
                    error = message;
                    return false;
                }
            }
        }
    }
    return true;
}
