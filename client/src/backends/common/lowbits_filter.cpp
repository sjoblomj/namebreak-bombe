#include "backends/common/lowbits_filter.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <type_traits>

namespace {

inline void mpqStep(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = ord + seed1 + seed2 + (seed2 << 5) + 3;
}

} // namespace

std::vector<uint64_t> buildLowBitsFilterTable(const SearchConstants& constants) {
    // Every entry's state has only kLowBitsFilterBits bits (the rest are 0 -
    // they can't affect the result's low bits), and only the result's low
    // kLowBitsFilterBits bits matter. The hash step is a T-function - no bit
    // of its result depends on a higher bit of its input - so all of it can
    // be computed in the narrowest integer that holds those bits, modulo its
    // size: 8-bit lanes for up to 8 bits, 16-bit ones up to 10. The low bits
    // come out exactly as the 32-bit hash's, and the compiler packs four or
    // two times as many lanes into each vector instruction. (The check below,
    // and lowbits_filter_test, recompute entries with the 32-bit hash, from
    // states with random high bits.)
    using Lane = std::conditional_t<(kLowBitsFilterBits <= 8), uint8_t, uint16_t>;
    static_assert(kLowBitsFilterBits <= 8 * sizeof(Lane), "the table's state bits must fit a lane");
    std::vector<uint64_t> table(kLowBitsFilterEntries, 0);
    const uint32_t* cryptTable = constants.cryptTable;
    std::vector<Lane> suffixKey, suffixOrd;
    for (unsigned char ch : constants.suffix) {
        suffixOrd.push_back((Lane) ch);
        suffixKey.push_back((Lane) cryptTable[0x100 + ch]);
    }
    const Lane targetA = (Lane) constants.targetHashA;
    const Lane hashMask = (Lane) kLowBitsFilterHashMask;
    // mpqStep, in Lane arithmetic.
    auto step = [](Lane& seed1, Lane& seed2, Lane key, Lane ord) {
        seed1 = (Lane) (key ^ (Lane) (seed1 + seed2));
        seed2 = (Lane) (ord + seed1 + seed2 + (Lane) (seed2 << 5) + 3);
    };

    // A block of entries at a time, one hash step across the whole block per
    // loop, so the compiler can vectorize it; the block's entries are
    // gathered in `mask`, and written to the table once.
    constexpr uint32_t kBlock = 1024;
    Lane seed1[kBlock], seed2[kBlock];
    uint64_t mask[kBlock];
    for (uint32_t base = 0; base < kLowBitsFilterEntries; base += kBlock) {
        const uint32_t count = std::min(kBlock, kLowBitsFilterEntries - base);
        for (uint32_t b = 0; b < count; ++b)
            mask[b] = 0;
        for (size_t k = 0; k < constants.alphabet.size(); ++k) {
            const unsigned char ch = (unsigned char) constants.alphabet[k];
            const Lane ord = (Lane) ch, key = (Lane) cryptTable[0x100 + ch];
            for (uint32_t b = 0; b < count; ++b) {
                seed1[b] = (Lane) ((base + b) & kLowBitsFilterStateMask);
                seed2[b] = (Lane) ((base + b) >> kLowBitsFilterBits);
                step(seed1[b], seed2[b], key, ord);
            }
            for (size_t i = 0; i < suffixKey.size(); ++i) {
                for (uint32_t b = 0; b < count; ++b)
                    step(seed1[b], seed2[b], suffixKey[i], suffixOrd[i]);
            }
            for (uint32_t b = 0; b < count; ++b)
                mask[b] |= (uint64_t) ((Lane) ((seed1[b] ^ targetA) & hashMask) == 0) << k;
        }
        for (uint32_t b = 0; b < count; ++b)
            table[base + b] = mask[b];
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
