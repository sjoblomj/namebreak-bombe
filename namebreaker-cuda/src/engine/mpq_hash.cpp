#include "engine/mpq_hash.h"

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

std::pair<uint32_t, uint32_t> mpqHashWithPrefixCache_CPU(const char* str, const uint32_t* cryptTable) {
    uint32_t seed1 = 0x7FED7FED;
    uint32_t seed2 = 0xEEEEEEEE;
    // unsigned so a byte >= 0x80 zero-extends into the crypt-table index/seed
    // arithmetic below instead of sign-extending to a negative value - must
    // match cuda_backend.cu's device-side hash functions exactly, or a match
    // found on one side would never reproduce on the other.
    unsigned char ch;

    while ((ch = *str++) != '\0') {
        seed1 = cryptTable[0x100 + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return {seed1, seed2};
}

// One step of the same MPQ hash recurrence as mpqHashWithPrefixCache_CPU
// above (and hash_kernels.cuh's mpqHashStep on the device side) - factored
// out so IncrementalPrefixHasher's reset()/advance() can't drift from it.
static inline void mpqHashStepCPU(unsigned char ch, const uint32_t* cryptTable, uint32_t& seed1, uint32_t& seed2) {
    seed1 = cryptTable[0x100 + ch] ^ (seed1 + seed2);
    seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
}

IncrementalPrefixHasher::IncrementalPrefixHasher(std::pair<uint32_t, uint32_t> baseState, int leadingLen,
                                                  std::string alphabet, const uint32_t* cryptTable)
    : alphabet_(std::move(alphabet)), cryptTable_(cryptTable), leadingLen_(leadingLen) {
    leading_.assign(leadingLen_, alphabet_.empty() ? '\0' : alphabet_[0]);
    digitIndex_.assign(leadingLen_, 0);
    stack_.reserve(leadingLen_ + 1);
    stack_.push_back(baseState);
}

void IncrementalPrefixHasher::reset(uint64_t leadingIdx) {
    uint64_t idx = leadingIdx;
    for (int i = leadingLen_ - 1; i >= 0; --i) {
        digitIndex_[i] = (int) (idx % alphabet_.size());
        leading_[i] = alphabet_[digitIndex_[i]];
        idx /= alphabet_.size();
    }

    stack_.resize(1); // keep only stack_[0] (the base state), rebuild the rest
    for (int i = 0; i < leadingLen_; ++i) {
        uint32_t seed1 = stack_.back().first, seed2 = stack_.back().second;
        mpqHashStepCPU((unsigned char) leading_[i], cryptTable_, seed1, seed2);
        stack_.emplace_back(seed1, seed2);
    }
}

void IncrementalPrefixHasher::advance() {
    int p = leadingLen_ - 1;
    while (p >= 0) {
        if (digitIndex_[p] + 1 < (int) alphabet_.size()) {
            digitIndex_[p] += 1;
            leading_[p] = alphabet_[digitIndex_[p]];
            break;
        }
        digitIndex_[p] = 0;
        leading_[p] = alphabet_[0];
        p -= 1;
    }
    // p < 0 means every digit was at its max and just wrapped to 0 (a full-
    // width carry, e.g. leadingLen_==1 wrapping from the last character back
    // to the first) - digitIndex_/leading_ are already correctly all-zero
    // from the loop above, so rebuilding from position 0 (using the base
    // state) is exactly right; clamping to 0 here handles that the same way
    // as any other carry, rather than as a special case.
    int rebuildFrom = p < 0 ? 0 : p;

    // Positions [0, rebuildFrom) are unaffected by this increment;
    // [rebuildFrom, leadingLen_) just changed (rebuildFrom itself
    // incremented, or wrapped to 0; anything after it was reset to 0 by the
    // carry above) and needs re-hashing from stack_[rebuildFrom] onward.
    stack_.resize(rebuildFrom + 1);
    for (int i = rebuildFrom; i < leadingLen_; ++i) {
        uint32_t seed1 = stack_.back().first, seed2 = stack_.back().second;
        mpqHashStepCPU((unsigned char) leading_[i], cryptTable_, seed1, seed2);
        stack_.emplace_back(seed1, seed2);
    }
}
