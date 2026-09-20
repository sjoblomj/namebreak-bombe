#ifndef NAMEBREAK_ENGINE_MPQ_HASH_H
#define NAMEBREAK_ENGINE_MPQ_HASH_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// The MPQ hash on the CPU. The GPU backends have their own copy of the same
// recurrence; the two must agree bit for bit, or a match found on one side
// would never reproduce on the other.

// Fills `table` (0x500 entries) with the MPQ crypt table.
void prepareCryptTable(uint32_t* table);

// Hash state (seed1, seed2) after feeding `str` through the hashA recurrence
// (crypt table offset 0x100), starting from the standard initial seeds. Used
// to precompute a prefix's contribution once, rather than per candidate.
std::pair<uint32_t, uint32_t> mpqHashWithPrefixCache_CPU(const char* str, const uint32_t* cryptTable);

// Incrementally hashes a fixed base state (typically the hash of a fixed
// filename prefix) extended by a variable-length "leading" string of
// `leadingLen` alphabet characters, walked in the same odometer order as
// indexToString/stringToIndex (last character fastest-changing, like an
// odometer's rightmost digit).
//
// advance() moves to the next leading value the same way runSearch's
// leadingIdx loop does (always exactly +1), but instead of re-hashing the
// whole leading string from scratch each time (what calling
// mpqHashWithPrefixCache_CPU(prefix + indexToString(leadingIdx, ...))  every
// iteration amounts to - leadingLen+ hash steps every single time), it keeps
// a stack of the hash state after each prefix length (0 characters of
// leading, 1 character, 2, ...) and only re-hashes from the lowest position
// that actually changed - the same amortized-O(1)-per-step argument as
// incrementing a mixed-radix counter (position p only changes every
// alphabet.size()^p increments, so summed over N increments the total work
// is O(N), not O(N * leadingLen)). Only reset() ever pays the full
// O(leadingLen) cost, once, to seed a specific starting value.
class IncrementalPrefixHasher {
public:
    // `baseState` is the hash state of the fixed part alone (e.g.
    // mpqHashWithPrefixCache_CPU(prefix, cryptTable)) - what advance()/reset()
    // extend by the leading characters. `alphabet` and `cryptTable` must
    // outlive this object; both are copied/referenced as given elsewhere in
    // this codebase (small, process-lifetime data).
    IncrementalPrefixHasher(std::pair<uint32_t, uint32_t> baseState, int leadingLen,
                             std::string alphabet, const uint32_t* cryptTable);

    // One-time full hash to (re)start at a specific leading value - the only
    // O(leadingLen) operation this class performs. Must be called once
    // before the first advance().
    void reset(uint64_t leadingIdx);

    // Moves to the next leading value (leadingIdx + 1) - O(1) amortized.
    // Undefined which leading value results if called before reset(), or
    // past the last representable value (alphabet.size()^leadingLen - 1);
    // callers own bounding the loop, exactly as the leadingIdx loop already
    // does via startLeadingIdx/endLeadingIdx.
    void advance();

    // The current leading characters (leadingLen of them).
    const std::string& leading() const { return leading_; }

    // Hash state after the fixed part followed by the current leading value -
    // what mpqHashWithPrefixCache_CPU(prefix + leading()) would return.
    std::pair<uint32_t, uint32_t> state() const { return stack_.back(); }

private:
    std::string alphabet_;
    const uint32_t* cryptTable_;
    int leadingLen_;
    std::string leading_;
    std::vector<int> digitIndex_;
    // stack_[k] = hash state after the fixed part + leading_[0..k-1];
    // stack_[0] = baseState, stack_[leadingLen_] = state().
    std::vector<std::pair<uint32_t, uint32_t>> stack_;
};

#endif // NAMEBREAK_ENGINE_MPQ_HASH_H
