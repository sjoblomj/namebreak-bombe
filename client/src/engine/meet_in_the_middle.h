#ifndef NAMEBREAK_ENGINE_MEET_IN_THE_MIDDLE_H
#define NAMEBREAK_ENGINE_MEET_IN_THE_MIDDLE_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/candidate.h"
#include "engine/hash_match.h"
#include "engine/limits.h"
#include "engine/mpq_hash.h"

// A meet-in-the-middle search: the file's name is a known prefix, an unknown
// directory, and a known suffix - and the hash state right after the
// directory, where the suffix begins, is already known for hashA and for
// hashB. That state is the meeting point, and it's what the search looks
// for: the directory is whatever string takes the hash from the prefix's
// state to it.
// (A meeting point is recovered from two files of the same unknown directory
// whose basenames are known - see README.md's "Meet-in-the-middle mode".)
//
// Knowing the state at both ends is what makes the mode worth having,
// because every step of the hash
//     seed1 = key[ch] ^ (seed1 + seed2)
//     seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3
// can be undone when the character is known (hashStepBack below). So the
// search can come at the directory from both ends and meet in the middle:
//
//   * forward, from the prefix's state: every directory of the table's
//     length, kept in a table by the state it reaches (MeetInTheMiddleTable);
//   * backward, from the meeting point: every *tail* - a directory's last
//     characters - rewound character by character (MeetInTheMiddleWalk),
//     and the state it reaches looked up in the table.
//
// A tail of j characters whose state is in the table means a directory of
// tableLength + j characters: the table's prefix, then the tail. So a
// directory of n characters costs about alphabet^(n - tableLength) tails
// instead of the alphabet^n candidates a brute force hashes - the table
// buys back one character of search per character it holds, as far as there
// is memory for it. Directories shorter than the table's length are found
// while it's being built, by the same walk (MeetInTheMiddleTable::build).
//
// Terminology:
// * Directory    = the part of the name that is unknown, what a
//   meet-in-the-middle search brute-forces (the "candidate" of the other
//   modes)
// * Filename     = Prefix + Directory + Suffix
// * MeetingPoint = the hash state after Prefix + Directory
// * Tail         = a directory's last characters, which the backward walk
//   rewinds from the meeting point; its first characters are the table's

// 33's inverse modulo 2^32 (33 * kInv33 == 1): seed2's step multiplies it by
// 1 + (1 << 5), so undoing that step divides by 33 - which, 33 being odd, is
// a plain multiplication by this.
constexpr uint32_t kInv33 = 0x3E0F83E1u;

// One step of the MPQ hash recurrence undone: the state hashStep (mpq_hash.h)
// made `after` from, given the character it hashed. `table` is the crypt
// table at that hash's offset (cryptTable + kHashAOffset), as hashStep takes
// it. hashStepBack(hashStep(s, ch, table), ch, table) == s for every state
// and character.
inline HashState hashStepBack(HashState after, unsigned char ch, const uint32_t* table) {
    // From seed2's step, seed2_after = ch + seed1_after + 3 + 33 * seed2_before.
    const uint32_t before2 = (after.second - ch - after.first - 3) * kInv33;
    // And from seed1's, seed1_after ^ key[ch] = seed1_before + seed2_before.
    const uint32_t before1 = (after.first ^ table[ch]) - before2;
    return {before1, before2};
}

// `state` with `text` taken back off its end, the last character first - the
// inverse of continueHash(state, text, offset, cryptTable).
HashState rewindHash(HashState state, const std::string& text, int offset, const uint32_t* cryptTable);

// The key a meet-in-the-middle search compares states by - see meetingPointKey
// (hash_match.h), which masks it to the bits that must match (all 64 in the
// program, fewer in the tests' dense-hit builds).
inline uint64_t meetingPointKey(HashState state) {
    return meetingPointKey(state.first, state.second);
}

// --- The pruning rules, from the right ---------------------------------------
//
// The same rules as the other modes (PruneRules, candidate.h), as much of
// them as a directory's *tail* can be judged by on its own: a tail that
// breaks one of these is never part of a directory that passes them, and
// neither is any longer tail ending in it, so the backward walk can skip the
// lot.
//
// Not every rule is one of those. min_backslash_count can always be reached
// by characters further left, and an unopened bracket in a tail may well
// have been opened to its left, so neither says anything about a tail - they
// are only checked once a whole directory is in hand (dirPasses).

// What the rules need to know about the characters to the right of the one
// being checked - the ones the walk has already rewound.
struct TailPruneState {
    int symbolRun = 0;            // non-alphanumeric, non-space characters in a row, up to 2
    int backslashes = 0;          // how many '\' the tail has so far
    bool nextIsBackslash = false; // whether the character to the right is a '\'
};

// The state a tail starts in: nothing of the directory to its right, but the
// suffix is - so with prune_adjacent_backslashes, a suffix that begins with
// a '\' leaves the directory unable to end with one. The symbol run starts
// over at the directory's last character, as it does at a candidate's first
// character everywhere else (PruneRules): the rule is about the directory,
// not about the name around it.
TailPruneState tailPruneStart(const std::string& suffix);

// Steps `state` over `c`, the character to the left of those so far; false if
// `c` breaks one of `rules` (`state` is then no longer meaningful). A false
// here holds for every longer tail that ends in these characters too.
bool tailPruneStep(const PruneRules& rules, TailPruneState& state, char c);

// The PruneState a directory's first character is checked from: the brackets
// `prefix` leaves open, and whether it ends with a '\'.
PruneState dirPruneStart(const std::string& prefix);

// Whether `dir` passes `rules` in full - every character of it checked, left
// to right, from dirPruneStart(prefix), plus the rules that only a whole
// directory settles: min_backslash_count over all of it, and a '\' it ends
// with next to one `suffix` begins with. Every directory the walk and the
// table skip is one this rejects, so a meet-in-the-middle search reports
// exactly the directories this accepts (and that match the meeting point).
bool dirPasses(const PruneRules& rules, const std::string& prefix, const std::string& suffix, const std::string& dir);

// --- The space ---------------------------------------------------------------

// Which directories a meet-in-the-middle search covers, and how the nodes of
// its backward walk are numbered.
//
// A node is one tail, of 0 to maxTailLength() characters. Its number counts
// through the lengths, shortest first, and within a length through the tails
// in the order of the *reversed* tail - the last character of the directory
// is the slowest to change, because it's the first the walk rewinds and so
// the one whose state the most other tails share. (The empty tail, number 0,
// is a node like any other: the directory is then exactly the table's
// length.)
class MeetInTheMiddleSpace {
public:
    // False, with `error` set, on an unusable alphabet or length - including
    // a tail space too big to number in 64 bits.
    //
    // `tableLength` is how many of a directory's first characters the table
    // holds, 0 to maxLength (clamp it before calling: a table longer than
    // the longest directory is the caller's mistake, not a silent fix).
    static bool create(const std::string& alphabet, int minLength, int maxLength, int tableLength, MeetInTheMiddleSpace& out,
                       std::string& error);

    const std::string& alphabet() const { return alphabet_; }
    int minLength() const { return minLength_; }
    int maxLength() const { return maxLength_; }
    int tableLength() const { return tableLength_; }
    // The longest tail the walk goes to: maxLength() - tableLength().
    int maxTailLength() const { return maxLength_ - tableLength_; }

    // How many tails of `tailLength` characters there are: alphabet^length.
    uint64_t countOf(int tailLength) const { return counts_[tailLength]; }
    // How many nodes there are in all - one per tail of every length.
    uint64_t size() const { return size_; }

    // Node `number`'s tail: its length, its index among the tails of that
    // length, and the tail itself (in the directory's own order, its last
    // character last).
    int tailLengthOf(uint64_t number) const;
    uint64_t indexOf(uint64_t number) const;
    std::string tailOf(uint64_t number) const;
    // The number of the node of tail `index` of `tailLength` characters -
    // and of the first node of a length, index 0.
    uint64_t numberOf(int tailLength, uint64_t index) const;

    // The index of `tail` among the tails of its length: the index of the
    // tail reversed (stringToIndex, candidate.h). False, with `error` set,
    // if it has a character the alphabet doesn't.
    bool indexOfTail(const std::string& tail, uint64_t& out, std::string& error) const;

private:
    std::string alphabet_;
    int minLength_ = 1;
    int maxLength_ = 1;
    int tableLength_ = 0;
    // counts_[j] = alphabet^j, firsts_[j] = the number of its first node.
    std::vector<uint64_t> counts_;
    std::vector<uint64_t> firsts_;
    uint64_t size_ = 0;
};

// How many directories a table of `tableLength` characters holds, at most -
// the pruning rules may leave out some of them. UINT64_MAX if that doesn't
// fit in 64 bits.
uint64_t meetInTheMiddleTableCount(int alphabetSize, int tableLength);

// The largest table length whose table fits in `bytes` of memory, for an
// alphabet of `alphabetSize` characters - 0 if even a one-character table
// doesn't, and never more than `maxLength`. See
// MeetInTheMiddleTable::bytesFor.
int meetInTheMiddleTableLengthForBytes(int alphabetSize, int maxLength, uint64_t bytes);

// --- The forward half --------------------------------------------------------

// Every directory of the space's table length, by the hash state it reaches:
// the forward half of the search, which the backward walk looks up in.
//
// Only the keys are kept, sorted - eight bytes a directory, where the
// directory itself would be another eight (and the table is as big as
// memory allows, so that doubling would cost a whole character of reach).
// The few directories a search ever looks up again are re-walked instead
// (prefixesFor).
class MeetInTheMiddleTable {
public:
    // The memory a table of `tableLength` holds, for an alphabet of
    // `alphabetSize`: its keys and its filter. UINT64_MAX if that length's
    // table can't be counted in 64 bits at all.
    static uint64_t bytesFor(int alphabetSize, int tableLength);

    // Builds the table of `space`: the hashA state of every directory of
    // space.tableLength() characters that `rules` allow, hashed onto
    // `startState` (the state after the filename's prefix) and keyed by
    // meetingPointKey.
    //
    // The walk gets there through every *shorter* directory, so it checks
    // those against `meetingPoint` as it goes, since nothing else would: the
    // backward walk only ever adds characters to a full-length prefix.
    // Every shorter directory that matches the meeting point - and that is at
    // least space.minLength() characters - is appended to `shortMatches`,
    // for the caller to check in full as it does the walk's own hits.
    //
    // `cryptTable` must outlive the table: prefixesFor walks the hash again.
    // False, with `error` set, if the table doesn't fit in memory.
    static bool build(const MeetInTheMiddleSpace& space, HashState startState, const PruneState& prefixEntry, const PruneRules& rules,
                      const uint32_t* cryptTable, HashState meetingPoint, std::vector<std::string>& shortMatches, MeetInTheMiddleTable& out,
                      std::string& error);

    // Whether the table may hold `key` - a bitmap of the keys' mixed bits,
    // which answers no for about all but one lookup in kFilterBitsPerKey
    // without touching the keys themselves. Every key it holds answers yes.
    bool mayHold(uint64_t key) const {
        const uint64_t bit = filterBit(key);
        return (filter_[bit >> 6] >> (bit & 63)) & 1;
    }
    // Whether the table holds `key`.
    bool holds(uint64_t key) const;
    // Every directory of the table's length whose state has one of `keys` -
    // out[i] is key[i]'s, in alphabet order. One walk of the table's space
    // serves them all, so a caller with several keys should ask once.
    void prefixesFor(const std::vector<uint64_t>& keys, std::vector<std::vector<std::string>>& out) const;
    void prefixesFor(uint64_t key, std::vector<std::string>& out) const;

    // How many directories the table holds (the rules having skipped the
    // rest), and the memory it takes.
    uint64_t size() const { return keys_.size(); }
    uint64_t bytes() const { return keys_.size() * sizeof(uint64_t) + filter_.size() * sizeof(uint64_t); }
    // The keys, sorted, and the filter - for a backend that probes the table
    // itself (a GPU backend copies both into its own memory).
    const uint64_t* keys() const { return keys_.data(); }
    const uint64_t* filter() const { return filter_.data(); }
    uint64_t filterWords() const { return filter_.size(); }
    uint64_t filterBits() const { return filterBits_; }
    // How far down filterBit shifts a key's mixed bits: 64 - log2(filterBits).
    // A backend that probes the filter itself does the mixing in its kernel,
    // but takes the shift from here rather than working it out again.
    int filterShift() const { return filterShift_; }
    // The bit of the filter `key` is kept in. The same mixing a backend's
    // own copy of the filter must use.
    uint64_t filterBit(uint64_t key) const {
        // Multiply-shift: the key's bits, mixed, taken from the top. The
        // state's low bits carry no information about its high ones (see
        // backends/common/lowbits_filter.h), so a filter indexed by the key
        // as it is would leave whole stretches of itself unused.
        return (key * 0x9E3779B97F4A7C15ull) >> filterShift_;
    }

    // About how many bits of filter each key gets: a lookup that isn't in
    // the table gets past it with a probability of about one in this.
    static constexpr uint64_t kFilterBitsPerKey = 8;

private:
    MeetInTheMiddleSpace space_;
    HashState startState_{0, 0};
    PruneState prefixEntry_;
    PruneRules rules_;
    const uint32_t* cryptTable_ = nullptr;
    std::vector<uint64_t> keys_; // sorted
    std::vector<uint64_t> filter_;
    uint64_t filterBits_ = 0;
    int filterShift_ = 64;
};

// --- The backward half -------------------------------------------------------

// The tails of one length, walked from the meeting point: the mirror of
// IncrementalPrefixHasher (mpq_hash.h), running the hash backwards.
//
// Like it, it keeps the hash state after each rewound character and
// re-rewinds only from the one that changed, so a step costs O(1) hash steps
// amortized rather than one per character of the tail. The pruning rules ride
// along on the same stack: a tail they reject is skipped together with every
// longer tail that ends in it, by jumping the index straight past them all.
class MeetInTheMiddleWalk {
public:
    // `table` is the crypt table at hashA's offset (cryptTable +
    // kHashAOffset), and must outlive the walk. `entry` is the rules' state
    // to the directory's right - tailPruneStart(suffix).
    MeetInTheMiddleWalk(std::string alphabet, int tailLength, HashState meetingPoint, const uint32_t* table, const PruneRules& rules,
                 const TailPruneState& entry);

    // Positions the walk on the first tail the rules allow whose index is in
    // [from, to) - false if there is none, and the walk is then spent.
    bool start(uint64_t from, uint64_t to);
    // Moves on to the next such tail; false once the range is done.
    bool next();

    // The tail the walk is on: its index, the characters themselves (in the
    // directory's own order, its last character last) and the hash state
    // after the directory's first characters - the meeting point, rewound over
    // the tail. Only between a true start()/next() and the next call.
    uint64_t index() const { return index_; }
    const std::string& tail() const { return tail_; }
    HashState state() const { return stack_[tailLength_]; }

private:
    // Rebuilds the stack from the digit that last changed, and - if the
    // rules reject the tail - skips past it and tries again, until a tail
    // they allow turns up or the range runs out.
    bool settle();
    // Rewinds digit `depth` (the depth+1'th character from the directory's
    // end) onto stack_[depth], and applies the rules to it; fills in
    // stack_[depth + 1] and prune_[depth + 1]. False if the rules reject it.
    bool stepInto(int depth);
    // Rebuilds the stack from `depth` down, stopping at the first character
    // the rules reject; returns how deep it got - tailLength_ if the whole
    // tail is allowed.
    int rebuildFrom(int depth);
    // Moves the index to the next tail whose first `depth` rewound
    // characters differ from this one's (depth == tailLength_: the next tail),
    // and the digits with it. False once that's past the range.
    bool skip(int depth);

    std::string alphabet_;
    const uint32_t* table_;
    PruneRules rules_;
    int tailLength_;
    uint64_t index_ = 0;
    uint64_t end_ = 0;
    // The shallowest digit that changed since the stack was built - where
    // settle() has to rebuild it from.
    int changedFrom_ = 0;
    std::string tail_;
    // digit_[k] is the alphabet position of the k'th character the walk
    // rewinds - tail_[tailLength_ - 1 - k] - so digit_[0] is the index's
    // most significant.
    std::vector<int> digit_;
    // place_[k] = alphabet^(tailLength_ - k): what one step of digit_[k] is
    // worth in the index.
    std::vector<uint64_t> place_;
    // stack_[k] = the meeting point rewound over the first k characters,
    // prune_[k] = the rules' state after them. stack_[0] is the meeting point.
    std::vector<HashState> stack_;
    std::vector<TailPruneState> prune_;
};

#endif // NAMEBREAK_ENGINE_MEET_IN_THE_MIDDLE_H
