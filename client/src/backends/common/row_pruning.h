#ifndef NAMEBREAK_BACKENDS_COMMON_ROW_PRUNING_H
#define NAMEBREAK_BACKENDS_COMMON_ROW_PRUNING_H

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "engine/backend.h"

// Which row groups a GPU kernel searches, when a search prunes the whole
// candidate (SearchConstants::trailingRules). When it doesn't, a kernel walks
// every group the batch touches, as it always did: a kernel walking a list of
// them all measured 5-6% slower.
//
// A row *group* is the alphabetSize consecutive rows that share every row
// character but the last (see kChunksPerGroup in backends/cuda/cuda_backend.cu):
// group g holds rows g * alphabetSize .. g * alphabetSize + alphabetSize - 1,
// the first trailingLen - 2 characters of a candidate being g's digits, and
// the row's own last character d the next. Whether a candidate breaks a rule
// at those characters, and the text inserted between them
// (SearchConstants::trailingInsertions) - every one but its last character,
// which is never checked - depends only on them and on the state the
// leading characters left (BatchParams::pruneEntry). So does whether the
// last could still make up the backslashes the rules ask for
// (canReachMinBackslashes_CPU). So for each such state there's a fixed list
// of the groups that survive, and for each of those, which of its rows do: a
// kernel walks the list instead of every group, so that a pruned group costs
// it nothing at all. Leaving the pruned ones out on the GPU instead would
// cost nearly as much as searching them: a warp waits for its slowest lane.
//
// An entry of a list is (group << kRowFlagBits) | classes, sorted by group,
// where bit c of classes says whether the rows whose own last character d is
// of class c (rowClass) survive: the rules tell characters apart only by
// those seven classes, so every character of one gives the same verdict.
// rowMasks()[classes] is the set of d allowed.
constexpr int kRowFlagBits = 7;
constexpr uint32_t kRowFlagCount = 1u << kRowFlagBits;
// Which of the kRowFlagBits classes a character is of, as far as the rules
// go: 0 a letter, digit or space; 1 any other symbol but these: 2 '(', 3 ')',
// 4 '[', 5 ']', 6 '\\'.
int rowClass(char c);
// Group numbers must leave room for the classes: 63^4 groups (trailingLen 6)
// is within this.
constexpr uint64_t kMaxPrunableGroups = uint64_t(1) << (32 - kRowFlagBits);

class RowPruning {
public:
    // The entries of one batch's list: arena()[offset .. offset + count).
    struct Slice {
        uint32_t offset;
        uint32_t count;
    };

    // Called by beginSearch. The lists depend on the alphabet, the rules and
    // the text inserted into the trailing part alone - not the target, the
    // prefix or the suffix - so they're kept for the next search if those are
    // the same, and the backend need not upload them again.
    void begin(const SearchConstants& constants);

    // The surviving groups, among firstGroup..lastGroup, of a batch of
    // trailing length trailingLen that starts in state `entry` - building
    // that state's list, at the end of arena(), the first time it's asked
    // for. Exits the process if a list would be too large (trailingLen past
    // what a GPU backend supports).
    Slice groupsFor(int trailingLen, const PruneState& entry, uint64_t firstGroup, uint64_t lastGroup);

    // Every list built so far, one after another. Only ever grows (until
    // begin() is called with another alphabet or other rules, which clears
    // it), so a backend need only upload what's past the part it has.
    const std::vector<uint32_t>& arena() const { return arena_; }
    // How many times begin() has cleared arena() - a backend that sees this
    // change must upload all of it again.
    uint64_t generation() const { return generation_; }

    // The rows' last characters d allowed by an entry's classes -
    // kRowFlagCount masks, bit d for alphabet position d, never past the
    // alphabet.
    const uint64_t* rowMasks() const { return rowMasks_; }

private:
    std::string alphabet_;
    PruneRules rules_;
    std::vector<TrailingInsertion> insertions_;
    bool begun_ = false;
    uint64_t generation_ = 0;
    uint64_t rowMasks_[kRowFlagCount] = {};
    std::vector<uint32_t> arena_;
    // (trailingLen, and the entry state reduced to what can matter - see
    // groupsFor) -> where its list is in arena_.
    std::map<std::tuple<int, int, int, int, int, bool>, Slice> lists_;

    // The text inserted before trailing character i (1 .. trailingLen - 1),
    // or "".
    const std::string& insertedBefore(int trailingLen, int i) const;
};

#endif // NAMEBREAK_BACKENDS_COMMON_ROW_PRUNING_H
