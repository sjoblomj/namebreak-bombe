#include "backends/common/row_pruning.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {

// The flag bits of kRowFlagBits, and the characters each keeps out of a
// row's last character.
constexpr uint32_t kNoSymbol = 1;        // the symbol run is at 2: no run symbol (isRunSymbol_CPU)
constexpr uint32_t kNoRoundCloser = 2;   // no '(' open: no ')'
constexpr uint32_t kNoSquareCloser = 4;  // no '[' open: no ']'
constexpr uint32_t kNoBackslash = 8;     // every backslash allowed is used, or the last character was one: no '\'
constexpr uint32_t kOnlyBackslash = 16;  // a backslash short of rules.minBackslashCount: only '\'

// The flags of a group whose characters leave `state`: which characters the
// row's own last character may not be. The same verdicts pruneStep_CPU gives
// for that character - it breaks a rule exactly when one of these applies to
// it (a state that got this far has no negative bracket count, and no more
// backslashes than allowed) - and canReachMinBackslashes_CPU after it, with
// the candidate's last character still to come: anything but a backslash
// leaves one character for the backslashes still needed.
uint32_t flagsAfter(const PruneRules& rules, const PruneState& state) {
    uint32_t flags = 0;
    if (rules.symbolRuns && state.symbolRun >= 2)
        flags |= kNoSymbol;
    if (rules.unopenedBrackets && state.open.round == 0)
        flags |= kNoRoundCloser;
    if (rules.unopenedBrackets && state.open.square == 0)
        flags |= kNoSquareCloser;
    if ((rules.maxBackslashCount != 0 && state.backslashes >= rules.maxBackslashCount) || (rules.adjacentBackslashes && state.lastWasBackslash))
        flags |= kNoBackslash;
    if (rules.minBackslashCount != 0 && state.backslashes + 1 < rules.minBackslashCount)
        flags |= kOnlyBackslash;
    return flags;
}

} // namespace

void RowPruning::begin(const SearchConstants& constants) {
    const PruneRules& rules = constants.trailingRules;
    const bool same = begun_ && alphabet_ == constants.alphabet && rules_.symbolRuns == rules.symbolRuns &&
                      rules_.unopenedBrackets == rules.unopenedBrackets && rules_.maxBackslashCount == rules.maxBackslashCount &&
                      rules_.minBackslashCount == rules.minBackslashCount && rules_.adjacentBackslashes == rules.adjacentBackslashes;
    if (same)
        return;
    begun_ = true;
    alphabet_ = constants.alphabet;
    rules_ = rules;
    arena_.clear();
    lists_.clear();
    ++generation_;

    const int alphabetSize = (int) alphabet_.size();
    for (uint32_t flags = 0; flags < kRowFlagCount; ++flags) {
        uint64_t mask = 0;
        for (int d = 0; d < alphabetSize; ++d) {
            const char c = alphabet_[d];
            const bool out = ((flags & kNoSymbol) && isRunSymbol_CPU(c)) || ((flags & kNoRoundCloser) && c == ')') ||
                             ((flags & kNoSquareCloser) && c == ']') || ((flags & kNoBackslash) && c == '\\') ||
                             ((flags & kOnlyBackslash) && c != '\\');
            if (!out)
                mask |= uint64_t(1) << d;
        }
        rowMasks_[flags] = mask;
    }
}

RowPruning::Slice RowPruning::groupsFor(int trailingLen, const PruneState& entry, uint64_t firstGroup, uint64_t lastGroup) {
    const uint64_t alphabetSize = alphabet_.size();
    // The characters checked: the group's trailingLen - 2 and the row's own
    // last one - none with trailingLen 1, when a row has no characters.
    const int groupDigits = std::max(0, trailingLen - 2);
    uint64_t groupCount = 1;
    for (int i = 0; i < groupDigits; ++i)
        groupCount *= alphabetSize;
    if (groupCount > kMaxPrunableGroups) {
        fprintf(stderr, "INTERNAL ERROR: %llu row groups (trailing length %d) are too many to list - exiting\n", (unsigned long long) groupCount,
                trailingLen);
        exit(1);
    }

    // The entry state, reduced to what can make a difference over the at
    // most trailingLen - 1 characters checked: nothing of a rule that's off,
    // and no more open brackets left than there are characters to close
    // them. Likewise the backslashes: when there are too few for
    // trailingLen - 1 characters to reach maxBackslashCount, and enough for
    // minBackslashCount (or the rules are off), how many makes no difference.
    // Every state that reduces to the same key gets the same list, which is
    // built from the key itself.
    const int reach = std::max(1, trailingLen - 1);
    const int run = rules_.symbolRuns ? std::min(entry.symbolRun, 2) : 0;
    const int round = rules_.unopenedBrackets ? std::min(entry.open.round, reach) : 0;
    const int square = rules_.unopenedBrackets ? std::min(entry.open.square, reach) : 0;
    const bool belowMax = rules_.maxBackslashCount == 0 || entry.backslashes <= rules_.maxBackslashCount - reach;
    const bool reachedMin = entry.backslashes >= rules_.minBackslashCount;
    const int backslashes = belowMax && reachedMin ? rules_.minBackslashCount : entry.backslashes;
    const bool lastWasBackslash = rules_.adjacentBackslashes && entry.lastWasBackslash;
    const auto key = std::make_tuple(trailingLen, run, round, square, backslashes, lastWasBackslash);

    auto it = lists_.find(key);
    if (it == lists_.end()) {
        PruneState start;
        start.symbolRun = run;
        start.open.round = round;
        start.open.square = square;
        start.backslashes = backslashes;
        start.lastWasBackslash = lastWasBackslash;
        const Slice list{(uint32_t) arena_.size(), 0};
        if (trailingLen == 1) {
            // A row has no characters of its own: its only group is 0, and
            // nothing but the last character is left to check.
            arena_.push_back(0);
        } else {
            // Every group in order, depth first, leaving out everything
            // under a character that breaks a rule, or leaves too few
            // characters after it for the backslashes still needed.
            std::vector<PruneState> state(groupDigits + 1);
            std::vector<uint64_t> digit(groupDigits, 0);
            state[0] = start;
            int depth = 0;
            uint64_t group = 0;
            while (true) {
                if (depth == groupDigits) {
                    arena_.push_back((uint32_t) (group << kRowFlagBits) | flagsAfter(rules_, state[depth]));
                } else {
                    state[depth + 1] = state[depth];
                    if (pruneStep_CPU(rules_, state[depth + 1], alphabet_[digit[depth]]) &&
                        canReachMinBackslashes_CPU(rules_, state[depth + 1], trailingLen - depth - 1)) {
                        group = group * alphabetSize + digit[depth];
                        ++depth;
                        continue;
                    }
                }
                // Next: the following character at this depth, or up.
                if (depth == groupDigits) {
                    if (depth == 0)
                        break;
                    --depth;
                    group /= alphabetSize;
                }
                while (++digit[depth] == alphabetSize) {
                    digit[depth] = 0;
                    if (depth == 0)
                        goto built;
                    --depth;
                    group /= alphabetSize;
                }
            }
        built:;
        }
        it = lists_.emplace(key, Slice{list.offset, (uint32_t) (arena_.size() - list.offset)}).first;
    }

    // The part of the list in firstGroup..lastGroup.
    const Slice list = it->second;
    const uint32_t* begin = arena_.data() + list.offset;
    const uint32_t* end = begin + list.count;
    const uint32_t* lo = std::lower_bound(begin, end, firstGroup, [](uint32_t e, uint64_t g) { return (e >> kRowFlagBits) < g; });
    const uint32_t* hi = std::upper_bound(lo, end, lastGroup, [](uint64_t g, uint32_t e) { return g < (e >> kRowFlagBits); });
    return Slice{(uint32_t) (lo - arena_.data()), (uint32_t) (hi - lo)};
}
