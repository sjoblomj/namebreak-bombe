#include "backends/common/row_pruning.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

int rowClass(char c) {
    switch (c) {
        case '(': return 2;
        case ')': return 3;
        case '[': return 4;
        case ']': return 5;
        case '\\': return 6;
        default: return isRunSymbol_CPU(c) ? 1 : 0;
    }
}

namespace {

// Every class: the rows of a group a list entry can have.
constexpr uint32_t kAllClasses = kRowFlagCount - 1;

// Steps `state` over every character of `text`; false if one breaks a rule.
bool stepOver(const PruneRules& rules, PruneState& state, const std::string& text) {
    for (char c : text)
        if (!pruneStep_CPU(rules, state, c))
            return false;
    return true;
}

} // namespace

void RowPruning::begin(const SearchConstants& constants) {
    const PruneRules& rules = constants.trailingRules;
    const bool same = begun_ && alphabet_ == constants.alphabet && rules_.symbolRuns == rules.symbolRuns &&
                      rules_.unopenedBrackets == rules.unopenedBrackets && rules_.maxBackslashCount == rules.maxBackslashCount &&
                      rules_.minBackslashCount == rules.minBackslashCount && rules_.adjacentBackslashes == rules.adjacentBackslashes &&
                      insertions_ == constants.trailingInsertions;
    if (same)
        return;
    begun_ = true;
    alphabet_ = constants.alphabet;
    rules_ = rules;
    insertions_ = constants.trailingInsertions;
    arena_.clear();
    lists_.clear();
    ++generation_;

    const int alphabetSize = (int) alphabet_.size();
    for (uint32_t classes = 0; classes < kRowFlagCount; ++classes) {
        uint64_t mask = 0;
        for (int d = 0; d < alphabetSize; ++d)
            if (classes >> rowClass(alphabet_[d]) & 1)
                mask |= uint64_t(1) << d;
        rowMasks_[classes] = mask;
    }
}

const std::string& RowPruning::insertedBefore(int trailingLen, int i) const {
    static const std::string none;
    for (const TrailingInsertion& insertion : insertions_)
        if (trailingLen - insertion.charsAfter == i)
            return insertion.text;
    return none;
}

RowPruning::Slice RowPruning::groupsFor(int trailingLen, const PruneState& entry, uint64_t firstGroup, uint64_t lastGroup) {
    const uint64_t alphabetSize = alphabet_.size();
    // The characters checked: the group's trailingLen - 2 and the row's own
    // last one, and the text inserted before each but the first and after
    // the last - none with trailingLen 1, when a row has no characters.
    const int groupDigits = std::max(0, trailingLen - 2);
    uint64_t groupCount = 1;
    for (int i = 0; i < groupDigits; ++i)
        groupCount *= alphabetSize;
    if (groupCount > kMaxPrunableGroups) {
        fprintf(stderr, "INTERNAL ERROR: %llu row groups (trailing length %d) are too many to list - exiting\n", (unsigned long long) groupCount,
                trailingLen);
        exit(1);
    }
    int inserted = 0;
    for (const TrailingInsertion& insertion : insertions_)
        inserted += (int) insertion.text.size();

    // The entry state, reduced to what can make a difference over the at
    // most trailingLen - 1 characters checked, and the text inserted between
    // them: nothing of a rule that's off, and no more open brackets left
    // than there are characters to close them. Likewise the backslashes:
    // when there are too few for those characters to reach
    // maxBackslashCount, and enough for minBackslashCount (or the rules are
    // off), how many makes no difference. Every state that reduces to the
    // same key gets the same list, which is built from the key itself.
    const int reach = std::max(1, trailingLen - 1 + inserted);
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
            arena_.push_back(kAllClasses);
        } else {
            // What the trailing characters from i on, and the text inserted
            // between them, can add to the backslashes (see
            // canReachMinBackslashes_CPU) - for every i a group's characters
            // or a row's own can leave to come.
            std::vector<TailCapacity> capacityFrom(trailingLen);
            for (int i = 1; i < trailingLen; ++i) {
                std::string tail;
                for (int t = i; t < trailingLen; ++t) {
                    if (t > i)
                        tail += insertedBefore(trailingLen, t);
                    tail += kFreeChar;
                }
                capacityFrom[i] = tailCapacity_CPU(rules_, tail);
            }
            // A representative of each class the alphabet has.
            int representative[kRowFlagBits];
            std::fill(representative, representative + kRowFlagBits, -1);
            for (int d = (int) alphabetSize - 1; d >= 0; --d)
                representative[rowClass(alphabet_[d])] = d;
            // Which classes of the row's own last character survive after a
            // group's characters leave `state`: those that break no rule,
            // nor does the text inserted after them, and leave the last
            // character room for the backslashes still needed.
            const std::string& afterRow = insertedBefore(trailingLen, trailingLen - 1);
            auto rowClasses = [&](const PruneState& state) {
                uint32_t classes = 0;
                for (int c = 0; c < kRowFlagBits; ++c) {
                    PruneState s = state;
                    if (representative[c] >= 0 && pruneStep_CPU(rules_, s, alphabet_[representative[c]]) && stepOver(rules_, s, afterRow) &&
                        canReachMinBackslashes_CPU(rules_, s, capacityFrom[trailingLen - 1]))
                        classes |= 1u << c;
                }
                return classes;
            };

            // Every group in order, depth first, leaving out everything
            // under a character that breaks a rule - or the text inserted
            // after it does - or that leaves too few characters after it for
            // the backslashes still needed, and every group none of whose
            // rows survive. What a character does depends only on its class
            // (rowClass), so that's worked out once per class at each
            // depth: it makes building the lists several times faster.
            auto visit = [&](auto&& self, int depth, const PruneState& state, uint64_t group) -> void {
                struct Child {
                    bool known = false;
                    bool ok = false;
                    uint32_t classes = 0; // the last group character's: its rows' classes
                    PruneState state;
                };
                Child child[kRowFlagBits];
                const bool last = depth + 1 == groupDigits;
                for (uint64_t d = 0; d < alphabetSize; ++d) {
                    Child& c = child[rowClass(alphabet_[d])];
                    if (!c.known) {
                        c.known = true;
                        c.state = state;
                        c.ok = pruneStep_CPU(rules_, c.state, alphabet_[d]) && stepOver(rules_, c.state, insertedBefore(trailingLen, depth + 1)) &&
                               canReachMinBackslashes_CPU(rules_, c.state, capacityFrom[depth + 1]);
                        if (c.ok && last)
                            c.classes = rowClasses(c.state);
                    }
                    if (!c.ok)
                        continue;
                    const uint64_t next = group * alphabetSize + d;
                    if (!last)
                        self(self, depth + 1, c.state, next);
                    else if (c.classes != 0)
                        arena_.push_back((uint32_t) (next << kRowFlagBits) | c.classes);
                }
            };
            if (groupDigits == 0) {
                const uint32_t classes = rowClasses(start);
                if (classes != 0)
                    arena_.push_back(classes);
            } else {
                visit(visit, 0, start, 0);
            }
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
