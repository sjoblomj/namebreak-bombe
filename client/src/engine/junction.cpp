#include "engine/junction.h"

#include <algorithm>
#include <utility>

namespace {

// base^exponent, or UINT64_MAX if that doesn't fit in 64 bits.
uint64_t powerOrMax(uint64_t base, int exponent) {
    uint64_t result = 1;
    for (int i = 0; i < exponent; ++i) {
        if (base != 0 && result > UINT64_MAX / base)
            return UINT64_MAX;
        result *= base;
    }
    return result;
}

// a + b, or UINT64_MAX if that doesn't fit.
uint64_t sumOrMax(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

// How many bits of filter a table of `keys` keys gets: about
// kFilterBitsPerKey each, rounded up to a power of two (the filter is
// indexed by the top bits of a mixed key), and at least one word. Capped at
// 2^32 bits (512 MB), which is past where a table's keys themselves would
// have run a machine out of memory.
uint64_t filterBitsFor(uint64_t keys) {
    const uint64_t wanted = keys > UINT64_MAX / JunctionTable::kFilterBitsPerKey ? UINT64_MAX : keys * JunctionTable::kFilterBitsPerKey;
    uint64_t bits = 64;
    while (bits < wanted && bits < ((uint64_t) 1 << 32))
        bits *= 2;
    return bits;
}

} // namespace

HashState rewindHash(HashState state, const std::string& text, int offset, const uint32_t* cryptTable) {
    for (size_t i = text.size(); i-- > 0;)
        state = hashStepBack(state, (unsigned char) text[i], cryptTable + offset);
    return state;
}

TailPruneState tailPruneStart(const std::string& suffix) {
    TailPruneState state;
    state.nextIsBackslash = !suffix.empty() && suffix.front() == '\\';
    return state;
}

bool tailPruneStep(const PruneRules& rules, TailPruneState& state, char c) {
    // The same rules as pruneStep_CPU (candidate.h), in the mirror: a run of
    // symbols is a run either way round, and so are two backslashes next to
    // each other - only the side the state comes from differs.
    if (isRunSymbol_CPU(c)) {
        if (++state.symbolRun >= 3 && rules.symbolRuns)
            return false;
        state.symbolRun = std::min(state.symbolRun, 2);
    } else {
        state.symbolRun = 0;
    }
    if (c == '\\') {
        if (state.nextIsBackslash && rules.adjacentBackslashes)
            return false;
        if (++state.backslashes > rules.maxBackslashCount && rules.maxBackslashCount != 0)
            return false;
    }
    state.nextIsBackslash = c == '\\';
    return true;
}

PruneState dirPruneStart(const std::string& prefix) {
    PruneState state;
    state.open = openBracketsAfter_CPU(prefix);
    state.lastWasBackslash = !prefix.empty() && prefix.back() == '\\';
    return state;
}

bool dirPasses(const PruneRules& rules, const std::string& prefix, const std::string& suffix, const std::string& dir) {
    if (!rules.any())
        return true;
    PruneState state = dirPruneStart(prefix);
    for (char c : dir) {
        if (!pruneStep_CPU(rules, state, c))
            return false;
    }
    // A '\' the directory ends with is next to one the suffix begins with -
    // the same two-in-a-row the rule is about, across the join.
    if (rules.adjacentBackslashes && state.lastWasBackslash && !suffix.empty() && suffix.front() == '\\')
        return false;
    // And nothing is left to make up a shortfall of backslashes.
    return canReachMinBackslashes_CPU(rules, state, 0);
}

// --- JunctionSpace -----------------------------------------------------------

bool JunctionSpace::create(const std::string& alphabet, int minLength, int maxLength, int tableLength, JunctionSpace& out,
                           std::string& error) {
    if (alphabet.empty()) {
        error = "the alphabet is empty";
        return false;
    }
    if ((int) alphabet.size() > MAX_ALPHABET_SIZE) {
        error = "the alphabet has " + std::to_string(alphabet.size()) + " characters - at most " + std::to_string(MAX_ALPHABET_SIZE);
        return false;
    }
    for (size_t i = 0; i < alphabet.size(); ++i) {
        if (alphabet.find(alphabet[i], i + 1) != std::string::npos) {
            error = std::string("the alphabet has '") + alphabet[i] + "' more than once";
            return false;
        }
    }
    if (minLength < 1) {
        error = "the shortest directory must be at least one character";
        return false;
    }
    if (maxLength < minLength) {
        error = "the longest directory (" + std::to_string(maxLength) + " characters) is shorter than the shortest (" +
                std::to_string(minLength) + ")";
        return false;
    }
    if (maxLength > MAX_CANDIDATE_LEN) {
        error = "the longest directory is " + std::to_string(maxLength) + " characters - at most " + std::to_string(MAX_CANDIDATE_LEN);
        return false;
    }
    if (tableLength < 0 || tableLength > maxLength) {
        error = "the table's " + std::to_string(tableLength) + " characters must be between 0 and the longest directory's " +
                std::to_string(maxLength);
        return false;
    }

    out = JunctionSpace();
    out.alphabet_ = alphabet;
    out.minLength_ = minLength;
    out.maxLength_ = maxLength;
    out.tableLength_ = tableLength;
    const int maxTail = maxLength - tableLength;
    out.counts_.resize(maxTail + 1);
    out.firsts_.resize(maxTail + 2);
    uint64_t total = 0;
    for (int j = 0; j <= maxTail; ++j) {
        out.counts_[j] = powerOrMax(alphabet.size(), j);
        out.firsts_[j] = total;
        total = sumOrMax(total, out.counts_[j]);
        if (total == UINT64_MAX) {
            error = "too many tails to number: " + std::to_string(alphabet.size()) + " characters to the power of " +
                    std::to_string(maxTail) + " (the table's " + std::to_string(tableLength) +
                    " characters leave that many of a directory's " + std::to_string(maxLength) +
                    " to search) doesn't fit in 64 bits - search shorter directories, or give the table more memory";
            return false;
        }
    }
    out.firsts_[maxTail + 1] = total;
    out.size_ = total;
    return true;
}

int JunctionSpace::tailLengthOf(uint64_t number) const {
    // At most MAX_CANDIDATE_LEN + 1 lengths, so a walk up them beats the
    // binary search it would take to be clever.
    for (int j = 0; j <= maxTailLength(); ++j) {
        if (number < firsts_[j + 1])
            return j;
    }
    return maxTailLength();
}

uint64_t JunctionSpace::indexOf(uint64_t number) const {
    return number - firsts_[tailLengthOf(number)];
}

uint64_t JunctionSpace::numberOf(int tailLength, uint64_t index) const {
    return firsts_[tailLength] + index;
}

std::string JunctionSpace::tailOf(uint64_t number) const {
    const int tailLength = tailLengthOf(number);
    // The index is the reversed tail's: the directory's last character is the
    // most significant digit, because it's the first the walk rewinds.
    std::string reversed = indexToString(indexOf(number), tailLength, alphabet_);
    return std::string(reversed.rbegin(), reversed.rend());
}

bool JunctionSpace::indexOfTail(const std::string& tail, uint64_t& out, std::string& error) const {
    return stringToIndex(std::string(tail.rbegin(), tail.rend()), alphabet_, out, error);
}

uint64_t junctionTableCount(int alphabetSize, int tableLength) {
    return powerOrMax((uint64_t) alphabetSize, tableLength);
}

int junctionTableLengthForBytes(int alphabetSize, int maxLength, uint64_t bytes) {
    for (int tableLength = maxLength; tableLength > 0; --tableLength) {
        if (JunctionTable::bytesFor(alphabetSize, tableLength) <= bytes)
            return tableLength;
    }
    return 0;
}

// --- JunctionTable -----------------------------------------------------------

uint64_t JunctionTable::bytesFor(int alphabetSize, int tableLength) {
    const uint64_t keys = junctionTableCount(alphabetSize, tableLength);
    if (keys == UINT64_MAX || keys > UINT64_MAX / sizeof(uint64_t))
        return UINT64_MAX;
    return sumOrMax(keys * sizeof(uint64_t), filterBitsFor(keys) / 8);
}

namespace {

// The walk the table is built by: every directory of up to tableLength()
// characters the rules allow, in alphabet order, with the hash state it
// reaches. A plain recursion, at most MAX_CANDIDATE_LEN deep.
struct TableBuilder {
    const JunctionSpace& space;
    const uint32_t* table; // cryptTable + kHashAOffset
    const PruneRules& rules;
    uint64_t wanted;                        // the junction's key, for the shorter directories
    std::vector<std::string>* shortMatches; // every shorter directory that has it
    std::vector<uint64_t>* keys;            // every full-length directory's key
    std::string dir;                        // the characters so far

    void walk(int depth, HashState state, const PruneState& prune) {
        if (depth == space.tableLength()) {
            keys->push_back(junctionKey(state));
            return;
        }
        for (char c : space.alphabet()) {
            PruneState next = prune;
            if (rules.any() && !pruneStep_CPU(rules, next, c))
                continue;
            const HashState stepped = hashStep(state, (unsigned char) c, table);
            dir.push_back(c);
            // A directory shorter than the table's own length: nothing else
            // in the search would ever look at it, so it's checked here. The
            // ones of exactly the table's length are the table's own, found
            // through it by the walk's empty tail.
            if ((int) dir.size() < space.tableLength() && (int) dir.size() >= space.minLength() && junctionKey(stepped) == wanted)
                shortMatches->push_back(dir);
            walk(depth + 1, stepped, next);
            dir.pop_back();
        }
    }
};

} // namespace

bool JunctionTable::build(const JunctionSpace& space, HashState startState, const PruneState& prefixEntry, const PruneRules& rules,
                          const uint32_t* cryptTable, HashState junction, std::vector<std::string>& shortMatches, JunctionTable& out,
                          std::string& error) {
    const uint64_t room = junctionTableCount((int) space.alphabet().size(), space.tableLength());
    if (bytesFor((int) space.alphabet().size(), space.tableLength()) == UINT64_MAX) {
        error = "a table of " + std::to_string(space.tableLength()) + " characters is far too big for any machine's memory";
        return false;
    }

    out = JunctionTable();
    out.space_ = space;
    out.startState_ = startState;
    out.prefixEntry_ = prefixEntry;
    out.rules_ = rules;
    out.cryptTable_ = cryptTable;
    try {
        // Room for every directory of the table's length, pruned or not: one
        // allocation, of a size the caller was told about in advance
        // (bytesFor), rather than a growing one that would need half as much
        // again at its peak.
        out.keys_.reserve(room);
    } catch (const std::bad_alloc&) {
        error = "not enough memory for a table of " + std::to_string(space.tableLength()) + " characters (" +
                std::to_string(bytesFor((int) space.alphabet().size(), space.tableLength()) / (1024 * 1024)) + " MB)";
        return false;
    }

    TableBuilder builder{space, cryptTable + kHashAOffset, rules, junctionKey(junction), &shortMatches, &out.keys_, std::string()};
    builder.dir.reserve(space.tableLength());
    builder.walk(0, startState, prefixEntry);
    std::sort(out.keys_.begin(), out.keys_.end());

    out.filterBits_ = filterBitsFor(out.keys_.size());
    int bits = 0;
    while (((uint64_t) 1 << bits) < out.filterBits_)
        ++bits;
    out.filterShift_ = 64 - bits;
    try {
        out.filter_.assign(out.filterBits_ / 64, 0);
    } catch (const std::bad_alloc&) {
        error = "not enough memory for the table's filter (" + std::to_string(out.filterBits_ / 8 / (1024 * 1024)) + " MB)";
        return false;
    }
    for (uint64_t key : out.keys_) {
        const uint64_t bit = out.filterBit(key);
        out.filter_[bit >> 6] |= (uint64_t) 1 << (bit & 63);
    }
    return true;
}

bool JunctionTable::holds(uint64_t key) const {
    return std::binary_search(keys_.begin(), keys_.end(), key);
}

void JunctionTable::prefixesFor(const std::vector<uint64_t>& keys, std::vector<std::vector<std::string>>& out) const {
    out.assign(keys.size(), {});
    if (keys.empty())
        return;
    // The keys asked about, sorted, each with where its answer goes - so one
    // walk of the table's space can serve them all, with a binary search per
    // directory it reaches.
    std::vector<std::pair<uint64_t, size_t>> wanted;
    wanted.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
        wanted.emplace_back(keys[i], i);
    std::sort(wanted.begin(), wanted.end());

    // The same walk the table was built by, reporting the directories whose
    // key is one of these instead of keeping every key.
    struct Resolver {
        const JunctionSpace& space;
        const uint32_t* table;
        const PruneRules& rules;
        const std::vector<std::pair<uint64_t, size_t>>& wanted;
        std::vector<std::vector<std::string>>& out;
        std::string dir;

        void walk(int depth, HashState state, const PruneState& prune) {
            if (depth == space.tableLength()) {
                const uint64_t key = junctionKey(state);
                auto it = std::lower_bound(wanted.begin(), wanted.end(), std::make_pair(key, (size_t) 0));
                for (; it != wanted.end() && it->first == key; ++it)
                    out[it->second].push_back(dir);
                return;
            }
            for (char c : space.alphabet()) {
                PruneState next = prune;
                if (rules.any() && !pruneStep_CPU(rules, next, c))
                    continue;
                dir.push_back(c);
                walk(depth + 1, hashStep(state, (unsigned char) c, table), next);
                dir.pop_back();
            }
        }
    };
    Resolver resolver{space_, cryptTable_ + kHashAOffset, rules_, wanted, out, std::string()};
    resolver.dir.reserve(space_.tableLength());
    resolver.walk(0, startState_, prefixEntry_);
}

void JunctionTable::prefixesFor(uint64_t key, std::vector<std::string>& out) const {
    std::vector<std::vector<std::string>> all;
    prefixesFor(std::vector<uint64_t>{key}, all);
    out = std::move(all.front());
}

// --- JunctionWalk ------------------------------------------------------------

JunctionWalk::JunctionWalk(std::string alphabet, int tailLength, HashState junction, const uint32_t* table, const PruneRules& rules,
                           const TailPruneState& entry)
    : alphabet_(std::move(alphabet)), table_(table), rules_(rules), tailLength_(tailLength) {
    tail_.assign(tailLength_, alphabet_.empty() ? '\0' : alphabet_[0]);
    digit_.assign(tailLength_, 0);
    place_.assign(tailLength_ + 1, 1);
    for (int k = tailLength_ - 1; k >= 0; --k)
        place_[k] = place_[k + 1] * alphabet_.size();
    stack_.assign(tailLength_ + 1, junction);
    prune_.assign(tailLength_ + 1, entry);
}

bool JunctionWalk::start(uint64_t from, uint64_t to) {
    end_ = std::min(to, place_[0]);
    if (from >= end_)
        return false;
    index_ = from;
    uint64_t rest = from;
    for (int k = 0; k < tailLength_; ++k) {
        digit_[k] = (int) (rest / place_[k + 1]);
        rest -= (uint64_t) digit_[k] * place_[k + 1];
        tail_[tailLength_ - 1 - k] = alphabet_[digit_[k]];
    }
    changedFrom_ = 0;
    return settle();
}

bool JunctionWalk::next() {
    return skip(tailLength_) && settle();
}

bool JunctionWalk::settle() {
    for (;;) {
        const int reached = rebuildFrom(changedFrom_);
        if (reached == tailLength_)
            return true;
        // The character at `reached` broke a rule, and so does every tail
        // that ends in these ones: all of them are skipped at once.
        if (!skip(reached + 1))
            return false;
    }
}

bool JunctionWalk::stepInto(int depth) {
    const char c = alphabet_[digit_[depth]];
    prune_[depth + 1] = prune_[depth];
    if (rules_.any() && !tailPruneStep(rules_, prune_[depth + 1], c))
        return false;
    stack_[depth + 1] = hashStepBack(stack_[depth], (unsigned char) c, table_);
    return true;
}

int JunctionWalk::rebuildFrom(int depth) {
    for (int k = depth; k < tailLength_; ++k) {
        if (!stepInto(k))
            return k;
    }
    return tailLength_;
}

bool JunctionWalk::skip(int depth) {
    // Past every tail that shares this one's first `depth` rewound
    // characters: place_[depth] of them, and the index is somewhere among
    // them.
    const uint64_t step = place_[depth] - index_ % place_[depth];
    if (step >= end_ - index_)
        return false;
    index_ += step;
    // The digits move with it: everything from `depth` on back to the
    // alphabet's first character, and the one before them up by one,
    // carrying left.
    for (int k = depth; k < tailLength_; ++k) {
        digit_[k] = 0;
        tail_[tailLength_ - 1 - k] = alphabet_[0];
    }
    int p = depth - 1;
    while (p >= 0) {
        if (++digit_[p] < (int) alphabet_.size()) {
            tail_[tailLength_ - 1 - p] = alphabet_[digit_[p]];
            break;
        }
        digit_[p] = 0;
        tail_[tailLength_ - 1 - p] = alphabet_[0];
        --p;
    }
    // p < 0 would mean every digit wrapped - the index past the last tail,
    // which the range check above has already caught.
    changedFrom_ = p < 0 ? 0 : p;
    return true;
}
