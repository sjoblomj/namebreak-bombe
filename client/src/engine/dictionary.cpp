#include "engine/dictionary.h"

#include <algorithm>
#include <limits>
#include <set>

namespace {

// a * b, or false if it doesn't fit.
bool multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        return false;
    out = a * b;
    return true;
}

// Where `start` and `bound` first differ: -1 if `start` sorts first there, 1
// if `bound` does, 0 if they don't differ on their common length.
int compareCommon(const std::string& start, const std::string& bound) {
    const size_t n = std::min(start.size(), bound.size());
    for (size_t i = 0; i < n; ++i) {
        const unsigned char a = (unsigned char) start[i], b = (unsigned char) bound[i];
        if (a != b)
            return a < b ? -1 : 1;
    }
    return 0;
}

} // namespace

bool DictionarySpace::create(DictionaryPattern pattern, DictionarySpace& out, std::string& error) {
    if (pattern.words.empty()) {
        error = "no words to search";
        return false;
    }
    for (size_t i = 1; i < pattern.words.size(); ++i) {
        if (!(pattern.words[i - 1] < pattern.words[i])) {
            error = "the words must be sorted and without duplicates";
            return false;
        }
    }
    if (pattern.words.size() > std::numeric_limits<uint32_t>::max()) {
        error = "too many words";
        return false;
    }
    if (pattern.separators.empty()) {
        error = "no separators - give \"\" for words written together";
        return false;
    }
    if (std::set<std::string>(pattern.separators.begin(), pattern.separators.end()).size() != pattern.separators.size()) {
        error = "a separator is listed more than once";
        return false;
    }
    if (pattern.minWords < 1 || pattern.maxWords < pattern.minWords || pattern.maxWords > kMaxDictionaryWords) {
        error = "the word counts must be 1 <= min_words <= max_words <= " + std::to_string(kMaxDictionaryWords) + " (got " +
                std::to_string(pattern.minWords) + " to " + std::to_string(pattern.maxWords) + ")";
        return false;
    }

    DictionarySpace space;
    space.blockStart_.assign(pattern.maxWords + 1, 0);
    space.blockSize_.assign(pattern.maxWords + 1, 0);
    const uint64_t w = pattern.words.size(), s = pattern.separators.size();
    uint64_t total = 0;
    for (int k = pattern.minWords; k <= pattern.maxWords; ++k) {
        // W^k * S^(k-1)
        uint64_t size = w;
        for (int i = 1; i < k; ++i) {
            if (!multiply(size, w, size) || !multiply(size, s, size)) {
                error = std::to_string(k) + " words of " + std::to_string(w) + ", with " + std::to_string(s) +
                        " separators, are more candidates than 64 bits can number - use fewer words";
                return false;
            }
        }
        if (total > std::numeric_limits<uint64_t>::max() - size) {
            error = "more candidates than 64 bits can number - use fewer words";
            return false;
        }
        space.blockStart_[k] = total;
        space.blockSize_[k] = size;
        total += size;
    }
    space.size_ = total;
    space.pattern_ = std::move(pattern);
    out = std::move(space);
    return true;
}

DictionaryChoice DictionarySpace::decode(uint64_t number) const {
    int k = pattern_.minWords;
    while (number >= blockStart_[k] + blockSize_[k])
        ++k;
    uint64_t local = number - blockStart_[k];
    const uint64_t w = pattern_.words.size(), s = pattern_.separators.size();

    DictionaryChoice choice;
    choice.words.assign(k, 0);
    choice.separators.assign(k - 1, 0);
    // The wheels from the right: the last word, then separator k-1 and word
    // k-1, and so on.
    choice.words[k - 1] = (uint32_t) (local % w);
    local /= w;
    for (int i = k - 2; i >= 0; --i) {
        choice.separators[i] = (uint32_t) (local % s);
        local /= s;
        choice.words[i] = (uint32_t) (local % w);
        local /= w;
    }
    return choice;
}

uint64_t DictionarySpace::encode(const DictionaryChoice& choice) const {
    const int k = (int) choice.words.size();
    const uint64_t w = pattern_.words.size(), s = pattern_.separators.size();
    uint64_t local = choice.words[0];
    for (int i = 1; i < k; ++i)
        local = (local * s + choice.separators[i - 1]) * w + choice.words[i];
    return blockStart_[k] + local;
}

std::string DictionarySpace::text(const DictionaryChoice& choice) const {
    std::string out = pattern_.words[choice.words[0]];
    for (size_t i = 1; i < choice.words.size(); ++i)
        out += pattern_.separators[choice.separators[i - 1]] + pattern_.words[choice.words[i]];
    return out;
}

bool FilenameBounds::contains(const std::string& filename) const {
    return (!hasLower || !(filename < lower)) && (!hasUpper || !(upper < filename));
}

FilenameBounds::Verdict FilenameBounds::classifyStart(const std::string& start) const {
    // Every filename f that starts with `start`, against each bound.
    bool lowerSettled = !hasLower; // every f >= lower
    if (hasLower) {
        const int c = compareCommon(start, lower);
        if (c < 0)
            return Verdict::Outside; // every f < lower
        // Differing upwards, or lower is a prefix of start (and so of f).
        lowerSettled = c > 0 || start.size() >= lower.size();
    }
    bool upperSettled = !hasUpper; // every f <= upper
    if (hasUpper) {
        const int c = compareCommon(start, upper);
        if (c > 0)
            return Verdict::Outside; // every f > upper
        if (c == 0 && start.size() > upper.size())
            return Verdict::Outside; // upper is a proper prefix of every f
        upperSettled = c < 0;
    }
    return lowerSettled && upperSettled ? Verdict::Inside : Verdict::Mixed;
}
