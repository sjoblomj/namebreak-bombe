#include "engine/dictionary.h"

#include <algorithm>
#include <limits>
#include <set>

#include "engine/wordlist.h"

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

// Every string of `minLength` to `maxLength` characters of `alphabet`,
// shortest first - unless there are more than `limit`.
bool stringsOf(const std::string& alphabet, int minLength, int maxLength, uint64_t limit, std::vector<std::string>& out) {
    uint64_t count = 0, ofLength = 1;
    for (int length = 0; length <= maxLength; ++length) {
        if (length >= minLength)
            count += ofLength;
        if (count > limit || !multiply(ofLength, alphabet.size(), ofLength))
            return false;
    }
    out.clear();
    std::vector<std::string> level = {""};
    for (int length = 0; length <= maxLength; ++length) {
        if (length >= minLength)
            out.insert(out.end(), level.begin(), level.end());
        if (length == maxLength)
            break;
        std::vector<std::string> next;
        for (const std::string& s : level) {
            for (char c : alphabet)
                next.push_back(s + c);
        }
        level = std::move(next);
    }
    return true;
}

// The strings one tail element stands for (see expandDictionaryTails).
bool tailElementOptions(const std::string& element, std::vector<std::string>& out, std::string& error) {
    for (const auto& [name, alphabet] : {std::pair<std::string, std::string>{"digits:", "0123456789"},
                                         std::pair<std::string, std::string>{"letters:", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}}) {
        if (element.compare(0, name.size(), name) != 0)
            continue;
        const std::string range = element.substr(name.size());
        const size_t dash = range.find('-');
        const std::string lowText = range.substr(0, dash), highText = dash == std::string::npos ? lowText : range.substr(dash + 1);
        auto number = [](const std::string& text, int& value) {
            if (text.empty() || text.size() > 2 || text.find_first_not_of("0123456789") != std::string::npos)
                return false;
            value = std::stoi(text);
            return true;
        };
        int low = 0, high = 0;
        if (!number(lowText, low) || !number(highText, high) || low > high) {
            error = "invalid tail element '" + element + "' (expected " + name + "N or " + name + "A-B, with A <= B)";
            return false;
        }
        if (high > kMaxDictionaryTailLength) {
            error = "tail element '" + element + "' makes tails longer than " + std::to_string(kMaxDictionaryTailLength) + " characters";
            return false;
        }
        if (!stringsOf(alphabet, low, high, kMaxDictionaryTails, out)) {
            error = "tail element '" + element + "' makes more than " + std::to_string(kMaxDictionaryTails) + " tails";
            return false;
        }
        return true;
    }
    // One of some literal strings, separated by '|' - but no name has a ':',
    // so one with a ':' is a misspelled digits: or letters:.
    if (element.find(':') != std::string::npos) {
        error = "unknown tail element '" + element + "' (expected digits:A-B, letters:A-B, or literal strings separated by '|')";
        return false;
    }
    out.clear();
    size_t start = 0;
    for (;;) {
        const size_t bar = element.find('|', start);
        out.push_back(element.substr(start, bar == std::string::npos ? std::string::npos : bar - start));
        if (bar == std::string::npos)
            break;
        start = bar + 1;
    }
    return true;
}

} // namespace

bool expandDictionaryTails(const std::vector<std::string>& elements, std::vector<std::string>& tails, std::string& error) {
    std::vector<std::string> made = {""};
    for (const std::string& element : elements) {
        std::vector<std::string> options;
        if (!tailElementOptions(element, options, error))
            return false;
        uint64_t count = 0;
        if (!multiply(made.size(), options.size(), count) || count > kMaxDictionaryTails) {
            error = "the tails are more than " + std::to_string(kMaxDictionaryTails) + " - use fewer, or shorter";
            return false;
        }
        std::vector<std::string> next;
        next.reserve((size_t) count);
        for (const std::string& a : made) {
            for (const std::string& b : options)
                next.push_back(a + b);
        }
        made = std::move(next);
    }
    for (std::string& tail : made) {
        tail = normalizeMpqName(tail);
        if (tail.size() > (size_t) kMaxDictionaryTailLength) {
            error = "the tail '" + tail + "' is longer than " + std::to_string(kMaxDictionaryTailLength) + " characters";
            return false;
        }
        for (char c : tail) {
            if (c == '\\' || (unsigned char) c < 0x20 || (unsigned char) c > 0x7E) {
                error = "the tail '" + tail + "' has a '\\' or a character that isn't printable ASCII - a tail is part of a basename";
                return false;
            }
        }
    }
    tails = sortedUniqueWords(std::move(made));
    return true;
}

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
    if (pattern.tails.empty() || pattern.tails.size() > kMaxDictionaryTails) {
        error = "the tails must be 1 to " + std::to_string(kMaxDictionaryTails) + " - give \"\" for none";
        return false;
    }
    for (size_t i = 0; i < pattern.tails.size(); ++i) {
        if ((i > 0 && !(pattern.tails[i - 1] < pattern.tails[i])) || pattern.tails[i].size() > (size_t) kMaxDictionaryTailLength ||
            pattern.tails[i].find('\\') != std::string::npos) {
            error = "the tails must be sorted, without duplicates, at most " + std::to_string(kMaxDictionaryTailLength) +
                    " characters each, and without a '\\'";
            return false;
        }
    }
    if (pattern.minWords < 1 || pattern.maxWords < pattern.minWords || pattern.maxWords > kMaxDictionaryWords) {
        error = "the word counts must be 1 <= min_words <= max_words <= " + std::to_string(kMaxDictionaryWords) + " (got " +
                std::to_string(pattern.minWords) + " to " + std::to_string(pattern.maxWords) + ")";
        return false;
    }

    DictionarySpace space;
    space.blockStart_.assign(pattern.maxWords + 1, 0);
    space.blockSize_.assign(pattern.maxWords + 1, 0);
    const uint64_t w = pattern.words.size(), s = pattern.separators.size(), t = pattern.tails.size();
    uint64_t total = 0;
    for (int k = pattern.minWords; k <= pattern.maxWords; ++k) {
        // W^k * S^(k-1) * T
        uint64_t size = w;
        bool fits = true;
        for (int i = 1; i < k && fits; ++i)
            fits = multiply(size, w, size) && multiply(size, s, size);
        if (!fits || !multiply(size, t, size)) {
            error = std::to_string(k) + " words of " + std::to_string(w) + ", with " + std::to_string(s) + " separators" +
                    (t > 1 ? " and " + std::to_string(t) + " tails" : "") + ", are more candidates than 64 bits can number - use fewer words";
            return false;
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
    const uint64_t w = pattern_.words.size(), s = pattern_.separators.size(), t = pattern_.tails.size();

    DictionaryChoice choice;
    choice.words.assign(k, 0);
    choice.separators.assign(k - 1, 0);
    // The wheels from the right: the tail, the last word, then separator
    // k-1 and word k-1, and so on.
    choice.tail = (uint32_t) (local % t);
    local /= t;
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
    return blockStart_[k] + local * pattern_.tails.size() + choice.tail;
}

std::string DictionarySpace::text(const DictionaryChoice& choice) const {
    std::string out = pattern_.words[choice.words[0]];
    for (size_t i = 1; i < choice.words.size(); ++i)
        out += pattern_.separators[choice.separators[i - 1]] + pattern_.words[choice.words[i]];
    return out + pattern_.tails[choice.tail];
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
