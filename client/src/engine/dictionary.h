#ifndef NAMEBREAK_ENGINE_DICTIONARY_H
#define NAMEBREAK_ENGINE_DICTIONARY_H

#include <cstdint>
#include <string>
#include <vector>

// The candidates of a dictionary search (dictionary_search.h), and their
// numbers. A candidate is one to maxWords words, with a separator between
// each two: word, separator, word, ... - every word from the same list,
// every separator from the same list. Its filename is prefix + candidate +
// suffix.
//
// Every candidate has a number, counting from 0, the way an odometer counts
// with a wheel per word and per separator: all one-word candidates first
// (if minWords is 1), then all two-word ones, and so on. Within a word
// count, the wheels from the left are word 1, separator 1, word 2, ...,
// word k, and the last word turns fastest:
//
//   number - start of the k-word block = ((w1 * S + s1) * W + w2) * S + s2 ... * W + wk
//
// with W words and S separators. The numbers only depend on the words,
// separators and word counts - never on the bounds, the prefix or the
// suffix - so a search can be split up, or resumed, by number.

// The most words a candidate can have.
constexpr int kMaxDictionaryWords = 8;

struct DictionaryPattern {
    // Sorted, without duplicates (sortedUniqueWords, wordlist.h), at least one.
    std::vector<std::string> words;
    // Distinct; at least one - "" for words written together.
    std::vector<std::string> separators;
    int minWords = 1;
    int maxWords = 1;
};

// One candidate, as positions in its pattern's lists: its words, and the
// separators between them (one fewer).
struct DictionaryChoice {
    std::vector<uint32_t> words;
    std::vector<uint32_t> separators;
};

class DictionarySpace {
public:
    // False, with `error` set, if `pattern` breaks one of the rules on
    // DictionaryPattern, has minWords < 1, maxWords < minWords or maxWords >
    // kMaxDictionaryWords, or has more candidates than 64 bits can number.
    static bool create(DictionaryPattern pattern, DictionarySpace& out, std::string& error);

    const std::vector<std::string>& words() const { return pattern_.words; }
    const std::vector<std::string>& separators() const { return pattern_.separators; }
    int minWords() const { return pattern_.minWords; }
    int maxWords() const { return pattern_.maxWords; }

    // How many candidates there are, of every word count.
    uint64_t size() const { return size_; }
    // The number of the first k-word candidate, and how many there are
    // (minWords <= k <= maxWords).
    uint64_t blockStart(int k) const { return blockStart_[k]; }
    uint64_t blockSize(int k) const { return blockSize_[k]; }

    // Candidate `number` (< size()).
    DictionaryChoice decode(uint64_t number) const;
    // The number of `choice`, which must be a valid one.
    uint64_t encode(const DictionaryChoice& choice) const;
    // The candidate's text: its words and separators, in order.
    std::string text(const DictionaryChoice& choice) const;
    std::string textOf(uint64_t number) const { return text(decode(number)); }

private:
    DictionaryPattern pattern_;
    uint64_t size_ = 0;
    std::vector<uint64_t> blockStart_;
    std::vector<uint64_t> blockSize_;
};

// The lower and upper bound of a search, as whole filenames, both inclusive
// - either may be left out. Compared byte by byte, so they need to be
// normalized (normalizeMpqName, wordlist.h) like the filenames are.
struct FilenameBounds {
    bool hasLower = false;
    std::string lower;
    bool hasUpper = false;
    std::string upper;

    bool contains(const std::string& filename) const;

    enum class Verdict {
        Outside, // no filename that starts with it is within the bounds
        Inside,  // every filename that starts with it is
        Mixed,   // some may be, some not - its continuations decide
    };
    // Where the filenames that start with `start` (`start` itself included)
    // are. Decided by `start` alone, so a search can skip, or stop checking,
    // every candidate after it.
    Verdict classifyStart(const std::string& start) const;
};

#endif // NAMEBREAK_ENGINE_DICTIONARY_H
