#ifndef NAMEBREAK_ENGINE_DICTIONARY_H
#define NAMEBREAK_ENGINE_DICTIONARY_H

#include <cstdint>
#include <string>
#include <vector>

// The candidates of a dictionary search (dictionary_search.h), and their
// numbers. A candidate is one to maxWords words, with a separator between
// each two, and then a tail: word, separator, word, ..., tail - every word
// from the same list, every separator from the same list, every tail from
// the same list (only "", without tails). Its filename is prefix +
// candidate + suffix.
//
// Every candidate has a number, counting from 0, the way an odometer counts
// with a wheel per word, per separator and for the tail: all one-word
// candidates first (if minWords is 1), then all two-word ones, and so on.
// Within a word count, the wheels from the left are word 1, separator 1,
// word 2, ..., word k, and the tail, which turns fastest:
//
//   number - start of the k-word block = (((w1 * S + s1) * W + w2) * S + s2 ... * W + wk) * T + t
//
// with W words, S separators and T tails. Without tails, T is 1 and t 0:
// the numbers are what they were before there were tails. The numbers only
// depend on the words, separators, word counts and tails - never on the
// bounds, the prefix or the suffix - so a search can be split up, or
// resumed, by number.

// The most words a candidate can have.
constexpr int kMaxDictionaryWords = 8;
// The longest a tail can be, and the most tails a search can have.
constexpr int kMaxDictionaryTailLength = 8;
constexpr uint32_t kMaxDictionaryTails = 1u << 20;

struct DictionaryPattern {
    // Sorted, without duplicates (sortedUniqueWords, wordlist.h), at least one.
    std::vector<std::string> words;
    // Distinct; at least one - "" for words written together.
    std::vector<std::string> separators;
    int minWords = 1;
    int maxWords = 1;
    // Sorted, without duplicates, at least one: what follows the last word
    // (see expandDictionaryTails). {""} for nothing.
    std::vector<std::string> tails = {""};
};

// The tails a list of tail elements makes (a [dictionary] section's
// `tails`, a coordinator target's): every string made of one of each
// element's, in turn - sorted, without duplicates, normalized as words are
// (normalizeMpqName, wordlist.h). An element is
//   digits:A-B   every string of A to B digits (digits:N for exactly N), as
//                0-9, 00-99 - leading zeros and all
//   letters:A-B  the same of the letters A-Z
//   X|Y|...      one of these literal strings - "" among them if the
//                element ends with '|', or has "||" in it - none with a ':'
//                (an element with one that isn't one of the above is an
//                error: a name never has one)
// No elements make {""}. False, with `error` set, for an element that's
// none of these, A > B, a tail with a '\', longer than
// kMaxDictionaryTailLength or with characters past ASCII, or more than
// kMaxDictionaryTails tails.
bool expandDictionaryTails(const std::vector<std::string>& elements, std::vector<std::string>& tails, std::string& error);

// One candidate, as positions in its pattern's lists: its words, and the
// separators between them (one fewer).
struct DictionaryChoice {
    std::vector<uint32_t> words;
    std::vector<uint32_t> separators;
    uint32_t tail = 0;
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
    const std::vector<std::string>& tails() const { return pattern_.tails; }

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
    // The candidate's text: its words and separators, in order, and its tail.
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
