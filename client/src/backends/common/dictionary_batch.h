#ifndef NAMEBREAK_BACKENDS_COMMON_DICTIONARY_BATCH_H
#define NAMEBREAK_BACKENDS_COMMON_DICTIONARY_BATCH_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/backend.h"

// Shared by the backends that search dictionaries (engine/dictionary_search.h):
// the word list and the suffix as a GPU kernel reads them, a call's batches
// as a launch's, and turning what a backend found into a DictionaryOutcome.
//
// A dictionary kernel hashes each of a batch's words, and then the suffix,
// on from the batch's state - one candidate per word. A launch's work is
// cut into *segments*: up to wordsPerSegment consecutive words of one batch,
// one per block of threads (work-group, threadgroup), so that each block
// reads its batch once and its threads read neighbouring words.

// The words as a kernel reads them. A kernel's threads take neighbouring
// words, and a warp (wavefront, SIMD-group) goes round a word's loop as
// many times as its longest word needs - so the words are kept in order of
// their length (and in the list's order among words of the same length),
// which a whole batch - every word of the list, nearly every batch - is
// searched in, and only a part of the list in the list's order.
//
// Each word has its entry, in that order: where its characters are -
// chars[offset ..], four to a uint32 (the first in its lowest byte), every
// word from a uint32 of its own - its length, the index of the character
// its basename starts at (just after its last '\', or 0 if it has none, when
// the basename is the batch's, continued) and its index in the list.
// positions[w] is word w's entry.
struct DictionaryWordEntry {
    uint32_t offset;
    uint32_t length;
    uint32_t basenameStart;
    uint32_t index;
};
static_assert(sizeof(DictionaryWordEntry) == 16, "DictionaryWordEntry is laid out as in the kernels");
struct DictionaryWordTable {
    std::vector<uint32_t> chars;
    std::vector<DictionaryWordEntry> entries;
    std::vector<uint32_t> positions;
};
// False if the words take more uint32s than 32 bits can index.
bool makeDictionaryWordTable(const std::vector<std::string>& words, DictionaryWordTable& out);

// The suffix's characters as a kernel hashes them: for character i,
// keys[3 * i] (hashA's crypt-table key), keys[3 * i + 1] (the basename
// hash's) and keys[3 * i + 2] (the character plus 3 - what a step of the
// hash adds to seed2 besides seed1 and 33 times seed2:
//     seed1 = key ^ (seed1 + seed2)
//     seed2 = seed1 + 33 * seed2 + (ch + 3)
// which is the usual step, seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3,
// added up in another order).
std::vector<uint32_t> dictionarySuffixKeys(const std::string& suffix, const uint32_t* cryptTable);

// The suffix filter: a kernel hashes the suffix in full only where it can
// make a match. A step of the hash lets no bit of the state affect a lower
// bit of the result (see backends/common/lowbits_filter.h), so the low
// kDictionaryFilterBits bits of seed1 after the suffix depend only on those
// of (seed1, seed2) before it. The filter has a bit for each of their
// 2^(2 * kDictionaryFilterBits) values - lowBitsFilterIndex(seed1, seed2,
// kDictionaryFilterBits) - set if the suffix leaves seed1's low bits, those
// of `mask`, the target's. A candidate that matches has all of its bits, so
// its bit is always set: only candidates that can't match are skipped. The
// bits are in uint32s, bit i in bit i % 32 of word i / 32.
//
// Overridable at compile time (-DNAMEBREAK_DICTIONARY_FILTER_BITS=N), so
// the tests can let most candidates through it.
#ifndef NAMEBREAK_DICTIONARY_FILTER_BITS
#define NAMEBREAK_DICTIONARY_FILTER_BITS 7
#endif
constexpr int kDictionaryFilterBits = NAMEBREAK_DICTIONARY_FILTER_BITS;
static_assert(kDictionaryFilterBits >= 1 && kDictionaryFilterBits <= 8, "NAMEBREAK_DICTIONARY_FILTER_BITS must be 1-8");
constexpr uint32_t kDictionaryFilterWords = (1u << (2 * kDictionaryFilterBits)) < 32 ? 1 : (1u << (2 * kDictionaryFilterBits)) / 32;
// The bits of seed1 a filter compares: those a hit must match (see
// hash_match.h), of hashA or the basename hash.
uint32_t dictionaryFilterMask(bool basename);
// The filter for the suffix hashed with the crypt table at `keyOffset`
// (kHashAOffset, kFileKeyOffset), to `target`'s bits of `mask`.
std::vector<uint32_t> buildDictionarySuffixFilter(const std::string& suffix, const uint32_t* cryptTable, int keyOffset, uint32_t target,
                                                  uint32_t mask);
// Checks every bit of `filter` against the definition above, independently
// of how it was built: from a state with random high bits, hashed with
// engine/dictionary_search.h's continueHash. False, with `error` saying
// which bit is wrong, if one is.
bool checkDictionarySuffixFilter(const std::vector<uint32_t>& filter, const std::string& suffix, const uint32_t* cryptTable, int keyOffset,
                                 uint32_t target, uint32_t mask, uint64_t seed, std::string& error);

// One batch of a launch, as a kernel reads it: the batch's states, its
// words and tails, and the launch's first segment of them. A kernel's thread
// takes a *cell*: a word, and a chunk of up to tailsPerCell of the batch's
// tails - tailChunks of them - so that one word with many tails is hashed by
// many threads. The batch's cells are its first chunk of each word, in
// turn, then its second, and so on: cell c is word c % wordCount (of the
// batch's) with chunk c / wordCount. Segments firstSegment .. firstSegment +
// ceil(wordCount * tailChunks / cellsPerSegment) - 1 are its. Must match its
// namesakes in the kernels (cuda_backend.cu, dictionary.cl,
// dictionary.metal).
struct DictionaryLaunchBatch {
    uint32_t seed1;
    uint32_t seed2;
    uint32_t basenameSeed1;
    uint32_t basenameSeed2;
    uint32_t firstWord;
    uint32_t wordCount;
    uint32_t firstSegment;
    uint32_t firstTail;
    uint32_t tailCount;
    uint32_t tailChunks;
};
static_assert(sizeof(DictionaryLaunchBatch) == 40, "DictionaryLaunchBatch is laid out as in the kernels");

// `batches` as a launch's, in `out`. Returns how many segments they have.
uint64_t planDictionaryLaunch(const std::vector<DictionaryBatch>& batches, uint32_t cellsPerSegment, uint32_t tailsPerCell,
                              std::vector<DictionaryLaunchBatch>& out);

// The tails as a kernel reads them: four uint32s each, the first two its
// characters (four to a uint32, the first in the lowest byte - at most
// kMaxDictionaryTailLength of them), the third its length.
std::vector<uint32_t> dictionaryTailTable(const std::vector<std::string>& tails);

// The batch a launch's segment is in: the last whose firstSegment is at most
// `segment` - what a kernel works out for its block, by binary search.
uint32_t dictionaryBatchOfSegment(const std::vector<DictionaryLaunchBatch>& batches, uint64_t segment);

// A candidate a kernel reports: word `word` (its index in the list) and tail
// `tail` (in the tails) of the launch's batch `batch`. Must match its
// namesakes in the kernels.
struct DictionaryHit {
    uint32_t batch;
    uint32_t word;
    uint32_t tail;
};

// Turns the hits a backend reports into what runDictionaryBatches returns:
// rebuilds each candidate's filename, hashes it from scratch - independent
// code, on the CPU - and drops it, with a WARNING (which the tests notice as
// a missing or extra hit), if that doesn't match; and checks each hashA
// hit's hashB.
class DictionaryHitVerifier {
public:
    void begin(const DictionaryConstants& constants);
    // The hashes a backend works out for each candidate, and reports the
    // hits of (see dictionaryHashes).
    DictionaryHashes hashes() const { return hashes_; }
    // Adds the hashA hits and basename hits of `batches` to `outcome` - the
    // basename hits only if they're recorded, and if the backend worked out
    // the basename hash alone, a basename hit whose hashA matches as a
    // hashA hit too - and if every candidate's basename is the suffix's,
    // and it matches, the first candidate of the call as the one basename
    // hit.
    void addHits(const std::vector<DictionaryBatch>& batches, const std::vector<DictionaryHit>& hashAHits,
                 const std::vector<DictionaryHit>& basenameHits, DictionaryOutcome& outcome) const;

private:
    std::vector<std::string> words_;
    std::vector<std::string> tails_;
    std::string suffix_;
    std::vector<uint32_t> cryptTable_;
    uint32_t targetA_ = 0;
    uint32_t targetB_ = 0;
    bool checkBasename_ = false;
    uint32_t basenameKey_ = 0;
    bool recordBasenames_ = false;
    DictionaryHashes hashes_ = DictionaryHashes::HashA;
    bool suffixBasenameMatches_ = false;
};

// How many hits of each kind a dictionary launch can record before it's
// searched again with room for all of them (see the GPU backends'
// runDictionaryBatches) - set small at compile time by the tests (see
// CMakeLists.txt), so that that happens.
#ifndef NAMEBREAK_DICTIONARY_HIT_CAPACITY
#define NAMEBREAK_DICTIONARY_HIT_CAPACITY 4096
#endif
static_assert(NAMEBREAK_DICTIONARY_HIT_CAPACITY >= 1, "NAMEBREAK_DICTIONARY_HIT_CAPACITY must be at least 1");

// A GPU backend's threads per block and words per thread in a dictionary
// launch - one segment is their product - unless the build overrides them
// for every GPU backend at once (-DNAMEBREAK_DICTIONARY_THREADS_PER_BLOCK=N,
// -DNAMEBREAK_DICTIONARY_WORDS_PER_THREAD=N): the tests do, so that their
// small word lists still make batches of several segments.
constexpr int dictionaryThreadsPerBlockOr([[maybe_unused]] int backendDefault) {
#ifdef NAMEBREAK_DICTIONARY_THREADS_PER_BLOCK
    return NAMEBREAK_DICTIONARY_THREADS_PER_BLOCK;
#else
    return backendDefault;
#endif
}
constexpr int dictionaryWordsPerThreadOr([[maybe_unused]] int backendDefault) {
#ifdef NAMEBREAK_DICTIONARY_WORDS_PER_THREAD
    return NAMEBREAK_DICTIONARY_WORDS_PER_THREAD;
#else
    return backendDefault;
#endif
}
// The same for how many tails a cell has (see DictionaryLaunchBatch):
// -DNAMEBREAK_DICTIONARY_TAILS_PER_CELL=N.
constexpr int dictionaryTailsPerCellOr([[maybe_unused]] int backendDefault) {
#ifdef NAMEBREAK_DICTIONARY_TAILS_PER_CELL
    return NAMEBREAK_DICTIONARY_TAILS_PER_CELL;
#else
    return backendDefault;
#endif
}

#endif // NAMEBREAK_BACKENDS_COMMON_DICTIONARY_BATCH_H
