#ifndef NAMEBREAK_ENGINE_BACKEND_H
#define NAMEBREAK_ENGINE_BACKEND_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/limits.h"

// What runSearch() (search.h) needs from the hardware it runs on. The engine
// walks the search space: it splits each candidate into a "leading" part,
// enumerated and hashed on the CPU, and a "trailing" part of about
// windowChars() characters, which a backend enumerates - one runBatch() call
// per chunk of the trailing space. Everything else (validation, bounds,
// pruning, pause/abort, writing matches) is the engine's, so a backend is
// just the hashing loop and whatever it takes to feed it.
//
// Terminology:
// * Candidate = The part of the name that we are brute-forcing
// * Filename  = The Prefix + Candidate + Suffix
// * Trailing index = a candidate's trailing part's position within the
//   trailing space, in the same last-character-fastest order as
//   indexToString/stringToIndex (candidate.h).

// Everything that changes from one leading value to the next (i.e. once per
// batch in the common case). A plain struct so a GPU backend can pass it to
// its kernel by value, as a kernel argument.
struct BatchParams {
    char prefix[kMaxPrefixSize]; // req.prefix + the leading characters, NUL-terminated
    short prefixSize;
    uint32_t seed1Start;         // hash state after the (extended) prefix - see
    uint32_t seed2Start;         // mpqHashWithPrefixCache_CPU / IncrementalPrefixHasher
};

// One runBatch() call's worth of work: the candidates whose trailing index is
// in [start, start + count), after params' prefix - see runBatches.
struct BatchRequest {
    uint64_t start;
    uint64_t count;
    BatchParams params;
};

// Everything that stays the same for a whole search.
struct SearchConstants {
    std::string alphabet;
    std::string suffix;
    const uint32_t* cryptTable = nullptr; // 0x500 entries, see prepareCryptTable
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
};

// What one runBatch() (or runBatches()) call found.
struct BatchOutcome {
    // How many candidates matched hashA. If this is more than MAX_MATCHES,
    // nothing below is filled in (no hit has been checked against hashB) and
    // the caller must search the range again in smaller pieces.
    int hitCount = 0;
    // The complete filename of every hashA hit, in no particular order.
    std::vector<std::string> hits;
    // Set once one of the hits also matches hashB.
    bool found = false;
    std::string foundFilename;
};

class SearchBackend {
public:
    virtual ~SearchBackend() = default;

    // Short lowercase name, as in --backend <name>.
    virtual const char* name() const = 0;

    // The alphabet sizes this backend can search, ascending - or empty if it
    // can search any size from 1 to MAX_ALPHABET_SIZE.
    virtual std::vector<int> supportedAlphabetSizes() const = 0;
    // How many trailing characters the backend should enumerate per batch -
    // the engine uses more only when a candidate is too long for the leading
    // part to be indexed in 64 bits.
    virtual int windowChars() const = 0;
    // The longest trailing part runBatch() accepts.
    virtual int maxTrailingLen() const = 0;
    // How many candidates one runBatch() call should cover at most. The engine
    // cuts each leading value's trailing space at multiples of this, so only
    // the first and last batch of a range start or end anywhere else.
    virtual uint64_t batchSize(int alphabetSize) const = 0;

    // Called once before a search's first runBatch(), and endSearch() once
    // after its last. A backend is reused for many searches (one per claimed
    // coordinator range), and must not carry anything over between them.
    virtual void beginSearch(const SearchConstants& constants) = 0;
    // Hashes every candidate whose trailing index is in [start, start + count)
    // (count > 0), each preceded by params' prefix and followed by the suffix.
    virtual BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) = 0;
    virtual void endSearch() = 0;

    // How many batches runBatches() takes at once. A GPU backend can search
    // several leading values' batches in one launch, which is longer than one
    // batch's - a launch's fixed cost, the GPU idle between two of them, then
    // matters less. With 1 (the default) the engine only calls runBatch().
    virtual int maxBatchesPerCall() const { return 1; }
    // Searches every batch of `batches` (at most maxBatchesPerCall() of them,
    // all of trailingLen characters), exactly as runBatch() would each, and
    // reports what they found together: hitCount is their total, and if it's
    // more than MAX_MATCHES nothing else is filled in, and the caller must
    // search each batch again on its own. The default calls runBatch() for
    // each.
    virtual BatchOutcome runBatches(int trailingLen, const std::vector<BatchRequest>& batches);
};

// `backend`'s supported alphabet sizes, for a message: "42, 43 or 50".
std::string describeAlphabetSizes(const SearchBackend& backend);

#endif // NAMEBREAK_ENGINE_BACKEND_H
