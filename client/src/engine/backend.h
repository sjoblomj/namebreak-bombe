#ifndef NAMEBREAK_ENGINE_BACKEND_H
#define NAMEBREAK_ENGINE_BACKEND_H

#include <cstdint>
#include <string>
#include <vector>

#include "engine/candidate.h"
#include "engine/match_check.h"
#include "engine/meet_in_the_middle.h"
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
    // The pruning rules' state after the leading characters - where a backend
    // starts checking the trailing ones, if SearchConstants::trailingRules
    // has any.
    PruneState pruneEntry;
};

// One runBatch() call's worth of work: the candidates whose trailing index is
// in [start, start + count), after params' prefix - see runBatches.
struct BatchRequest {
    uint64_t start;
    uint64_t count;
    BatchParams params;
};

// Text inserted into a candidate's trailing part (SearchRequest::
// insertFromStart and insertFromEnd), before its last charsAfter characters
// - from 1, between a row's last character and the candidate's last, to
// trailingLen - 1, after its first. Counted from the end so that it doesn't
// depend on the trailing length.
struct TrailingInsertion {
    int charsAfter = 0;
    std::string text;
    bool operator==(const TrailingInsertion& o) const { return charsAfter == o.charsAfter && text == o.text; }
};

// Everything that stays the same for a whole search.
struct SearchConstants {
    std::string alphabet;
    std::string suffix;
    const uint32_t* cryptTable = nullptr; // 0x500 entries, see prepareCryptTable
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
    // What a candidate matching both hashes must have as well to be the
    // file's name (see MatchCheck) - one that hasn't is a false match.
    MatchCheck matchCheck;
    // The rules every trailing character but the last must pass
    // (SearchRequest::pruneWholeCandidate), starting from each batch's
    // BatchParams::pruneEntry - or none, and every candidate of a batch is
    // searched. Every backend searches exactly what they allow, so that a
    // search covers the same candidates whichever backend runs it.
    PruneRules trailingRules;
    // Text inserted into every candidate's trailing part: at most two, at
    // different places, the one with the most characters after it first.
    // The rules check it as they do the characters around it. (Text
    // inserted after a candidate's last character is part of `suffix`; text
    // inserted into its leading part, of each batch's BatchParams::prefix.)
    std::vector<TrailingInsertion> trailingInsertions;
    // How a backend that can search what trailingRules leave either way -
    // walking lists of the row groups that survive them, or every group and
    // dropping the hits outside them (CUDA/HIP and OpenCL, see
    // NAMEBREAK_LIST_MIN_PRUNED_PERCENT) - chooses: by how much the lists
    // leave out, or always the one way. Only the self-test asks for one, to
    // test both: they find the same. Other backends ignore it.
    enum class ListWalking { Auto, Always, Never };
    ListWalking listWalking = ListWalking::Auto;
};

// A trailing part of `trailing` (trailingLen characters) with `insertions`
// inserted - what a candidate's trailing part is between its prefix and its
// suffix.
std::string withTrailingInsertions(const std::string& trailing, const std::vector<TrailingInsertion>& insertions);

// What one runBatch() (or runBatches()) call found.
struct BatchOutcome {
    // How many candidates matched hashA. If this is more than MAX_MATCHES,
    // nothing below is filled in (no hit has been checked against hashB) and
    // the caller must search the range again in smaller pieces.
    int hitCount = 0;
    // The complete filename of every hashA hit, in no particular order.
    std::vector<std::string> hits;
    // Set once one of the hits also matches hashB - and passes
    // SearchConstants::matchCheck.
    bool found = false;
    std::string foundFilename;
    // The hits that match hashB but not the MatchCheck - every one, found
    // or not: a false match is no reason to stop.
    std::vector<FalseMatch> falseMatches;
};

// Everything that stays the same for a whole dictionary search
// (dictionary_search.h).
struct DictionaryConstants {
    // The words a candidate's last word is one of - DictionaryBatch::firstWord
    // indexes them - and the tails that follow it (DictionaryBatch::firstTail;
    // {""} for none). And the separators: a batch's pairs are a word and a
    // separator each (see DictionaryBatch).
    std::vector<std::string> words;
    std::vector<std::string> tails = {""};
    std::vector<std::string> separators = {""};
    std::string suffix;
    const uint32_t* cryptTable = nullptr; // 0x500 entries, see prepareCryptTable
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
    // As SearchConstants::matchCheck.
    MatchCheck matchCheck;
    // The file's encryption key is known: compare each candidate's basename -
    // its filename after the last '\' - hashed as Storm makes the key (hash
    // type 3), to basenameKey. The file's name has that basename, so only a
    // candidate whose basename matches is compared to the targets - unless
    // recordHashAMatches (see dictionaryHashes).
    bool checkBasename = false;
    uint32_t basenameKey = 0;
    // With checkBasename: report every candidate whose basename matches
    // (DictionaryOutcome::basenameHits).
    bool recordBasenames = true;
    // With checkBasename: compare every candidate to the targets all the
    // same, as if the key weren't known - so that every hashA hit is
    // reported, and a wrong key can't hide the file.
    bool recordHashAMatches = false;
};

// Which hashes a dictionary search's backend works out for each candidate -
// numbered as the OpenCL and Metal kernels' HASHES.
enum class DictionaryHashes { HashA = 0, Basename = 1, Both = 2 };

// HashA, unless the basename key is checked and the suffix has no '\' (if
// it has one, every candidate's basename is the same, its end, and
// DictionaryHitVerifier checks it once a call). With the key, Basename: the
// file's name has the key's basename, so a candidate whose basename doesn't
// match it can't be the file, and its hashA would be worked out for
// nothing - the verifier checks hashA and hashB of those that match. With
// recordHashAMatches, every candidate's hashA all the same: Both - or
// HashA, if the basenames aren't recorded.
inline DictionaryHashes dictionaryHashes(const DictionaryConstants& constants) {
    if (!constants.checkBasename || constants.suffix.find('\\') != std::string::npos)
        return DictionaryHashes::HashA;
    if (!constants.recordHashAMatches)
        return DictionaryHashes::Basename;
    return constants.recordBasenames ? DictionaryHashes::Both : DictionaryHashes::HashA;
}

// One leading part of a dictionary search's candidates - its prefix, and its
// words and separators but the last word - followed by each of the words
// [firstWord, firstWord + wordCount), each followed by each of the tails
// [firstTail, firstTail + tailCount), and then the suffix: wordCount *
// tailCount candidates. A search only gives a batch of one word part of the
// tails, so that a batch's candidates are numbered one after another - but a
// backend searches any.
//
// With pairCount > 0, the leading part stops one word earlier, and a *pair*
// - a word and the separator after it - comes between it and the last word:
// pair p is word p / S and then separator p % S, of S separators
// (DictionaryConstants::separators), as a candidate's number has them. The
// batch's candidates are then `leading`, each of the pairs [firstPair,
// firstPair + pairCount), and each word and tail as above - pairCount *
// wordCount * tailCount of them, pair by pair. A search only gives a batch
// of pairs every word and every tail.
struct DictionaryBatch {
    std::string leading;
    // hashA's state after `leading` (see mpqHashWithPrefixCache_CPU).
    uint32_t seed1 = 0;
    uint32_t seed2 = 0;
    // The basename hash's state after `leading`: hash type 3 of what follows
    // its last '\' (see dictionary_search.h's continueBasenameHash).
    uint32_t basenameSeed1 = 0;
    uint32_t basenameSeed2 = 0;
    uint32_t firstWord = 0;
    uint32_t wordCount = 0;
    uint32_t firstTail = 0;
    uint32_t tailCount = 1;
    uint32_t firstPair = 0;
    uint32_t pairCount = 0;
};

// How many candidates `batch` has.
inline uint64_t dictionaryBatchCandidates(const DictionaryBatch& batch) {
    return (uint64_t) (batch.pairCount ? batch.pairCount : 1) * batch.wordCount * batch.tailCount;
}

// What one runDictionaryBatches() call found - every hit, in no particular
// order.
struct DictionaryOutcome {
    // The filename of every hashA hit - of the candidates compared to the
    // targets at all (see dictionaryHashes).
    std::vector<std::string> hits;
    // Set once one of them also matches hashB - and passes
    // DictionaryConstants::matchCheck.
    bool found = false;
    std::string foundFilename;
    // As BatchOutcome's.
    std::vector<FalseMatch> falseMatches;
    // With recordBasenames, the filename of every candidate whose basename
    // matched basenameKey - or, if the suffix has a '\', and so every
    // candidate has the same basename, of at least one of the call's
    // candidates if it matched (see DictionaryHitVerifier,
    // backends/common/dictionary_batch.h).
    std::vector<std::string> basenameHits;
};

// Everything that stays the same for a whole meet-in-the-middle search
// (engine/meet_in_the_middle_search.h).
struct MeetInTheMiddleConstants {
    std::string alphabet;
    // The hash state after the unknown directory, which the backward walk
    // rewinds from - hashA's. (hashB's, and the file's own two hashes, the
    // engine checks itself, on the few tails that get this far.)
    HashState meetingPoint{0, 0};
    const uint32_t* cryptTable = nullptr; // 0x500 entries, see prepareCryptTable
    // The forward half: every directory of the table's length, by the state
    // it reaches. The engine builds it and keeps it for the whole search.
    const MeetInTheMiddleTable* table = nullptr;
    // The rules a tail must pass, and the state the directory's last
    // character is checked from - tailPruneStart(suffix)
    // (meet_in_the_middle.h). Every backend skips exactly the tails they
    // reject, so that a search covers the same directories whichever one runs
    // it.
    PruneRules rules;
    TailPruneState tailEntry;
};

// One runMeetInTheMiddleBatch() call's worth of work: the tails of `tailLength`
// characters whose index is in [start, start + count) - see
// MeetInTheMiddleSpace (meet_in_the_middle.h) for the numbering.
struct MeetInTheMiddleBatch {
    int tailLength = 0;
    uint64_t start = 0;
    uint64_t count = 0;
};

// A tail whose state the table holds: the directory is one of the table's
// prefixes followed by this tail. Which prefix, the engine works out (the
// table keeps only the states), and then checks the directory in full.
struct MeetInTheMiddleHit {
    int tailLength = 0;
    uint64_t index = 0;
};

// What one runMeetInTheMiddleBatch() call found.
struct MeetInTheMiddleOutcome {
    // Every hit, in no particular order.
    std::vector<MeetInTheMiddleHit> hits;
    // How many of the batch's tails were looked up in the table - the rules
    // skipped the rest.
    uint64_t probed = 0;
};

class SearchBackend {
public:
    virtual ~SearchBackend() = default;

    // Short lowercase name, as in --backend <name>.
    virtual const char* name() const = 0;

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
    // (count > 0), each preceded by params' prefix and followed by the suffix
    // - except those whose trailing characters, all but the last, break
    // trailingRules (see SearchConstants).
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

    // Dictionary searches (dictionary_search.h) - only on the backends that
    // say they can. The others never get these calls.
    virtual bool supportsDictionary() const { return false; }
    // How many candidates one runDictionaryBatches() call should cover, about.
    virtual uint64_t dictionaryCandidatesPerCall() const { return 1u << 22; }
    // How many batches one runDictionaryBatches() call gets at most - with
    // few words, a call's candidates are many short batches.
    virtual size_t dictionaryBatchesPerCall() const { return 1u << 16; }
    // Called once before a dictionary search's first runDictionaryBatches(),
    // and endDictionarySearch() once after its last.
    virtual void beginDictionarySearch(const DictionaryConstants& constants);
    // Hashes every candidate of every batch.
    virtual DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches);
    virtual void endDictionarySearch() {}

    // Meet-in-the-middle searches (engine/meet_in_the_middle_search.h) - only
    // on the backends that say they can. The others never get these calls.
    virtual bool supportsMeetInTheMiddle() const { return false; }
    // About how many tails one runMeetInTheMiddleBatch() call should cover:
    // enough to be worth a call, few enough for pause and abort to feel
    // immediate.
    virtual uint64_t meetInTheMiddleTailsPerCall() const { return 1u << 22; }
    // Called once before a meet-in-the-middle search's first
    // runMeetInTheMiddleBatch(), and endMeetInTheMiddleSearch() once after
    // its last.
    virtual void beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants);
    // Rewinds every tail of the batch that the rules allow and looks its
    // state up in the table, reporting the ones it holds.
    virtual MeetInTheMiddleOutcome runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch);
    virtual void endMeetInTheMiddleSearch() {}
};


#endif // NAMEBREAK_ENGINE_BACKEND_H
