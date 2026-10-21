#include "backends/reference/reference_backend.h"

#include <cstdio>
#include <string>
#include <vector>

#include "engine/candidate.h"
#include "engine/hash_match.h"

namespace {

// Hashes every candidate from scratch (every trailing character, then the
// suffix) - none of the CUDA backend's row trick - so the only thing it
// shares with the fast path is the hash recurrence itself.
class ReferenceBackend : public SearchBackend {
public:
    const char* name() const override { return "reference"; }
    int windowChars() const override { return 4; }
    // The trailing index is a uint64_t and the engine keeps the trailing
    // part within what that can index, so any length the engine asks for works.
    int maxTrailingLen() const override { return MAX_CANDIDATE_LEN; }
    // About a tenth of a second of work - small enough for pause/abort to
    // feel immediate.
    uint64_t batchSize(int) const override { return 1u << 22; }

    void beginSearch(const SearchConstants& constants) override {
        alphabet_ = constants.alphabet;
        suffix_ = constants.suffix;
        cryptTable_.assign(constants.cryptTable, constants.cryptTable + 0x500);
        targetA_ = constants.targetHashA;
        targetB_ = constants.targetHashB;
        matchCheck_ = constants.matchCheck;
        rules_ = constants.trailingRules;
        insertions_ = constants.trailingInsertions;
    }

    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;

    void endSearch() override {}

    // A dictionary search, hashing every candidate's whole filename from
    // scratch - none of the batches' hash states - so it shares nothing
    // with the engine but the words.
    bool supportsDictionary() const override { return true; }
    uint64_t dictionaryCandidatesPerCall() const override { return 1u << 20; }
    void beginDictionarySearch(const DictionaryConstants& constants) override {
        dictionary_ = constants;
        cryptTable_.assign(constants.cryptTable, constants.cryptTable + 0x500);
        dictionary_.cryptTable = cryptTable_.data();
    }
    DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches) override;

    // A meet-in-the-middle search, rewinding every tail from scratch - none of
    // MeetInTheMiddleWalk's stack of states, and no filter in front of the
    // table.
    bool supportsMeetInTheMiddle() const override { return true; }
    uint64_t meetInTheMiddleTailsPerCall() const override { return 1u << 18; }
    void beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants& constants) override {
        meet_in_the_middle_ = constants;
        cryptTable_.assign(constants.cryptTable, constants.cryptTable + 0x500);
        meet_in_the_middle_.cryptTable = cryptTable_.data();
    }
    MeetInTheMiddleOutcome runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) override;

private:
    // One step of the MPQ hash recurrence, with the crypt table at `offset`
    // (0x100 for hashA, 0x200 for hashB).
    void step(uint32_t& seed1, uint32_t& seed2, unsigned char ch, int offset) const {
        seed1 = cryptTable_[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }

    // A hit whose hashB matches too: the match, or a false match.
    template <class Outcome>
    void addBothHashesMatch(const std::string& filename, const MatchCheck& check, Outcome& outcome) const {
        std::string reason = falseMatchReason(check, filename, cryptTable_.data());
        if (!reason.empty()) {
            outcome.falseMatches.push_back({filename, std::move(reason)});
        } else if (!outcome.found) {
            outcome.found = true;
            outcome.foundFilename = filename;
        }
    }

    uint32_t hashFromScratch(const std::string& filename, int offset) const {
        uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
        for (unsigned char ch : filename)
            step(seed1, seed2, ch, offset);
        return seed1;
    }

    std::string alphabet_;
    std::string suffix_;
    std::vector<uint32_t> cryptTable_;
    uint32_t targetA_ = 0;
    uint32_t targetB_ = 0;
    MatchCheck matchCheck_;
    PruneRules rules_;
    std::vector<TrailingInsertion> insertions_;
    DictionaryConstants dictionary_;
    MeetInTheMiddleConstants meet_in_the_middle_;
};

MeetInTheMiddleOutcome ReferenceBackend::runMeetInTheMiddleBatch(const MeetInTheMiddleBatch& batch) {
    MeetInTheMiddleOutcome outcome;
    const std::string& alphabet = meet_in_the_middle_.alphabet;
    for (uint64_t n = 0; n < batch.count; ++n) {
        const uint64_t index = batch.start + n;
        // The tail of that index: the directory's last character is the
        // index's most significant digit, being the first one rewound (see
        // MeetInTheMiddleSpace).
        std::string tail(batch.tailLength, ' ');
        uint64_t rest = index;
        for (int k = 0; k < batch.tailLength; ++k) {
            tail[k] = alphabet[rest % alphabet.size()];
            rest /= alphabet.size();
        }
        // The rules, over the whole tail from the right - no skipping of
        // anything they reject, just a verdict per tail.
        bool pruned = false;
        if (meet_in_the_middle_.rules.any()) {
            TailPruneState prune = meet_in_the_middle_.tailEntry;
            for (int k = batch.tailLength - 1; k >= 0 && !pruned; --k)
                pruned = !tailPruneStep(meet_in_the_middle_.rules, prune, tail[k]);
        }
        if (pruned)
            continue;
        ++outcome.probed;
        if (meet_in_the_middle_.table->holds(
                meetingPointKey(rewindHash(meet_in_the_middle_.meetingPoint, tail, kHashAOffset, meet_in_the_middle_.cryptTable))))
            outcome.hits.push_back({batch.tailLength, index});
    }
    return outcome;
}

DictionaryOutcome ReferenceBackend::runDictionaryBatches(const std::vector<DictionaryBatch>& batches) {
    DictionaryOutcome outcome;
    // With the key, and a suffix that has no '\', only a candidate whose
    // basename matches it is compared to the targets - unless every
    // candidate's hashA is to be recorded (see dictionaryHashes).
    const bool basenameFirst =
        dictionary_.checkBasename && !dictionary_.recordHashAMatches && dictionary_.suffix.find('\\') == std::string::npos;
    const std::vector<std::string>& separators = dictionary_.separators;
    for (const DictionaryBatch& batch : batches) {
        const uint64_t perPair = (uint64_t) batch.wordCount * batch.tailCount;
        for (uint64_t c = 0; c < dictionaryBatchCandidates(batch); ++c) {
            // Its pair, if the batch has pairs (see DictionaryBatch).
            std::string pair;
            if (batch.pairCount) {
                const uint64_t p = batch.firstPair + c / perPair;
                pair = dictionary_.words[p / separators.size()] + separators[p % separators.size()];
            }
            const std::string filename = batch.leading + pair + dictionary_.words[batch.firstWord + c % perPair / batch.tailCount] +
                                         dictionary_.tails[batch.firstTail + c % batch.tailCount] + dictionary_.suffix;
            bool basenameMatches = false;
            if (dictionary_.checkBasename) {
                const size_t slash = filename.rfind('\\');
                const std::string basename = slash == std::string::npos ? filename : filename.substr(slash + 1);
                basenameMatches = basenameKeyMatches(hashFromScratch(basename, 0x300), dictionary_.basenameKey);
                if (basenameMatches && dictionary_.recordBasenames)
                    outcome.basenameHits.push_back(filename);
            }
            if ((!basenameFirst || basenameMatches) && hashAMatches(hashFromScratch(filename, 0x100), dictionary_.targetHashA)) {
                outcome.hits.push_back(filename);
                if (hashFromScratch(filename, 0x200) == dictionary_.targetHashB)
                    addBothHashesMatch(filename, dictionary_.matchCheck, outcome);
            }
        }
    }
    return outcome;
}

BatchOutcome ReferenceBackend::runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) {
    BatchOutcome outcome;
    const std::string prefix(params.prefix, params.prefixSize);
    const uint64_t alphabetSize = alphabet_.size();

    // The trailing characters of candidate `start`, as alphabet positions
    // (most significant first), advanced like an odometer below.
    std::vector<size_t> digit(trailingLen);
    uint64_t index = start;
    for (int i = trailingLen - 1; i >= 0; --i) {
        digit[i] = index % alphabetSize;
        index /= alphabetSize;
    }

    for (uint64_t n = 0; n < count; ++n) {
        // The trailing part, with the text inserted into it.
        std::string trailing;
        for (int i = 0; i < trailingLen; ++i)
            trailing += alphabet_[digit[i]];
        const std::string expanded = withTrailingInsertions(trailing, insertions_);

        // Pruned: every character of it but the last, checked from where
        // the leading characters left off - and the last one, if even a
        // backslash there would leave too few.
        bool pruned = false;
        if (rules_.any()) {
            PruneState state = params.pruneEntry;
            for (size_t i = 0; i + 1 < expanded.size() && !pruned; ++i)
                pruned = !pruneStep_CPU(rules_, state, expanded[i]);
            pruned = pruned || !canReachMinBackslashes_CPU(rules_, state, 1);
        }
        uint32_t seed1 = params.seed1Start, seed2 = params.seed2Start;
        for (unsigned char ch : expanded)
            step(seed1, seed2, ch, 0x100);
        for (unsigned char ch : suffix_)
            step(seed1, seed2, ch, 0x100);

        if (!pruned && hashAMatches(seed1, targetA_) && ++outcome.hitCount <= MAX_MATCHES) {
            const std::string filename = prefix + expanded + suffix_;
            // The same cross-check HitVerifier does for the other backends:
            // hashing the complete filename from scratch must agree with
            // the prefix-state path that found it.
            uint32_t verifyHashA = hashFromScratch(filename, 0x100);
            if (verifyHashA != seed1) {
                printf("WARNING: hashA mismatch for '%s' - incremental hash 0x%08X, full-filename hash 0x%08X\n", filename.c_str(), seed1,
                       verifyHashA);
            }
            if (hashFromScratch(filename, 0x200) == targetB_)
                addBothHashesMatch(filename, matchCheck_, outcome);
            outcome.hits.push_back(filename);
        }

        for (int i = trailingLen - 1; i >= 0; --i) {
            if (++digit[i] < alphabetSize)
                break;
            digit[i] = 0;
        }
    }

    // As documented on BatchOutcome: past MAX_MATCHES, report only the count.
    if (outcome.hitCount > MAX_MATCHES) {
        outcome.hits.clear();
        outcome.found = false;
        outcome.foundFilename.clear();
    }
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeReferenceBackend() {
    return std::make_unique<ReferenceBackend>();
}
