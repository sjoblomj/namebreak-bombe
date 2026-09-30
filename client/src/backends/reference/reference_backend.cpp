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
    std::vector<int> supportedAlphabetSizes() const override { return {}; }
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
        rules_ = constants.trailingRules;
    }

    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;

    void endSearch() override {}

private:
    // One step of the MPQ hash recurrence, with the crypt table at `offset`
    // (0x100 for hashA, 0x200 for hashB).
    void step(uint32_t& seed1, uint32_t& seed2, unsigned char ch, int offset) const {
        seed1 = cryptTable_[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
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
    PruneRules rules_;
};

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
        // Pruned: every trailing character but the last, checked from
        // where the leading characters left off.
        bool pruned = false;
        if (rules_.any()) {
            PruneState state = params.pruneEntry;
            for (int i = 0; i + 1 < trailingLen && !pruned; ++i)
                pruned = !pruneStep_CPU(rules_, state, alphabet_[digit[i]]);
        }
        uint32_t seed1 = params.seed1Start, seed2 = params.seed2Start;
        for (int i = 0; i < trailingLen; ++i)
            step(seed1, seed2, (unsigned char) alphabet_[digit[i]], 0x100);
        for (unsigned char ch : suffix_)
            step(seed1, seed2, ch, 0x100);

        if (!pruned && hashAMatches(seed1, targetA_) && ++outcome.hitCount <= MAX_MATCHES) {
            std::string filename = prefix;
            for (int i = 0; i < trailingLen; ++i)
                filename += alphabet_[digit[i]];
            filename += suffix_;
            // The same cross-check HitVerifier does for the other backends:
            // hashing the complete filename from scratch must agree with
            // the prefix-state path that found it.
            uint32_t verifyHashA = hashFromScratch(filename, 0x100);
            if (verifyHashA != seed1) {
                printf("WARNING: hashA mismatch for '%s' - incremental hash 0x%08X, full-filename hash 0x%08X\n", filename.c_str(), seed1,
                       verifyHashA);
            }
            if (!outcome.found && hashFromScratch(filename, 0x200) == targetB_) {
                outcome.found = true;
                outcome.foundFilename = filename;
            }
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
