#include "backends/common/dictionary_batch.h"

#include <algorithm>
#include <cstdio>
#include <random>

#include "backends/common/lowbits_filter.h"
#include "engine/dictionary.h"
#include "engine/dictionary_search.h"
#include "engine/hash_match.h"

bool makeDictionaryWordTable(const std::vector<std::string>& words, const std::vector<std::string>& separators, DictionaryWordTable& out) {
    out = DictionaryWordTable();
    uint64_t total = 0;
    for (const std::vector<std::string>* list : {&words, &separators}) {
        for (const std::string& text : *list)
            total += (text.size() + 3) / 4;
    }
    if (total > UINT32_MAX || words.size() + separators.size() > UINT32_MAX)
        return false;
    std::vector<uint32_t> order(words.size());
    for (uint32_t w = 0; w < (uint32_t) words.size(); ++w)
        order[w] = w;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return words[a].size() < words[b].size(); });
    out.chars.reserve((size_t) total);
    out.entries.reserve(words.size() + separators.size());
    out.positions.resize(words.size());
    auto add = [&](const std::string& text, uint32_t index) {
        const size_t slash = text.rfind('\\');
        out.entries.push_back({(uint32_t) out.chars.size(), (uint32_t) text.size(),
                               slash == std::string::npos ? 0 : (uint32_t) slash + 1, index});
        for (size_t i = 0; i < text.size(); i += 4) {
            uint32_t four = 0;
            for (size_t j = 0; j < 4 && i + j < text.size(); ++j)
                four |= (uint32_t) (unsigned char) text[i + j] << (8 * j);
            out.chars.push_back(four);
        }
    };
    for (uint32_t w : order) {
        out.positions[w] = (uint32_t) out.entries.size();
        add(words[w], w);
    }
    for (uint32_t s = 0; s < (uint32_t) separators.size(); ++s)
        add(separators[s], s);
    return true;
}

std::vector<uint32_t> dictionarySuffixKeys(const std::string& suffix, const uint32_t* cryptTable) {
    std::vector<uint32_t> keys;
    for (unsigned char ch : suffix) {
        keys.push_back(cryptTable[kHashAOffset + ch]);
        keys.push_back(cryptTable[kFileKeyOffset + ch]);
        keys.push_back(ch + 3u);
    }
    return keys;
}

uint32_t dictionaryFilterMask(bool basename) {
    return lowBitsFilterStateMask(kDictionaryFilterBits) & (basename ? kBasenameMatchMask : kHashAMatchMask);
}

std::vector<uint32_t> buildDictionarySuffixFilter(const std::string& suffix, const uint32_t* cryptTable, int keyOffset, uint32_t target,
                                                  uint32_t mask) {
    const uint32_t states = 1u << kDictionaryFilterBits;
    std::vector<uint32_t> filter(kDictionaryFilterWords, 0);
    for (uint32_t low1 = 0; low1 < states; ++low1) {
        for (uint32_t low2 = 0; low2 < states; ++low2) {
            uint32_t seed1 = low1, seed2 = low2;
            for (unsigned char ch : suffix) {
                seed1 = cryptTable[keyOffset + ch] ^ (seed1 + seed2);
                seed2 = seed1 + 33 * seed2 + (ch + 3u);
            }
            if ((seed1 & mask) == (target & mask)) {
                const uint32_t index = lowBitsFilterIndex(low1, low2, kDictionaryFilterBits);
                filter[index / 32] |= 1u << (index % 32);
            }
        }
    }
    return filter;
}

bool checkDictionarySuffixFilter(const std::vector<uint32_t>& filter, const std::string& suffix, const uint32_t* cryptTable, int keyOffset,
                                 uint32_t target, uint32_t mask, uint64_t seed, std::string& error) {
    if (filter.size() != kDictionaryFilterWords) {
        error = "the suffix filter has " + std::to_string(filter.size()) + " words, not " + std::to_string(kDictionaryFilterWords);
        return false;
    }
    std::mt19937_64 rng(seed);
    const uint32_t stateMask = lowBitsFilterStateMask(kDictionaryFilterBits);
    for (uint32_t index = 0; index < lowBitsFilterEntries(kDictionaryFilterBits); ++index) {
        // A state with random high bits that the index is of.
        const uint32_t seed1 = ((uint32_t) rng() & ~stateMask) | (index & stateMask);
        const uint32_t seed2 = ((uint32_t) rng() & ~stateMask) | ((index >> kDictionaryFilterBits) & stateMask);
        if (lowBitsFilterIndex(seed1, seed2, kDictionaryFilterBits) != index) {
            error = "the suffix filter's index of a state isn't the one it was made for";
            return false;
        }
        const bool matches = (continueHash({seed1, seed2}, suffix, keyOffset, cryptTable).first & mask) == (target & mask);
        const bool set = filter[index / 32] >> (index % 32) & 1;
        if (matches != set) {
            char buf[160];
            snprintf(buf, sizeof buf, "the suffix filter's bit %u is %d, but the suffix hashed from (0x%08X, 0x%08X) %s", index, set, seed1, seed2,
                     matches ? "matches" : "doesn't match");
            error = buf;
            return false;
        }
    }
    return true;
}

uint64_t planDictionaryLaunch(const std::vector<DictionaryBatch>& batches, uint32_t cellsPerSegment, uint32_t tailsPerCell,
                              std::vector<DictionaryLaunchBatch>& out) {
    out.clear();
    out.reserve(batches.size());
    uint64_t segments = 0;
    for (const DictionaryBatch& batch : batches) {
        DictionaryLaunchBatch b;
        b.seed1 = batch.seed1;
        b.seed2 = batch.seed2;
        b.basenameSeed1 = batch.basenameSeed1;
        b.basenameSeed2 = batch.basenameSeed2;
        b.firstWord = batch.firstWord;
        b.wordCount = batch.wordCount;
        b.firstSegment = (uint32_t) segments;
        b.firstTail = batch.firstTail;
        b.tailCount = batch.tailCount;
        b.tailChunks = (batch.tailCount + tailsPerCell - 1) / tailsPerCell;
        b.firstPair = batch.firstPair;
        b.pairCount = batch.pairCount;
        // Pairs no segment has part of (see DictionaryLaunchBatch).
        const uint64_t pairCells = (uint64_t) batch.wordCount * b.tailChunks, pairs = std::max<uint32_t>(batch.pairCount, 1);
        if (pairCells > cellsPerSegment) {
            b.segmentsPerPair = (uint32_t) ((pairCells + cellsPerSegment - 1) / cellsPerSegment);
            b.pairsPerSegment = 1;
        } else {
            b.segmentsPerPair = 1;
            b.pairsPerSegment = (uint32_t) std::min<uint64_t>(kDictionaryPairSlots, pairCells ? cellsPerSegment / pairCells : 1);
        }
        out.push_back(b);
        if (pairCells)
            segments += (pairs * b.segmentsPerPair + b.pairsPerSegment - 1) / b.pairsPerSegment;
    }
    return segments;
}

static_assert(kMaxDictionaryTailLength <= 8, "a tail's characters fit in two uint32s of the tail table");

std::vector<uint32_t> dictionaryTailTable(const std::vector<std::string>& tails) {
    std::vector<uint32_t> table(4 * tails.size(), 0);
    for (size_t t = 0; t < tails.size(); ++t) {
        for (size_t c = 0; c < tails[t].size(); ++c)
            table[4 * t + c / 4] |= (uint32_t) (unsigned char) tails[t][c] << (8 * (c % 4));
        table[4 * t + 2] = (uint32_t) tails[t].size();
    }
    return table;
}

uint32_t dictionaryBatchOfSegment(const std::vector<DictionaryLaunchBatch>& batches, uint64_t segment) {
    uint32_t lo = 0, hi = (uint32_t) batches.size() - 1;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1) / 2;
        if (batches[mid].firstSegment <= segment)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

bool dictionaryHitInBatches(const std::vector<DictionaryBatch>& batches, const DictionaryHit& hit) {
    if (hit.batch >= batches.size())
        return false;
    const DictionaryBatch& batch = batches[hit.batch];
    return hit.word >= batch.firstWord && hit.word - batch.firstWord < batch.wordCount && hit.tail >= batch.firstTail &&
           hit.tail - batch.firstTail < batch.tailCount &&
           (batch.pairCount == 0 || (hit.pair >= batch.firstPair && hit.pair - batch.firstPair < batch.pairCount));
}

void DictionaryHitVerifier::begin(const DictionaryConstants& constants) {
    words_ = constants.words;
    tails_ = constants.tails;
    separators_ = constants.separators;
    suffix_ = constants.suffix;
    cryptTable_.assign(constants.cryptTable, constants.cryptTable + 0x500);
    targetA_ = constants.targetHashA;
    targetB_ = constants.targetHashB;
    checkBasename_ = constants.checkBasename;
    basenameKey_ = constants.basenameKey;
    recordBasenames_ = constants.checkBasename && constants.recordBasenames;
    hashes_ = dictionaryHashes(constants);
    const size_t slash = suffix_.rfind('\\');
    suffixBasenameMatches_ = recordBasenames_ && slash != std::string::npos &&
                             basenameKeyMatches(continueBasenameHash(kInitialHashState, suffix_.substr(slash), cryptTable_.data()).first, basenameKey_);
}

std::string DictionaryHitVerifier::filenameOf(const DictionaryBatch& batch, uint32_t word, uint32_t tail, uint32_t pair) const {
    std::string filename = batch.leading;
    if (batch.pairCount)
        filename += words_[pair / separators_.size()] + separators_[pair % separators_.size()];
    return filename + words_[word] + tails_[tail] + suffix_;
}

void DictionaryHitVerifier::addHits(const std::vector<DictionaryBatch>& batches, const std::vector<DictionaryHit>& hashAHits,
                                    const std::vector<DictionaryHit>& basenameHits, DictionaryOutcome& outcome) const {
    const uint32_t* table = cryptTable_.data();
    auto nameOf = [&](const DictionaryHit& hit) { return filenameOf(batches[hit.batch], hit.word, hit.tail, hit.pair); };
    // A candidate whose hashA matches: a hashA hit, and the match if its
    // hashB does too.
    auto addHashAHit = [&](const std::string& filename) {
        outcome.hits.push_back(filename);
        if (!outcome.found && continueHash(kInitialHashState, filename, kHashBOffset, table).first == targetB_) {
            outcome.found = true;
            outcome.foundFilename = filename;
        }
    };
    for (const DictionaryHit& hit : hashAHits) {
        const std::string filename = nameOf(hit);
        const uint32_t hashA = continueHash(kInitialHashState, filename, kHashAOffset, table).first;
        if (!hashAMatches(hashA, targetA_)) {
            printf("WARNING: hashA mismatch for '%s' - the backend reported a hit, the full filename hashes to 0x%08X\n", filename.c_str(),
                   hashA);
            continue;
        }
        addHashAHit(filename);
    }
    if (!checkBasename_)
        return;
    for (const DictionaryHit& hit : basenameHits) {
        const std::string filename = nameOf(hit);
        if (!basenameKeyMatches(continueBasenameHash(kInitialHashState, filename, table).first, basenameKey_)) {
            printf("WARNING: basename hash mismatch for '%s' - the backend reported a hit, the full filename doesn't match\n",
                   filename.c_str());
            continue;
        }
        if (recordBasenames_)
            outcome.basenameHits.push_back(filename);
        // The backend worked out the basename hash alone: hashA is checked
        // here.
        if (hashes_ == DictionaryHashes::Basename && hashAMatches(continueHash(kInitialHashState, filename, kHashAOffset, table).first, targetA_))
            addHashAHit(filename);
    }
    if (suffixBasenameMatches_ && !batches.empty())
        outcome.basenameHits.push_back(filenameOf(batches[0], batches[0].firstWord, batches[0].firstTail, batches[0].firstPair));
}
