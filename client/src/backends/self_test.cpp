#include "backends/self_test.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/hash_match.h"
#include "engine/limits.h"
#include "engine/mpq_hash.h"

namespace {

// The real search's 49 characters, plus one to make 50 (MAX_ALPHABET_SIZE).
const std::string kCharacters = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_#";

struct Case {
    int alphabetSize;
    std::string prefix, suffix;
    int trailingLen;
    // The planted candidate's last character (its position in the alphabet).
    int k;
    // Where the planted candidate is: inside the searched range; its very
    // first or very last candidate (the range then starts or ends mid-row,
    // at it); the range's only candidate; or just before its start / just
    // after its end, in the same row.
    enum { Inside, AtStart, AtEnd, Only, JustBefore, JustAfter } where;
};

uint32_t hashFromScratch(const std::string& filename, const uint32_t* cryptTable, int offset) {
    uint32_t seed1 = 0x7FED7FED, seed2 = 0xEEEEEEEE;
    for (unsigned char ch : filename) {
        seed1 = cryptTable[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return seed1;
}

std::string trailingString(uint64_t index, int len, const std::string& alphabet) {
    std::string s(len, alphabet[0]);
    for (int i = len - 1; i >= 0; --i) {
        s[i] = alphabet[index % alphabet.size()];
        index /= alphabet.size();
    }
    return s;
}

struct Result {
    std::vector<std::string> hits;
    bool found = false;
    std::string foundFilename;
};

// One runBatch() call, searched again in halves if it had more hits than it
// could record - the same as the engine's searchChunk (search.cpp).
void searchChunk(SearchBackend& backend, int trailingLen, uint64_t start, uint64_t count, const BatchParams& params, Result& result) {
    if (result.found)
        return;
    BatchOutcome outcome = backend.runBatch(trailingLen, start, count, params);
    if (outcome.hitCount > MAX_MATCHES) {
        const uint64_t half = count / 2;
        searchChunk(backend, trailingLen, start, half, params, result);
        searchChunk(backend, trailingLen, start + half, count - half, params, result);
        return;
    }
    result.hits.insert(result.hits.end(), outcome.hits.begin(), outcome.hits.end());
    if (outcome.found) {
        result.found = true;
        result.foundFilename = outcome.foundFilename;
    }
}

bool runCase(SearchBackend& backend, const Case& c, const uint32_t* cryptTable, std::string& error) {
    const std::string alphabet = kCharacters.substr(0, c.alphabetSize);
    const uint64_t as = (uint64_t) c.alphabetSize;
    uint64_t space = 1;
    for (int i = 0; i < c.trailingLen; ++i)
        space *= as;
    const uint64_t rows = space / as;
    const uint64_t planted = (rows / 3) * as + (uint64_t) c.k;

    // A few candidates on either side, reaching into the neighbouring rows
    // where there are any, so the range starts and ends mid-row.
    const uint64_t reach = c.trailingLen > 1 ? as + 3 : 2;
    uint64_t start = 0, end = 0;
    switch (c.where) {
        case Case::Inside:
            start = planted - std::min(planted, reach);
            end = std::min(space, planted + reach + 1);
            break;
        case Case::AtStart:
            start = planted;
            end = std::min(space, planted + reach + 1);
            break;
        case Case::AtEnd:
            start = planted - std::min(planted, reach);
            end = planted + 1;
            break;
        case Case::Only:
            start = planted;
            end = planted + 1;
            break;
        case Case::JustBefore:
            start = planted + 1;
            end = std::min(space, start + reach);
            break;
        case Case::JustAfter:
            end = planted;
            start = end - std::min(end, reach);
            break;
    }

    const std::string plantedName = c.prefix + trailingString(planted, c.trailingLen, alphabet) + c.suffix;
    SearchConstants constants;
    constants.alphabet = alphabet;
    constants.suffix = c.suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = hashFromScratch(plantedName, cryptTable, 0x100);
    constants.targetHashB = hashFromScratch(plantedName, cryptTable, 0x200);

    BatchParams params;
    memcpy(params.prefix, c.prefix.c_str(), c.prefix.size() + 1);
    params.prefixSize = (short) c.prefix.size();
    std::pair<uint32_t, uint32_t> seeds = mpqHashWithPrefixCache_CPU(c.prefix.c_str(), cryptTable);
    params.seed1Start = seeds.first;
    params.seed2Start = seeds.second;

    // Cut at multiples of the backend's batch size, as the engine does.
    Result result;
    const uint64_t batchSize = backend.batchSize(c.alphabetSize);
    backend.beginSearch(constants);
    for (uint64_t i = start; i < end && !result.found;) {
        const uint64_t chunkEnd = std::min(end, (i / batchSize + 1) * batchSize);
        searchChunk(backend, c.trailingLen, i, chunkEnd - i, params, result);
        i = chunkEnd;
    }
    backend.endSearch();

    char where[256];
    snprintf(where, sizeof(where), " (alphabet size %d, prefix length %zu, suffix length %zu, trailing length %d, last character %d)",
             c.alphabetSize, c.prefix.size(), c.suffix.size(), c.trailingLen, c.k);
    for (const std::string& hit : result.hits) {
        if (hit.size() != c.prefix.size() + c.trailingLen + c.suffix.size() || hit.compare(0, c.prefix.size(), c.prefix) != 0 ||
            !hashAMatches(hashFromScratch(hit, cryptTable, 0x100), constants.targetHashA)) {
            error = "reported '" + hit + "', which doesn't match the target" + where;
            return false;
        }
    }
    const bool reported = std::find(result.hits.begin(), result.hits.end(), plantedName) != result.hits.end();
    if (c.where != Case::JustBefore && c.where != Case::JustAfter) {
        if (!reported) {
            error = "missed the planted candidate '" + plantedName + "'" + where;
            return false;
        }
        if (!result.found || result.foundFilename != plantedName) {
            error = "didn't report the planted candidate '" + plantedName + "' as matching both hashes" + where;
            return false;
        }
    } else if (reported || result.found) {
        error = "reported '" + plantedName + "', which is outside the searched range" + where;
        return false;
    }
    return true;
}

} // namespace

bool selfTestBackend(SearchBackend& backend, std::string& error) {
    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);

    const std::vector<int> sizes = backend.supportedAlphabetSizes(); // ascending; empty means any
    const int big = sizes.empty() ? MAX_ALPHABET_SIZE : sizes.back();
    const int small = sizes.empty() ? 29 : sizes.front();
    const int window = std::min(backend.windowChars(), backend.maxTrailingLen());
    const std::string longSuffix = "\\A LONGER.SUFFIX\xE9"; // longer than the CUDA kernel's compiled-in suffix lengths

    const std::vector<Case> cases = {
        {big, "REZ\\", ".WAV", window, 0, Case::Inside},
        {big, "REZ\\", ".WAV", window, 31, Case::Inside},
        {big, "REZ\\", ".WAV", window, 32, Case::Inside},
        {big, "REZ\\", ".WAV", window, big - 1, Case::Inside},
        {big, "REZ\\", "12345678", std::min(3, window), 40, Case::Inside},
        {big, "", "", 1, big / 2, Case::Inside},
        {big, "Z\xC4\\", longSuffix, std::min(2, window), 33, Case::Inside},
        {small, "REZ\\", ".WAV", window, small - 1, Case::Inside},
        {small, "", ".TXT", 1, 0, Case::Inside},
        // At the very edges of the range, where a kernel handles a row cut
        // short: its first and last candidate, and a range of one.
        {big, "REZ\\", ".WAV", window, 20, Case::AtStart},
        {big, "REZ\\", ".WAV", window, 33, Case::AtStart},
        {big, "REZ\\", ".WAV", window, 20, Case::AtEnd},
        {big, "REZ\\", ".WAV", window, 31, Case::AtEnd},
        {big, "REZ\\", ".WAV", window, 40, Case::Only},
        {big, "Z\xC4\\", longSuffix, std::min(2, window), 32, Case::AtStart},
        {small, "", "", 1, 5, Case::AtEnd},
        {big, "REZ\\", ".WAV", window, 20, Case::JustBefore},
        {big, "REZ\\", ".WAV", window, 20, Case::JustAfter},
    };
    for (const Case& c : cases) {
        if (c.k >= c.alphabetSize)
            continue;
        if (!runCase(backend, c, cryptTable, error))
            return false;
    }
    return true;
}
