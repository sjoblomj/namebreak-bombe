#include "backends/self_test.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/candidate.h"
#include "engine/hash_match.h"
#include "engine/limits.h"
#include "engine/mpq_hash.h"

namespace {

// The real search's 49 characters, plus 14 more to make 63 (MAX_ALPHABET_SIZE).
const std::string kCharacters = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_#$%*:;<=>?@^`~";

struct Case {
    int alphabetSize;
    std::string prefix, suffix;
    int trailingLen;
    // The planted candidate's last character (its position in the alphabet).
    int k;
    // Where the planted candidate is: inside the searched range; its very
    // first or very last candidate (the range then starts or ends mid-row,
    // at it); the range's only candidate; or just before its start / just
    // after its end, in the same row. Or in a longer range - a few rows
    // after its start, and three row groups before its end - where a backend
    // reaches it by stepping from row to row (a CPU work item, a GPU chunk)
    // rather than by working its row out afresh.
    enum { Inside, AtStart, AtEnd, Only, JustBefore, JustAfter, InLongRange } where;
    // Which row of its row group - the alphabetSize rows that share every
    // character but the row's last (see kChunksPerGroup in the CUDA backend) -
    // the planted candidate is in: one in the middle, or the group's first or
    // last row, with the range reaching into the group before or after it.
    enum { MiddleRow, FirstRowOfGroup, LastRowOfGroup } row = MiddleRow;
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
    uint64_t row = rows / 3;
    if (c.trailingLen > 1 && c.row == Case::FirstRowOfGroup)
        row -= row % as;
    else if (c.trailingLen > 1 && c.row == Case::LastRowOfGroup)
        row += as - 1 - row % as;
    const uint64_t planted = row * as + (uint64_t) c.k;

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
        case Case::InLongRange:
            start = planted - std::min(planted, 5 * as + 2);
            end = std::min(space, planted + 3 * as * as);
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
    snprintf(where, sizeof(where), " (alphabet size %d, prefix length %zu, suffix length %zu, trailing length %d, row %llu, last character %d)",
             c.alphabetSize, c.prefix.size(), c.suffix.size(), c.trailingLen, (unsigned long long) row, c.k);
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

// Several batches in one runBatches() call, as the engine hands a backend
// several leading values at once (see SearchBackend::maxBatchesPerCall): each
// has its own prefix, so its own seeds, and its own range, starting and ending
// mid-row at a different point. The planted candidate is in batch
// `plantBatch` - inside it, at its first or last candidate, or just after its
// end, where no batch searches it with its prefix and it must not be reported.
// A backend that used one batch's seeds or edges for another, lost track of
// which batch a hit was in (it would report the wrong filename), or skipped a
// batch, fails.
//
// With `prune`, the search prunes symbol runs at every character but the
// last, and every batch starts at a trailing part beginning with '!' - after
// a symbol run of 2 in the odd batches, whose rows are then all pruned, and
// none in the even ones, which lose none: a backend that used one batch's
// pruning for another misses the planted candidate or reports a pruned one.
struct GroupedCase {
    int plantBatch; // -1: the last
    enum { Inside, AtStart, AtEnd, JustAfter } where;
    bool prune = false;
};

bool runGroupedCase(SearchBackend& backend, const GroupedCase& c, int batchCount, int alphabetSize, int trailingLen, const uint32_t* cryptTable,
                    std::string& error) {
    const std::string alphabet = kCharacters.substr(0, alphabetSize);
    const uint64_t as = (uint64_t) alphabetSize;
    uint64_t space = 1;
    for (int i = 0; i < trailingLen; ++i)
        space *= as;
    const std::string suffix = ".WAV";
    const int plantBatch = c.plantBatch < 0 ? batchCount - 1 : c.plantBatch;
    if (plantBatch >= batchCount)
        return true; // fewer batches per call than this case needs

    std::vector<BatchRequest> batches(batchCount);
    std::vector<std::string> prefixes(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        prefixes[b] = std::string("REZ\\") + kCharacters[10 + b]; // '0', '1', ... - a different leading character each
        BatchRequest& batch = batches[b];
        batch.start = std::min(space - 1, (c.prune ? alphabet.find('!') * (space / as) : space / 3) + (uint64_t) b * 7);
        if (c.prune)
            batch.params.pruneEntry.symbolRun = b % 2 == 1 ? 2 : 0;
        batch.count = std::min(space - batch.start, 2 * as + 5 + (uint64_t) b);
        memcpy(batch.params.prefix, prefixes[b].c_str(), prefixes[b].size() + 1);
        batch.params.prefixSize = (short) prefixes[b].size();
        std::pair<uint32_t, uint32_t> seeds = mpqHashWithPrefixCache_CPU(prefixes[b].c_str(), cryptTable);
        batch.params.seed1Start = seeds.first;
        batch.params.seed2Start = seeds.second;
    }
    const BatchRequest& target = batches[plantBatch];
    uint64_t planted = 0;
    switch (c.where) {
        case GroupedCase::Inside: planted = target.start + target.count / 2; break;
        case GroupedCase::AtStart: planted = target.start; break;
        case GroupedCase::AtEnd: planted = target.start + target.count - 1; break;
        case GroupedCase::JustAfter: planted = target.start + target.count; break;
    }
    if (planted >= space)
        return true; // the trailing space is too small for this case

    PruneRules rules;
    rules.symbolRuns = c.prune;
    const std::string plantedTrailing = trailingString(planted, trailingLen, alphabet);
    PruneState state = target.params.pruneEntry;
    bool survives = true;
    for (int i = 0; i + 1 < trailingLen && survives; ++i)
        survives = pruneStep_CPU(rules, state, plantedTrailing[i]);
    const bool inside = c.where != GroupedCase::JustAfter && survives;

    const std::string plantedName = prefixes[plantBatch] + plantedTrailing + suffix;
    SearchConstants constants;
    constants.alphabet = alphabet;
    constants.suffix = suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = hashFromScratch(plantedName, cryptTable, 0x100);
    constants.targetHashB = hashFromScratch(plantedName, cryptTable, 0x200);
    constants.trailingRules = rules;

    // As the engine does it (searchBatches, search.cpp): all at once, or each
    // on its own if their hits together were more than could be recorded.
    auto search = [&]() {
        Result result;
        BatchOutcome outcome = backend.runBatches(trailingLen, batches);
        if (outcome.hitCount > MAX_MATCHES) {
            for (const BatchRequest& batch : batches)
                searchChunk(backend, trailingLen, batch.start, batch.count, batch.params, result);
        } else {
            result.hits = outcome.hits;
            result.found = outcome.found;
            result.foundFilename = outcome.foundFilename;
        }
        std::sort(result.hits.begin(), result.hits.end());
        return result;
    };
    // Twice in one search: a real search has many launches, and whatever one
    // leaves behind - a hit count not reset, say - must not change the next.
    backend.beginSearch(constants);
    const Result result = search();
    const Result again = search();
    backend.endSearch();

    char where[256];
    snprintf(where, sizeof(where), " (%d batches searched together, planted in batch %d, alphabet size %d, trailing length %d%s)", batchCount,
             plantBatch, alphabetSize, trailingLen, c.prune ? ", pruning symbol runs" : "");
    if (again.hits != result.hits || again.found != result.found || again.foundFilename != result.foundFilename) {
        error = "searching the same batches again in the same search reported something else - " + std::to_string(result.hits.size()) +
                " hit(s) the first time, " + std::to_string(again.hits.size()) + " the second" + where;
        return false;
    }
    for (const std::string& hit : result.hits) {
        if (!hashAMatches(hashFromScratch(hit, cryptTable, 0x100), constants.targetHashA)) {
            error = "reported '" + hit + "', which doesn't match the target" + where;
            return false;
        }
    }
    const bool reported = std::find(result.hits.begin(), result.hits.end(), plantedName) != result.hits.end();
    if (inside && (!reported || !result.found || result.foundFilename != plantedName)) {
        error = "missed the planted candidate '" + plantedName + "'" + where;
        return false;
    }
    if (!inside && (reported || result.found)) {
        error = "reported '" + plantedName + "', which is outside the searched ranges" + (survives ? "" : ", or pruned") + where;
        return false;
    }
    return true;
}

// A search that prunes the whole candidate (SearchConstants::trailingRules):
// a candidate planted at `trailing` - every rule on, at most one backslash,
// starting from a state as if the leading characters had left a symbol run
// of 1, a '(' open and a backslash used - must be reported exactly when its
// characters, all but the last, break none of them. The range reaches two
// row groups either side, where other groups and rows are pruned or not, so
// a backend that searched the wrong ones, or used the wrong rows of one,
// misses it or reports a pruned one.
bool runPruneCase(SearchBackend& backend, int alphabetSize, const std::string& trailing, const uint32_t* cryptTable, std::string& error) {
    // With a backslash, in place of the last character.
    std::string alphabet = kCharacters.substr(0, alphabetSize);
    alphabet.back() = '\\';
    const uint64_t as = (uint64_t) alphabetSize;
    const int trailingLen = (int) trailing.size();
    uint64_t space = 1, planted = 0;
    for (char c : trailing) {
        space *= as;
        planted = planted * as + alphabet.find(c);
    }
    const std::string prefix = "REZ\\(!";
    const std::string suffix = ".WAV";
    const std::string plantedName = prefix + trailing + suffix;

    SearchConstants constants;
    constants.alphabet = alphabet;
    constants.suffix = suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = hashFromScratch(plantedName, cryptTable, 0x100);
    constants.targetHashB = hashFromScratch(plantedName, cryptTable, 0x200);
    constants.trailingRules.symbolRuns = true;
    constants.trailingRules.unopenedBrackets = true;
    constants.trailingRules.maxBackslashCount = 1;

    BatchParams params;
    memcpy(params.prefix, prefix.c_str(), prefix.size() + 1);
    params.prefixSize = (short) prefix.size();
    std::pair<uint32_t, uint32_t> seeds = mpqHashWithPrefixCache_CPU(prefix.c_str(), cryptTable);
    params.seed1Start = seeds.first;
    params.seed2Start = seeds.second;
    params.pruneEntry.symbolRun = 1;
    params.pruneEntry.open.round = 1;
    params.pruneEntry.backslashes = 1;

    // What the rules say, worked out here one character at a time.
    PruneState state = params.pruneEntry;
    bool survives = true;
    for (int i = 0; i + 1 < trailingLen && survives; ++i)
        survives = pruneStep_CPU(constants.trailingRules, state, trailing[i]);

    const uint64_t reach = 2 * as * as + 7;
    const uint64_t start = planted - std::min(planted, reach);
    const uint64_t end = std::min(space, planted + reach);
    Result result;
    const uint64_t batchSize = backend.batchSize(alphabetSize);
    backend.beginSearch(constants);
    for (uint64_t i = start; i < end && !result.found;) {
        const uint64_t chunkEnd = std::min(end, (i / batchSize + 1) * batchSize);
        searchChunk(backend, trailingLen, i, chunkEnd - i, params, result);
        i = chunkEnd;
    }
    backend.endSearch();

    const std::string where = " (pruning the whole candidate, alphabet size " + std::to_string(alphabetSize) + ", trailing length " +
                              std::to_string(trailingLen) + ")";
    const bool reported = std::find(result.hits.begin(), result.hits.end(), plantedName) != result.hits.end();
    if (survives && (!reported || !result.found || result.foundFilename != plantedName)) {
        error = "missed the planted candidate '" + plantedName + "'" + where;
        return false;
    }
    if (!survives && (reported || result.found)) {
        error = "reported '" + plantedName + "', which the pruning rules leave out" + where;
        return false;
    }
    return true;
}

// Text inserted into the trailing part (SearchConstants::trailingInsertions),
// with two backslashes next to each other pruned and the whole candidate
// pruned: a candidate planted at `trailing` must be reported, as a filename
// with the text, exactly when its characters and the text, all but the last
// character, have no two backslashes next to each other. The range reaches
// two row groups either side, so a backend that hashed the text in the wrong
// place, or not at all, misses it - and one that checked the rows' own last
// characters without the text after them reports a pruned one.
bool runInsertCase(SearchBackend& backend, int alphabetSize, const std::string& trailing, const std::vector<TrailingInsertion>& insertions,
                   const uint32_t* cryptTable, std::string& error) {
    // With a backslash, in place of the last character.
    std::string alphabet = kCharacters.substr(0, alphabetSize);
    alphabet.back() = '\\';
    const uint64_t as = (uint64_t) alphabetSize;
    const int trailingLen = (int) trailing.size();
    uint64_t space = 1, planted = 0;
    for (char c : trailing) {
        space *= as;
        planted = planted * as + alphabet.find(c);
    }
    const std::string prefix = "REZ_";
    const std::string suffix = ".WAV";
    const std::string expanded = withTrailingInsertions(trailing, insertions);
    const std::string plantedName = prefix + expanded + suffix;

    SearchConstants constants;
    constants.alphabet = alphabet;
    constants.suffix = suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = hashFromScratch(plantedName, cryptTable, 0x100);
    constants.targetHashB = hashFromScratch(plantedName, cryptTable, 0x200);
    constants.trailingRules.adjacentBackslashes = true;
    constants.trailingInsertions = insertions;

    BatchParams params;
    memcpy(params.prefix, prefix.c_str(), prefix.size() + 1);
    params.prefixSize = (short) prefix.size();
    std::pair<uint32_t, uint32_t> seeds = mpqHashWithPrefixCache_CPU(prefix.c_str(), cryptTable);
    params.seed1Start = seeds.first;
    params.seed2Start = seeds.second;

    // What the rule says, worked out here one character at a time.
    PruneState state = params.pruneEntry;
    bool survives = true;
    for (size_t i = 0; i + 1 < expanded.size() && survives; ++i)
        survives = pruneStep_CPU(constants.trailingRules, state, expanded[i]);

    const uint64_t reach = 2 * as * as + 7;
    const uint64_t start = planted - std::min(planted, reach);
    const uint64_t end = std::min(space, planted + reach);
    Result result;
    const uint64_t batchSize = backend.batchSize(alphabetSize);
    backend.beginSearch(constants);
    for (uint64_t i = start; i < end && !result.found;) {
        const uint64_t chunkEnd = std::min(end, (i / batchSize + 1) * batchSize);
        searchChunk(backend, trailingLen, i, chunkEnd - i, params, result);
        i = chunkEnd;
    }
    backend.endSearch();

    const std::string where = " (text inserted into the trailing part, alphabet size " + std::to_string(alphabetSize) + ", trailing length " +
                              std::to_string(trailingLen) + ")";
    const bool reported = std::find(result.hits.begin(), result.hits.end(), plantedName) != result.hits.end();
    if (survives && (!reported || !result.found || result.foundFilename != plantedName)) {
        error = "missed the planted candidate '" + plantedName + "'" + where;
        return false;
    }
    if (!survives && (reported || result.found)) {
        error = "reported '" + plantedName + "', which has two backslashes next to each other" + where;
        return false;
    }
    return true;
}

} // namespace

bool selfTestBackend(SearchBackend& backend, std::string& error) {
    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);

    // Alphabet sizes: the largest allowed, the default alphabet's, 42 and 43
    // (which the CUDA kernel has compiled in, rather than taking at runtime),
    // a smaller one, and one small enough that a row group's chunks have only
    // a few rows each.
    const int big = MAX_ALPHABET_SIZE;
    const int common = 49;
    const int compiled = 42;
    const int compiled2 = 43;
    const int small = 29;
    const int odd = 13;
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
        // In the first and last row of a row group, with the range reaching
        // into the next or previous group.
        {big, "REZ\\", ".WAV", window, 20, Case::Inside, Case::FirstRowOfGroup},
        {big, "REZ\\", ".WAV", window, 45, Case::Inside, Case::LastRowOfGroup},
        {big, "REZ\\", ".WAV", window, 33, Case::AtStart, Case::FirstRowOfGroup},
        {big, "REZ\\", ".WAV", window, 31, Case::AtEnd, Case::LastRowOfGroup},
        {small, "REZ\\", ".WAV", window, 7, Case::Inside, Case::LastRowOfGroup},
        {big, "Z\xC4\\", longSuffix, std::min(3, window), 12, Case::Inside, Case::FirstRowOfGroup},
        // Reached by stepping from row to row, into the last row of a group,
        // where the characters before the last wrap around.
        {big, "REZ\\", ".WAV", window, 20, Case::InLongRange, Case::LastRowOfGroup},
        {small, "REZ\\", ".WAV", window, 3, Case::InLongRange, Case::LastRowOfGroup},
        {big, "REZ\\", ".WAV", window, 20, Case::JustBefore},
        {big, "REZ\\", ".WAV", window, 20, Case::JustAfter},
        // The sizes compiled into the CUDA kernel, and an odd small one.
        {common, "REZ\\", ".WAV", window, 0, Case::Inside},
        {common, "REZ\\", ".WAV", window, 48, Case::Inside},
        {common, "REZ\\", ".WAV", window, 20, Case::AtStart},
        {common, "REZ\\", ".WAV", window, 31, Case::AtEnd},
        {common, "REZ\\", ".WAV", window, 20, Case::Inside, Case::FirstRowOfGroup},
        {common, "REZ\\", ".WAV", window, 45, Case::Inside, Case::LastRowOfGroup},
        {common, "REZ\\", ".WAV", window, 20, Case::InLongRange, Case::LastRowOfGroup},
        {common, "REZ\\", ".WAV", window, 20, Case::JustBefore},
        {common, "REZ\\", ".WAV", window, 20, Case::JustAfter},
        {common, "Z\xC4\\", longSuffix, std::min(2, window), 33, Case::Inside},
        {compiled, "REZ\\", ".WAV", window, 41, Case::Inside},
        {compiled, "REZ\\", ".WAV", window, 20, Case::AtStart},
        {compiled, "REZ\\", ".WAV", window, 31, Case::AtEnd},
        {compiled, "REZ\\", ".WAV", window, 20, Case::Inside, Case::FirstRowOfGroup},
        {compiled, "REZ\\", ".WAV", window, 40, Case::Inside, Case::LastRowOfGroup},
        {compiled, "REZ\\", ".WAV", window, 20, Case::InLongRange, Case::LastRowOfGroup},
        {compiled, "REZ\\", ".WAV", window, 20, Case::JustAfter},
        {compiled2, "REZ\\", ".WAV", window, 42, Case::Inside},
        {compiled2, "REZ\\", ".WAV", window, 20, Case::AtStart},
        {compiled2, "REZ\\", ".WAV", window, 31, Case::AtEnd},
        {compiled2, "REZ\\", ".WAV", window, 41, Case::Inside, Case::LastRowOfGroup},
        {compiled2, "REZ\\", ".WAV", window, 20, Case::InLongRange, Case::LastRowOfGroup},
        {odd, "REZ\\", ".WAV", window, 12, Case::Inside},
        {odd, "REZ\\", ".WAV", window, 0, Case::AtStart, Case::FirstRowOfGroup},
        {odd, "REZ\\", ".WAV", window, 5, Case::InLongRange, Case::LastRowOfGroup},
        {odd, "REZ\\", ".WAV", window, 7, Case::JustAfter},
    };
    for (const Case& c : cases) {
        if (c.k >= c.alphabetSize)
            continue;
        if (!runCase(backend, c, cryptTable, error))
            return false;
    }

    // Pruning the whole candidate (see runPruneCase), with the rules' state
    // from before the trailing part as if the leading characters had left a
    // symbol run of 1, a '(' open and the one backslash allowed used. Each
    // breaks - or doesn't - at a different character: in a row group's
    // characters or at the row's own last one, and never at the last, which
    // isn't checked.
    if (window >= 3) {
        // As they are at a window of 3; a longer window adds 'A's.
        const std::string pad(window - 3, 'A');
        const std::string shorter(std::max(0, window - 4), 'A');
        const std::vector<std::string> pruneCases = {
            "!" + pad + "A!",                             // a symbol run of 2: survives - and the last character isn't checked
            "!!" + pad + "A",                             // a run of 3, in the row group's characters: pruned
            window == 3 ? "!!A" : shorter + "!!!A",       // a run of 3 at the row's own last character: pruned
            ")" + pad + "AA",                             // closes the '(' left open: survives
            window == 3 ? "))A" : "A)" + shorter + ")A",  // closes one more, at the row's own last character: pruned
            "A" + pad + "))",                             // ... at the last character: survives
            "A\\" + pad + "A",                            // a second backslash: pruned
            pad + "A\\A",                                 // ... at the row's own last character: pruned
            pad + "AA\\",                                 // ... at the last character: survives
            ")" + pad + "(A",                             // a '(' right after the last open bracket is closed: survives
            "AB" + pad + "(",                             // nothing: survives
        };
        for (int size : {big, common, compiled, compiled2, small}) {
            for (const std::string& trailing : pruneCases) {
                if (!runPruneCase(backend, size, trailing, cryptTable, error))
                    return false;
            }
        }
    }

    // Text inserted into the trailing part (see runInsertCase): after the
    // row's own last character, after the row group's characters, after its
    // first one, and two at once - each with a backslash right before it in
    // the candidate (pruned) and without (survives).
    if (window >= 3) {
        std::vector<int> places = {1, 2, window - 1};
        places.erase(std::unique(places.begin(), places.end()), places.end());
        for (int size : {big, common, compiled, small}) {
            for (int charsAfter : places) {
                std::string plain(window, 'A');
                plain[0] = 'B';
                std::string beside = plain;
                beside[window - charsAfter - 1] = '\\';
                for (const std::string& trailing : {plain, beside}) {
                    if (!runInsertCase(backend, size, trailing, {{charsAfter, "\\"}}, cryptTable, error))
                        return false;
                }
            }
            if (window >= 4 && !runInsertCase(backend, size, std::string(window, 'C'), {{window - 1, "(S"}, {1, "E)"}}, cryptTable, error))
                return false;
        }
    }

    // As many batches at once as the backend takes (see runGroupedCase).
    const int batchCount = std::min(backend.maxBatchesPerCall(), (int) kCharacters.size() - 10);
    if (batchCount > 1) {
        const std::vector<GroupedCase> groupedCases = {
            {0, GroupedCase::Inside},    {1, GroupedCase::Inside},  {-1, GroupedCase::Inside},
            {-1, GroupedCase::AtStart},  {-1, GroupedCase::AtEnd},  {0, GroupedCase::AtEnd},
            {1, GroupedCase::AtStart},   {0, GroupedCase::JustAfter}, {-1, GroupedCase::JustAfter},
            {0, GroupedCase::Inside, true}, {1, GroupedCase::Inside, true}, {2, GroupedCase::AtEnd, true},
            {3, GroupedCase::AtStart, true}, {-1, GroupedCase::Inside, true},
        };
        for (int size : {big, common, compiled, compiled2}) {
            for (const GroupedCase& c : groupedCases) {
                if (!runGroupedCase(backend, c, batchCount, size, window, cryptTable, error))
                    return false;
            }
        }
    }
    return true;
}
