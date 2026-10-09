#include "backends/self_test.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "engine/candidate.h"
#include "engine/dictionary_search.h"
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

// The rules of a search that prunes the whole candidate, the prefix, and the
// state the leading characters leave the rules in (BatchParams::pruneEntry).
struct PruneSetup {
    std::string prefix;
    PruneRules rules;
    PruneState entry;
    // Which way a backend that can search either (CUDA, OpenCL) does - see
    // SearchConstants::listWalking. The cases run both ways.
    SearchConstants::ListWalking walking = SearchConstants::ListWalking::Auto;
};

// A search that prunes the whole candidate (SearchConstants::trailingRules):
// a candidate planted at `trailing` must be reported exactly when its
// characters, all but the last, break none of `setup`'s rules - and leave
// the last room for the backslashes minBackslashCount still asks for. The
// range reaches two row groups either side, where other groups and rows are
// pruned or not, so a backend that searched the wrong ones, or used the
// wrong rows of one, misses it or reports a pruned one.
bool runPruneCase(SearchBackend& backend, int alphabetSize, const std::string& trailing, const PruneSetup& setup, const uint32_t* cryptTable,
                  std::string& error) {
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
    const std::string& prefix = setup.prefix;
    const std::string suffix = ".WAV";
    const std::string plantedName = prefix + trailing + suffix;

    SearchConstants constants;
    constants.alphabet = alphabet;
    constants.suffix = suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = hashFromScratch(plantedName, cryptTable, 0x100);
    constants.targetHashB = hashFromScratch(plantedName, cryptTable, 0x200);
    constants.trailingRules = setup.rules;
    constants.listWalking = setup.walking;

    BatchParams params;
    memcpy(params.prefix, prefix.c_str(), prefix.size() + 1);
    params.prefixSize = (short) prefix.size();
    std::pair<uint32_t, uint32_t> seeds = mpqHashWithPrefixCache_CPU(prefix.c_str(), cryptTable);
    params.seed1Start = seeds.first;
    params.seed2Start = seeds.second;
    params.pruneEntry = setup.entry;

    // What the rules say, worked out here one character at a time - and then
    // whether the last character could still make up the backslashes.
    PruneState state = params.pruneEntry;
    bool survives = true;
    for (int i = 0; i + 1 < trailingLen && survives; ++i)
        survives = pruneStep_CPU(constants.trailingRules, state, trailing[i]);
    if (survives)
        survives = canReachMinBackslashes_CPU(constants.trailingRules, state, 1);

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
                   SearchConstants::ListWalking walking, const uint32_t* cryptTable, std::string& error) {
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
    constants.listWalking = walking;

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

    // Alphabet sizes: the largest allowed, the default alphabet's, 40, 42 and
    // 43 (which the CUDA kernel has compiled in, rather than taking at runtime),
    // a smaller one, and one small enough that a row group's chunks have only
    // a few rows each.
    const int big = MAX_ALPHABET_SIZE;
    const int common = 49;
    const int compiled = 42;
    const int compiled2 = 43;
    const int compiled3 = 40;
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
        {compiled3, "REZ\\", ".WAV", window, 39, Case::Inside},
        {compiled3, "REZ\\", ".WAV", window, 20, Case::AtStart},
        {compiled3, "REZ\\", ".WAV", window, 31, Case::AtEnd},
        {compiled3, "REZ\\", ".WAV", window, 38, Case::Inside, Case::LastRowOfGroup},
        {compiled3, "REZ\\", ".WAV", window, 20, Case::InLongRange, Case::LastRowOfGroup},
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

    // Pruning the whole candidate (see runPruneCase): every rule but those
    // about backslashes at least, with the rules' state from before the
    // trailing part as if the leading characters had left a symbol run of 1,
    // a '(' open and the one backslash allowed used. Each breaks - or
    // doesn't - at a different character: in a row group's characters or at
    // the row's own last one, and never at the last, which isn't checked.
    // Then at least two backslashes, from none: a row that leaves its last
    // character one short survives, one that leaves it two short doesn't,
    // though its group has room for both.
    if (window >= 3) {
        PruneSetup rules;
        rules.prefix = "REZ\\(!";
        rules.rules.symbolRuns = true;
        rules.rules.unopenedBrackets = true;
        rules.rules.maxBackslashCount = 1;
        rules.entry.symbolRun = 1;
        rules.entry.open.round = 1;
        rules.entry.backslashes = 1;
        PruneSetup atLeastTwo;
        atLeastTwo.prefix = "REZ_";
        atLeastTwo.rules.minBackslashCount = 2;

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
        const std::vector<std::string> minCases = {
            "B" + pad + "AA",   // none, and the row's own last character not one: pruned
            "B" + pad + "A\\",  // ... whatever the last character
            "B" + pad + "\\A",  // the row's own last character is one: survives
            "\\" + pad + "AA",  // one in the row group's characters: survives
        };
        // Both ways a backend may search what the rules leave: these ranges
        // are a few row groups, which it would search in whichever way their
        // share of pruned groups decides - and with only a row pruned, never
        // by walking the lists. Not walking them, the kernel is the one every
        // case above tests at every size, and the hits are checked on the
        // host: fewer sizes do.
        for (SearchConstants::ListWalking walking : {SearchConstants::ListWalking::Always, SearchConstants::ListWalking::Never}) {
            rules.walking = walking;
            atLeastTwo.walking = walking;
            const std::vector<int> sizes = walking == SearchConstants::ListWalking::Always
                                               ? std::vector<int>{big, common, compiled, compiled2, compiled3, small}
                                               : std::vector<int>{big, compiled3, small};
            for (int size : sizes) {
                for (const std::string& trailing : pruneCases) {
                    if (!runPruneCase(backend, size, trailing, rules, cryptTable, error))
                        return false;
                }
                for (const std::string& trailing : minCases) {
                    if (!runPruneCase(backend, size, trailing, atLeastTwo, cryptTable, error))
                        return false;
                }
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
        // Both ways, as the pruning cases above.
        for (SearchConstants::ListWalking walking : {SearchConstants::ListWalking::Always, SearchConstants::ListWalking::Never}) {
            const std::vector<int> sizes = walking == SearchConstants::ListWalking::Always ? std::vector<int>{big, common, compiled, small}
                                                                                           : std::vector<int>{big, small};
            for (int size : sizes) {
                for (int charsAfter : places) {
                    std::string plain(window, 'A');
                    plain[0] = 'B';
                    std::string beside = plain;
                    beside[window - charsAfter - 1] = '\\';
                    for (const std::string& trailing : {plain, beside}) {
                        if (!runInsertCase(backend, size, trailing, {{charsAfter, "\\"}}, walking, cryptTable, error))
                            return false;
                    }
                }
                if (window >= 4 &&
                    !runInsertCase(backend, size, std::string(window, 'C'), {{window - 1, "(S"}, {1, "E)"}}, walking, cryptTable, error))
                    return false;
            }
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
        for (int size : {big, common, compiled, compiled2, compiled3}) {
            for (const GroupedCase& c : groupedCases) {
                if (!runGroupedCase(backend, c, batchCount, size, window, cryptTable, error))
                    return false;
            }
        }
    }
    return true;
}

namespace {

// The dictionary cases' words: 9,000 of every length from 1 to 23 letters - more than a GPU backend's thread block takes of one batch, so
// that a batch of every word is split between several - and a few odd ones:
// with a '\' (in the middle, first and last), with characters past ASCII, a
// long one, and twenty that all end with the basename "FLOOD". Sorted, as a
// search's are.
std::vector<std::string> dictionarySelfTestWords() {
    std::set<std::string> words;
    uint32_t state = 12345;
    auto next = [&]() {
        state = state * 1103515245u + 12345u;
        return state >> 16;
    };
    while (words.size() < 9000) {
        std::string word;
        const int length = 1 + (int) (next() % 23);
        for (int c = 0; c < length; ++c)
            word += (char) ('A' + next() % 26);
        words.insert(word);
    }
    for (const char* odd : {"AB\\CD", "\\EF", "GH\\", "\xC9T\xE9", "LONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONG"})
        words.insert(odd);
    for (int i = 0; i < 20; ++i)
        words.insert("ZZ" + std::to_string(10 + i) + "\\FLOOD");
    return std::vector<std::string>(words.begin(), words.end());
}

// A batch of a dictionary case: its leading part, and its words.
struct DictionarySelfTestBatch {
    std::string leading;
    uint32_t firstWord;
    uint32_t wordCount;
};

DictionaryBatch dictionaryBatchOf(const DictionarySelfTestBatch& b, const uint32_t* cryptTable) {
    DictionaryBatch batch;
    batch.leading = b.leading;
    const HashState a = continueHash(kInitialHashState, b.leading, kHashAOffset, cryptTable);
    const HashState basename = continueBasenameHash(kInitialHashState, b.leading, cryptTable);
    batch.seed1 = a.first;
    batch.seed2 = a.second;
    batch.basenameSeed1 = basename.first;
    batch.basenameSeed2 = basename.second;
    batch.firstWord = b.firstWord;
    batch.wordCount = b.wordCount;
    return batch;
}

// What a dictionary case knows of the planted candidate's key (see
// dictionaryHashes).
enum class DictionaryCaseKey {
    None,
    Recorded,        // its key: the basenames hashed alone, and recorded
    Unrecorded,      // its key, the basenames hashed alone, not recorded
    EveryHashA,      // its key, every candidate's hashA hashed too
    WrongEveryHashA, // another basename's key, every hashA hashed too
};

// One dictionary case: `batches`, searched in one call, with the candidate
// made of batch `plantBatch`'s leading part, word `plantWord` and `suffix`
// as the target - and its basename's hash as the key, as `key` says. It
// must be found (as matching both hashes, and its basename as matching the
// key, if the basenames are recorded) exactly when the word is one of the
// batch's.
bool runDictionaryCase(SearchBackend& backend, const std::vector<std::string>& words, const std::vector<DictionarySelfTestBatch>& batches,
                       const std::string& suffix, int plantBatch, uint32_t plantWord, DictionaryCaseKey key, const uint32_t* cryptTable,
                       std::string& error) {
    const DictionarySelfTestBatch& planted = batches[plantBatch];
    const std::string plantedName = planted.leading + words[plantWord] + suffix;
    const size_t slash = plantedName.rfind('\\');
    const std::string plantedBasename = slash == std::string::npos ? plantedName : plantedName.substr(slash + 1);
    const bool inside = plantWord >= planted.firstWord && plantWord - planted.firstWord < planted.wordCount;

    DictionaryConstants constants;
    constants.words = words;
    constants.suffix = suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = hashFromScratch(plantedName, cryptTable, kHashAOffset);
    constants.targetHashB = hashFromScratch(plantedName, cryptTable, kHashBOffset);
    constants.checkBasename = key != DictionaryCaseKey::None;
    constants.basenameKey = hashFromScratch(key == DictionaryCaseKey::WrongEveryHashA ? "NO SUCH BASENAME" : plantedBasename, cryptTable,
                                            kFileKeyOffset);
    constants.recordBasenames = key != DictionaryCaseKey::Unrecorded;
    constants.recordHashAMatches = key == DictionaryCaseKey::EveryHashA || key == DictionaryCaseKey::WrongEveryHashA;
    std::vector<DictionaryBatch> calls;
    for (const DictionarySelfTestBatch& b : batches)
        calls.push_back(dictionaryBatchOf(b, cryptTable));
    backend.beginDictionarySearch(constants);
    const DictionaryOutcome outcome = backend.runDictionaryBatches(calls);
    backend.endDictionarySearch();

    static const char* const keyNames[] = {"", ", its key", ", its key, basenames not recorded", ", its key, every hashA",
                                           ", another key, every hashA"};
    char where[200];
    snprintf(where, sizeof where, " (dictionary search: %zu batch(es), planted in batch %d, word %u of %zu, suffix length %zu%s)", batches.size(),
             plantBatch, plantWord, words.size(), suffix.size(), keyNames[(int) key]);
    // Every candidate of a batch, in a set, to check that a reported
    // filename is one.
    auto isCandidate = [&](const std::string& filename) {
        for (const DictionarySelfTestBatch& b : batches) {
            if (filename.compare(0, b.leading.size(), b.leading) != 0 || filename.size() < b.leading.size() + suffix.size() ||
                filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0)
                continue;
            const std::string word = filename.substr(b.leading.size(), filename.size() - b.leading.size() - suffix.size());
            const auto it = std::lower_bound(words.begin(), words.end(), word);
            if (it != words.end() && *it == word && (uint32_t) (it - words.begin()) - b.firstWord < b.wordCount)
                return true;
        }
        return false;
    };
    for (const std::string& hit : outcome.hits) {
        if (!isCandidate(hit) || !hashAMatches(hashFromScratch(hit, cryptTable, kHashAOffset), constants.targetHashA)) {
            error = "reported '" + hit + "', which isn't a candidate matching the target" + where;
            return false;
        }
    }
    const bool reported = std::find(outcome.hits.begin(), outcome.hits.end(), plantedName) != outcome.hits.end();
    if (inside && !reported) {
        error = "missed the planted candidate '" + plantedName + "'" + where;
        return false;
    }
    if (inside && (!outcome.found || outcome.foundFilename != plantedName)) {
        error = "didn't report the planted candidate '" + plantedName + "' as matching both hashes" + where;
        return false;
    }
    if (!inside && (reported || outcome.found)) {
        error = "reported '" + plantedName + "', which isn't one of the batch's candidates" + where;
        return false;
    }
    if (!constants.checkBasename || !constants.recordBasenames) {
        if (!outcome.basenameHits.empty()) {
            error = std::string("reported a basename match without being asked to record basenames") + where;
            return false;
        }
        return true;
    }
    for (const std::string& hit : outcome.basenameHits) {
        const size_t s = hit.rfind('\\');
        const std::string basename = s == std::string::npos ? hit : hit.substr(s + 1);
        if (!isCandidate(hit) || !basenameKeyMatches(hashFromScratch(basename, cryptTable, kFileKeyOffset), constants.basenameKey)) {
            error = "reported '" + hit + "' as a basename match, which isn't a candidate whose basename matches the key" + where;
            return false;
        }
    }
    if (key == DictionaryCaseKey::WrongEveryHashA)
        return true;
    // Every candidate whose basename is the planted one's must be reported -
    // unless the suffix has a '\', and every candidate has that basename:
    // then one is enough.
    if (inside && suffix.find('\\') == std::string::npos &&
        std::find(outcome.basenameHits.begin(), outcome.basenameHits.end(), plantedName) == outcome.basenameHits.end()) {
        error = "missed the basename of the planted candidate '" + plantedName + "'" + where;
        return false;
    }
    if (inside && outcome.basenameHits.empty()) {
        error = "missed the basename every candidate has ('" + plantedBasename + "')" + where;
        return false;
    }
    return true;
}

} // namespace

bool selfTestDictionaryBackend(SearchBackend& backend, std::string& error) {
    if (!backend.supportsDictionary()) {
        error = std::string("the ") + backend.name() + " backend can't search dictionaries";
        return false;
    }
    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);
    const std::vector<std::string> words = dictionarySelfTestWords();
    const uint32_t count = (uint32_t) words.size();
    auto indexOf = [&](const std::string& word) { return (uint32_t) (std::lower_bound(words.begin(), words.end(), word) - words.begin()); };
    const std::string wav = ".WAV";

    // Every word, in one batch: the first and last word, words on both sides
    // of where a GPU backend's thread blocks split a batch (of 8,192 words in
    // the list's order, or in the order of their lengths), and the odd words.
    // Its leading part doesn't end with a '\', so that a word with one has
    // its basename start over from another state than the batch's.
    const std::vector<DictionarySelfTestBatch> all = {{"MUSIC\\BG_", 0, count}};
    std::vector<uint32_t> planted = {0, 1, 255, 256, 1023, 1024, 4095, 4096, 8191, 8192, count / 2, count - 2, count - 1};
    for (const char* odd : {"AB\\CD", "\\EF", "GH\\", "\xC9T\xE9", "LONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONGWORDLONG", "ZZ10\\FLOOD"})
        planted.push_back(indexOf(odd));
    {
        // By their lengths: the words around 8,192 in that order.
        std::vector<uint32_t> byLength(count);
        for (uint32_t w = 0; w < count; ++w)
            byLength[w] = w;
        std::stable_sort(byLength.begin(), byLength.end(), [&](uint32_t a, uint32_t b) { return words[a].size() < words[b].size(); });
        for (uint32_t at : {0u, 8191u, 8192u, count - 1})
            planted.push_back(byLength[at]);
    }
    // Each with the basenames hashed alone, and with hashA too.
    for (uint32_t w : planted) {
        for (DictionaryCaseKey key : {DictionaryCaseKey::Recorded, DictionaryCaseKey::EveryHashA}) {
            if (w < count && !runDictionaryCase(backend, words, all, wav, 0, w, key, cryptTable, error))
                return false;
        }
    }

    // Some of the words, in the list's order: a batch of more than 8,192
    // words, from its first word to its last, with the words just outside
    // it - and a batch of one.
    const std::vector<DictionarySelfTestBatch> part = {{"REZ\\CRDT_", 100, 8500}};
    for (uint32_t w : {99u, 100u, 101u, 100u + 8191, 100u + 8192, 100u + 8499, 100u + 8500}) {
        if (!runDictionaryCase(backend, words, part, wav, 0, w, DictionaryCaseKey::Recorded, cryptTable, error))
            return false;
    }
    const std::vector<DictionarySelfTestBatch> one = {{"A", 4321, 1}};
    for (uint32_t w : {4320u, 4321u, 4322u}) {
        if (!runDictionaryCase(backend, words, one, wav, 0, w, DictionaryCaseKey::Recorded, cryptTable, error))
            return false;
    }

    // Several batches at once, each with its own leading part - with a '\'
    // at the end, none, a character past ASCII, none at all - and words.
    const std::vector<DictionarySelfTestBatch> several = {
        {"X\\Y\\", 0, count}, {"", 5, 13}, {"A\xC4" "B", 1000, 101}, {"MUSIC\\BG_", 0, 1}, {"T_", count - 10, 10}};
    const struct { int batch; uint32_t word; } severalPlanted[] = {
        {0, 0}, {0, count - 1}, {1, 5}, {1, 17}, {1, 4}, {1, 18}, {2, 1000}, {2, 1100}, {2, 999}, {2, 1101}, {3, 0}, {3, 1},
        {4, count - 10}, {4, count - 1}, {4, count - 11},
    };
    for (const auto& p : severalPlanted) {
        if (!runDictionaryCase(backend, words, several, wav, p.batch, p.word, DictionaryCaseKey::Recorded, cryptTable, error))
            return false;
    }

    // Many batches, of a word or a few each: the first, a middle and the
    // last.
    std::vector<DictionarySelfTestBatch> many;
    for (uint32_t b = 0; b < 300; ++b)
        many.push_back({"M\\" + std::to_string(b) + "_", (b * 29) % (count - 3), 1 + b % 3});
    for (int b : {0, 1, 150, 298, 299}) {
        const uint32_t last = many[b].firstWord + many[b].wordCount - 1;
        if (!runDictionaryCase(backend, words, many, wav, b, last, DictionaryCaseKey::Recorded, cryptTable, error))
            return false;
    }

    // Other suffixes - none, of a character, longer than the CUDA kernel has
    // compiled in, longer than its constant memory holds, and with
    // characters past ASCII - each hashing the basenames alone, and hashA
    // too.
    const std::string veryLong(100, 'S');
    for (const std::string& suffix : {std::string(), std::string("X"), std::string(".A LONGER SUFFIX\xE9"), veryLong}) {
        for (uint32_t w : {0u, 8192u, count - 1}) {
            for (DictionaryCaseKey key : {DictionaryCaseKey::Recorded, DictionaryCaseKey::EveryHashA}) {
                if (!runDictionaryCase(backend, words, all, suffix, 0, w, key, cryptTable, error))
                    return false;
            }
        }
    }
    // Without the key; with it, its basenames not recorded; and with
    // another basename's key, every hashA hashed all the same - which must
    // find the planted candidate however wrong the key is.
    for (DictionaryCaseKey key : {DictionaryCaseKey::None, DictionaryCaseKey::Unrecorded, DictionaryCaseKey::WrongEveryHashA}) {
        for (uint32_t w : {0u, indexOf("AB\\CD"), indexOf("ZZ10\\FLOOD"), count - 1}) {
            if (!runDictionaryCase(backend, words, several, wav, 0, w, key, cryptTable, error) ||
                !runDictionaryCase(backend, words, several, wav, 2, 1050, key, cryptTable, error))
                return false;
        }
    }
    // A suffix with a '\': every candidate's basename is its end.
    for (uint32_t w : {0u, count / 3}) {
        if (!runDictionaryCase(backend, words, several, "\\Z.TXT", 0, w, DictionaryCaseKey::Recorded, cryptTable, error))
            return false;
    }

    // More basename matches than a GPU backend has room for at first
    // (NAMEBREAK_DICTIONARY_HIT_CAPACITY): the twenty words ending with
    // "\FLOOD" in 210 batches. Every one must be reported.
    {
        const uint32_t flood = indexOf("ZZ10\\FLOOD");
        std::vector<DictionarySelfTestBatch> floods;
        for (int b = 0; b < 210; ++b)
            floods.push_back({"F" + std::to_string(b), flood, 20});
        std::vector<DictionaryBatch> calls;
        for (const DictionarySelfTestBatch& b : floods)
            calls.push_back(dictionaryBatchOf(b, cryptTable));
        DictionaryConstants constants;
        constants.words = words;
        constants.suffix = wav;
        constants.cryptTable = cryptTable;
        constants.targetHashA = 0;
        constants.targetHashB = 0;
        constants.checkBasename = true;
        constants.basenameKey = hashFromScratch("FLOOD.WAV", cryptTable, kFileKeyOffset);
        backend.beginDictionarySearch(constants);
        const DictionaryOutcome outcome = backend.runDictionaryBatches(calls);
        // And the same call again: what the first left behind mustn't
        // change what the second finds.
        const DictionaryOutcome again = backend.runDictionaryBatches(calls);
        backend.endDictionarySearch();
        std::vector<std::string> expected;
        for (const DictionarySelfTestBatch& b : floods) {
            for (uint32_t w = flood; w < flood + 20; ++w)
                expected.push_back(b.leading + words[w] + wav);
        }
        std::sort(expected.begin(), expected.end());
        for (const DictionaryOutcome* o : {&outcome, &again}) {
            std::vector<std::string> got = o->basenameHits;
            std::sort(got.begin(), got.end());
            if (got != expected) {
                error = "reported " + std::to_string(got.size()) + " basename matches of " + std::to_string(expected.size()) +
                        " (dictionary search: 20 words with the basename FLOOD.WAV in 210 batches)";
                return false;
            }
        }
    }
    return true;
}
