#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "backends/common/lowbits_filter.h"
#include "backends/common/row_batch.h"
#include "engine/hash_match.h"
#include "engine/limits.h"

namespace {

// Batches smaller than this many candidates are searched on the calling
// thread - starting threads would cost more than it saves. Overridable at
// compile time so the tests can split even their small batches across
// threads (see CMakeLists.txt).
#ifndef NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD
#define NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD (1 << 16)
#endif
constexpr uint64_t kMinCandidatesPerThread = NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD;

// How many of the lookup filter's entries beginSearch checks against their
// definition before every search, as the GPU backends do (see
// checkLowBitsFilterTable).
constexpr uint32_t kFilterEntriesCheckedPerSearch = 1024;

inline void mpqStep(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = ord + seed1 + seed2 + (seed2 << 5) + 3;
}

// The lowest bit set in a non-zero mask.
inline int lowestBit(uint64_t mask) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward64(&index, mask);
    return (int) index;
#else
    return __builtin_ctzll(mask);
#endif
}

// What one thread found: the trailing index and batch of each hit, at most
// MAX_MATCHES of them.
struct ThreadHits {
    std::vector<uint64_t> trailingIndices;
    std::vector<int> batches;
    uint64_t hitCount = 0; // all hits, including those past MAX_MATCHES
};

// Everything fixed for one runBatch() call.
struct BatchContext {
    int alphabetSize;
    int suffixLen;
    uint32_t targetA;
    const uint32_t* key;       // crypt-table key per alphabet position
    const uint32_t* ord;       // the character itself
    const uint32_t* suffixKey; // suffixLen entries
    const uint32_t* suffixOrd;
    const char* alphabet;      // the characters themselves, for the pruning rules
    const uint64_t* table;     // this search's lookup filter (backends/common/lowbits_filter.h)
    RowRange rows;
    int prefixDigits;          // trailingLen - 1: the characters a row shares
    uint32_t seed1Start;
    uint32_t seed2Start;
    const PruneRules* rules;   // SearchConstants::trailingRules
    PruneState pruneEntry;     // BatchParams::pruneEntry
};

void record(ThreadHits& found, int batch, uint64_t trailingIndex) {
    if (found.hitCount++ < MAX_MATCHES) {
        found.trailingIndices.push_back(trailingIndex);
        found.batches.push_back(batch);
    }
}

// Searches rows [from, to) of batch `batch` (0 = its first row). A row's shared characters are hashed
// incrementally, like IncrementalPrefixHasher does for the leading part:
// consecutive rows differ in their last shared character almost every time,
// so moving to the next row costs about one hash step, not prefixDigits.
// Then, as the GPU kernels do, one lookup in this search's filter table
// gives the set of last characters whose hashA has the target's low bits -
// always including any that matches the target (README.md's "The lookup
// filter" has why) - and only those are hashed in full: about
// alphabetSize / 2^kLowBitsFilterBits of a row's candidates.
void searchRows(const BatchContext& ctx, int batch, uint64_t from, uint64_t to, ThreadHits& found) {
    const int as = ctx.alphabetSize;

    // digit[i]: the row's i-th shared character (most significant first);
    // state1/2[d]: the hash state after the first d of them.
    std::vector<int> digit(ctx.prefixDigits);
    uint64_t rowValue = ctx.rows.firstRow + from;
    for (int i = ctx.prefixDigits - 1; i >= 0; --i) {
        digit[i] = (int) (rowValue % as);
        rowValue /= as;
    }
    std::vector<uint32_t> state1(ctx.prefixDigits + 1), state2(ctx.prefixDigits + 1);
    state1[0] = ctx.seed1Start;
    state2[0] = ctx.seed2Start;
    // And, when the whole candidate is pruned, the rules' state after the
    // first d of them - a row is searched only if none of its characters
    // breaks a rule (valid[prefixDigits]).
    const bool pruning = ctx.rules->any();
    std::vector<PruneState> pruneState(ctx.prefixDigits + 1);
    std::vector<char> valid(ctx.prefixDigits + 1, 1);
    pruneState[0] = ctx.pruneEntry;
    auto rehashFrom = [&](int d) {
        for (; d < ctx.prefixDigits; ++d) {
            state1[d + 1] = state1[d];
            state2[d + 1] = state2[d];
            mpqStep(state1[d + 1], state2[d + 1], ctx.key[digit[d]], ctx.ord[digit[d]]);
            if (pruning) {
                pruneState[d + 1] = pruneState[d];
                valid[d + 1] = valid[d] && pruneStep_CPU(*ctx.rules, pruneState[d + 1], ctx.alphabet[digit[d]]);
            }
        }
    };
    rehashFrom(0);

    const uint64_t alphabetMask = (uint64_t(1) << as) - 1; // as is at most 63
    for (uint64_t i = from; i < to; ++i) {
        const uint64_t row = ctx.rows.firstRow + i;
        const uint32_t s1 = state1[ctx.prefixDigits];
        const uint32_t s2 = state2[ctx.prefixDigits];

        // Bit k: the candidate with last character k is worth hashing.
        // Restricted to the alphabet - which the table never exceeds anyway,
        // but a stray bit must not index past key/ord - and to the batch's
        // range in its first and last row. None, in a pruned row.
        uint64_t mask = valid[ctx.prefixDigits] ? ctx.table[lowBitsFilterIndex(s1, s2)] & alphabetMask : 0;
        if (i == 0)
            mask &= ~uint64_t(0) << ctx.rows.firstRowStartK;       // firstRowStartK is in [0, as)
        if (i == ctx.rows.rowCount - 1)
            mask &= (uint64_t(1) << ctx.rows.lastRowEndK) - 1;     // lastRowEndK is in [1, as]
        while (mask != 0) {
            const int k = lowestBit(mask);
            mask &= mask - 1;
            uint32_t a = s1, b = s2;
            mpqStep(a, b, ctx.key[k], ctx.ord[k]);
            for (int j = 0; j < ctx.suffixLen; ++j)
                mpqStep(a, b, ctx.suffixKey[j], ctx.suffixOrd[j]);
            if (hashAMatches(a, ctx.targetA))
                record(found, batch, row * as + k);
        }

        // Next row: increment the shared characters like an odometer, and
        // rehash from the first one that changed.
        int p = ctx.prefixDigits - 1;
        while (p >= 0 && ++digit[p] == as) {
            digit[p] = 0;
            --p;
        }
        rehashFrom(p < 0 ? 0 : p);
    }
}

// A search must not start with a filter table that is wrong: it could drop a
// match without any other sign. As in the GPU backends, this ends the
// process - a coordinator range is then reassigned when its lease runs out.
[[noreturn]] void refuseFilterTable(const std::string& detail) {
    fprintf(stderr, "INTERNAL ERROR: the lookup filter table failed its check (%s) - refusing to search with it\n", detail.c_str());
    exit(1);
}

class CpuBackend : public SearchBackend {
public:
    CpuBackend() : threadCount_(std::max(1u, std::thread::hardware_concurrency())) {}

    const char* name() const override { return "cpu"; }
    int windowChars() const override { return windowCharsOr(5); }
    // Row indices are 64-bit here, so any trailing length the engine keeps
    // within 64-bit indexing works.
    int maxTrailingLen() const override { return MAX_CANDIDATE_LEN; }
    // About a tenth of a second on a many-core CPU - short enough for pause
    // and abort to feel immediate, long enough to amortize starting threads.
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * rowsPerBatchOr(1u << 22); }

    void beginSearch(const SearchConstants& constants) override {
        verifier_.begin(constants);
        // This search's lookup filter, checked against its definition before
        // it's used, as the GPU backends do.
        table_ = buildLowBitsFilterTable(constants);
        std::string error;
        if (!checkLowBitsFilterTable(table_, constants, kFilterEntriesCheckedPerSearch, 2, std::random_device{}(), error))
            refuseFilterTable(error);
        alphabetSize_ = (int) constants.alphabet.size();
        alphabet_ = constants.alphabet;
        targetA_ = constants.targetHashA;
        rules_ = constants.trailingRules;
        std::fill(std::begin(key_), std::end(key_), 0);
        std::fill(std::begin(ord_), std::end(ord_), 0);
        for (int k = 0; k < alphabetSize_; ++k) {
            ord_[k] = (unsigned char) constants.alphabet[k];
            key_[k] = constants.cryptTable[0x100 + ord_[k]];
        }
        suffixKey_.clear();
        suffixOrd_.clear();
        for (unsigned char ch : constants.suffix) {
            suffixOrd_.push_back(ch);
            suffixKey_.push_back(constants.cryptTable[0x100 + ch]);
        }
    }

    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override;
    // Several batches - usually leading values - at once, so that its threads
    // start once for all of them (see runBatches).
    int maxBatchesPerCall() const override { return batchesPerLaunchOr(16); }
    BatchOutcome runBatches(int trailingLen, const std::vector<BatchRequest>& batches) override;

    void endSearch() override {}

private:
    unsigned threadCount_;
    HitVerifier verifier_;
    int alphabetSize_ = 0;
    std::string alphabet_;
    uint32_t targetA_ = 0;
    PruneRules rules_;
    uint32_t key_[MAX_ALPHABET_SIZE] = {};
    uint32_t ord_[MAX_ALPHABET_SIZE] = {};
    std::vector<uint64_t> table_;
    std::vector<uint32_t> suffixKey_;
    std::vector<uint32_t> suffixOrd_;
};

BatchOutcome CpuBackend::runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) {
    return runBatches(trailingLen, {BatchRequest{start, count, params}});
}

BatchOutcome CpuBackend::runBatches(int trailingLen, const std::vector<BatchRequest>& requests) {
    const int batchCount = (int) requests.size();
    std::vector<BatchContext> contexts(batchCount);
    uint64_t totalRows = 0, totalCount = 0;
    for (int b = 0; b < batchCount; ++b) {
        BatchContext& ctx = contexts[b];
        ctx.alphabetSize = alphabetSize_;
        ctx.suffixLen = (int) suffixKey_.size();
        ctx.targetA = targetA_;
        ctx.key = key_;
        ctx.ord = ord_;
        ctx.suffixKey = suffixKey_.data();
        ctx.suffixOrd = suffixOrd_.data();
        ctx.alphabet = alphabet_.data();
        ctx.table = table_.data();
        ctx.rows = rowRangeFor(requests[b].start, requests[b].count, alphabetSize_);
        ctx.prefixDigits = trailingLen - 1;
        ctx.seed1Start = requests[b].params.seed1Start;
        ctx.seed2Start = requests[b].params.seed2Start;
        ctx.rules = &rules_;
        ctx.pruneEntry = requests[b].params.pruneEntry;
        totalRows += ctx.rows.rowCount;
        totalCount += requests[b].count;
    }

    // Work items: slices of one batch's rows, about eight per thread, which
    // the threads take in turn as they finish - so that batches of any size
    // (a range's first and last are usually partial) share out evenly. All on
    // this thread, for a small call: starting threads would cost more than
    // it saves.
    uint64_t threads = std::min<uint64_t>(threadCount_, std::max<uint64_t>(1, totalCount / kMinCandidatesPerThread));
    threads = std::min<uint64_t>(threads, totalRows);
    const uint64_t rowsPerItem = std::max<uint64_t>(1, (totalRows + threads * 8 - 1) / (threads * 8));
    struct Item {
        int batch;
        uint64_t from, to;
    };
    std::vector<Item> items;
    for (int b = 0; b < batchCount; ++b) {
        for (uint64_t from = 0; from < contexts[b].rows.rowCount; from += rowsPerItem)
            items.push_back({b, from, std::min(contexts[b].rows.rowCount, from + rowsPerItem)});
    }
    std::vector<ThreadHits> found(threads);
    std::atomic<size_t> next{0};
    auto work = [&](ThreadHits& mine) {
        for (size_t i = next++; i < items.size(); i = next++)
            searchRows(contexts[items[i].batch], items[i].batch, items[i].from, items[i].to, mine);
    };
    if (threads == 1) {
        work(found[0]);
    } else {
        std::vector<std::thread> workers;
        for (uint64_t t = 1; t < threads; ++t)
            workers.emplace_back(work, std::ref(found[t]));
        work(found[0]);
        for (std::thread& worker : workers)
            worker.join();
    }

    BatchOutcome outcome;
    for (const ThreadHits& mine : found)
        outcome.hitCount += (int) std::min<uint64_t>(mine.hitCount, MAX_MATCHES + 1);
    // As documented on BatchOutcome: past MAX_MATCHES, report only the count.
    if (outcome.hitCount > MAX_MATCHES) {
        outcome.hitCount = MAX_MATCHES + 1;
        return outcome;
    }
    // Every hit, checked with its own batch's prefix (HitVerifier: rebuilt
    // from its trailing index and hashed from scratch, and hashB checked).
    for (const ThreadHits& mine : found) {
        for (size_t i = 0; i < mine.trailingIndices.size(); ++i)
            verifier_.addHits({mine.trailingIndices[i]}, trailingLen, requests[mine.batches[i]].params, outcome);
    }
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeCpuBackend(std::string&) {
    return std::make_unique<CpuBackend>();
}
