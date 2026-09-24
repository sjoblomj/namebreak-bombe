#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "backends/common/row_batch.h"
#include "engine/limits.h"

// The hot loop hashes kLanes candidates at once, as a few independent chains
// of SIMD vectors (independent, so each step of the serial hash recurrence
// has other work to overlap with). With GCC or Clang that's written with
// their vector types - 4 lanes wide, which is SSE2 on x86-64 and NEON on
// ARM, plus an 8-lane AVX2 version on x86-64 that's used when the CPU has
// it. Other compilers (MSVC) get plain loops over the lanes, which they
// vectorize as they see fit.
#if defined(__GNUC__) || defined(__clang__)
#define NAMEBREAK_VECTOR_TYPES 1
#define NAMEBREAK_ALWAYS_INLINE inline __attribute__((always_inline))
typedef uint32_t U32x4 __attribute__((vector_size(16)));
// NAMEBREAK_CPU_NO_AVX2 leaves the AVX2 version out, so the tests can
// exercise the portable one on a CPU that has AVX2 (see CMakeLists.txt).
#if defined(__x86_64__) && !defined(NAMEBREAK_CPU_NO_AVX2)
#define NAMEBREAK_AVX2 1
typedef uint32_t U32x8 __attribute__((vector_size(32)));
#endif
#else
#define NAMEBREAK_ALWAYS_INLINE inline
#endif

namespace {

// Candidates hashed side by side by the portable versions: four chains of
// 4-lane vectors.
constexpr int kLanes = 16;
// The AVX2 version hashes a whole row at once: as many 8-lane chains as the
// alphabet needs (up to 7 - more would no longer fit in its 16 registers).
constexpr int kAvx2Width = 8;
constexpr int kMaxAvx2Chains = (MAX_ALPHABET_SIZE + kAvx2Width - 1) / kAvx2Width;
static_assert(kMaxAvx2Chains <= 7, "a whole row's AVX2 chains must fit in registers");
// The alphabet padded to a whole number of lane groups; lanes past the
// alphabet's end hash harmless garbage that's never reported.
constexpr int kPaddedAlphabet = (MAX_ALPHABET_SIZE + kLanes - 1) / kLanes * kLanes;
static_assert(kPaddedAlphabet >= kMaxAvx2Chains * kAvx2Width, "the padding must cover a whole row's AVX2 chains");
// Batches smaller than this many candidates are searched on the calling
// thread - starting threads would cost more than it saves. Overridable at
// compile time so the tests can split even their small batches across
// threads (see CMakeLists.txt).
#ifndef NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD
#define NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD (1 << 16)
#endif
constexpr uint64_t kMinCandidatesPerThread = NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD;

inline void mpqStep(uint32_t& seed1, uint32_t& seed2, uint32_t key, uint32_t ord) {
    seed1 = key ^ (seed1 + seed2);
    seed2 = ord + seed1 + seed2 + (seed2 << 5) + 3;
}

// What one thread searches, and what it found.
struct RowJob {
    uint64_t from, to;          // rows [from, to) of the batch's RowRange (0 = its first row)
    std::vector<uint64_t> hits; // trailing indices, at most MAX_MATCHES of them
    uint64_t hitCount = 0;      // all hits, including those past MAX_MATCHES
};

// Everything fixed for one runBatch() call.
struct BatchContext {
    int alphabetSize;
    int suffixLen;
    uint32_t targetA;
    const uint32_t* key;       // kPaddedAlphabet entries: crypt-table key per alphabet position
    const uint32_t* ord;       // kPaddedAlphabet entries: the character itself
    const uint32_t* suffixKey; // suffixLen entries
    const uint32_t* suffixOrd;
    RowRange rows;
    int prefixDigits;          // trailingLen - 1: the characters a row shares
    uint32_t seed1Start;
    uint32_t seed2Start;
};

// Hashes candidates k0 .. k0 + Chains * (lanes per Vec) - 1 of a row whose
// shared characters left the hash state at (s1, s2), as Chains independent
// chains of Vec: stores each one's hashA in outA if any of them is the
// target, and returns whether one is.
template <typename Vec, int Chains>
NAMEBREAK_ALWAYS_INLINE bool hashLanes(const BatchContext& ctx, int k0, uint32_t s1, uint32_t s2, uint32_t* outA) {
    constexpr int kWidth = sizeof(Vec) / sizeof(uint32_t);
    constexpr int kChains = Chains;
    Vec a[kChains], b[kChains];
    const uint32_t sum = s1 + s2;
    const uint32_t bBase = s2 + (s2 << 5) + 3;
    for (int c = 0; c < kChains; ++c) {
        Vec key, ord;
        memcpy(&key, ctx.key + k0 + c * kWidth, sizeof(key));
        memcpy(&ord, ctx.ord + k0 + c * kWidth, sizeof(ord));
        a[c] = key ^ sum;
        b[c] = ord + a[c] + bBase;
    }
    for (int j = 0; j < ctx.suffixLen; ++j) {
        const uint32_t suffixKey = ctx.suffixKey[j], suffixOrd = ctx.suffixOrd[j];
        for (int c = 0; c < kChains; ++c) {
            Vec na = suffixKey ^ (a[c] + b[c]);
            b[c] = suffixOrd + na + b[c] + (b[c] << 5) + 3;
            a[c] = na;
        }
    }
    Vec hit = (Vec) (a[0] == ctx.targetA);
    for (int c = 1; c < kChains; ++c)
        hit |= (Vec) (a[c] == ctx.targetA);
    uint64_t words[sizeof(Vec) / sizeof(uint64_t)];
    memcpy(words, &hit, sizeof(words));
    uint64_t any = 0;
    for (uint64_t word : words)
        any |= word;
    if (any == 0)
        return false;
    for (int c = 0; c < kChains; ++c)
        memcpy(outA + c * kWidth, &a[c], sizeof(Vec));
    return true;
}

// The plain-loop version, for compilers without vector types: kLanes lanes.
NAMEBREAK_ALWAYS_INLINE bool hashLanesScalar(const BatchContext& ctx, int k0, uint32_t s1, uint32_t s2, uint32_t* outA) {
    uint32_t a[kLanes], b[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        a[l] = ctx.key[k0 + l] ^ (s1 + s2);
        b[l] = ctx.ord[k0 + l] + a[l] + (s2 + (s2 << 5) + 3);
    }
    for (int j = 0; j < ctx.suffixLen; ++j) {
        for (int l = 0; l < kLanes; ++l) {
            uint32_t na = ctx.suffixKey[j] ^ (a[l] + b[l]);
            b[l] = ctx.suffixOrd[j] + na + b[l] + (b[l] << 5) + 3;
            a[l] = na;
        }
    }
    bool any = false;
    for (int l = 0; l < kLanes; ++l) {
        outA[l] = a[l];
        any |= a[l] == ctx.targetA;
    }
    return any;
}

void record(RowJob& job, uint64_t trailingIndex) {
    if (job.hitCount++ < MAX_MATCHES)
        job.hits.push_back(trailingIndex);
}

// Searches rows [job.from, job.to). A row's shared characters are hashed
// incrementally, like IncrementalPrefixHasher does for the leading part:
// consecutive rows differ in their last shared character almost every time,
// so moving to the next row costs about one hash step, not prefixDigits.
// Hashes a whole row's candidates (s1, s2: the state after its shared
// characters), in groups of Group lanes; `hash` hashes one group.
template <int Group, typename HashGroup>
NAMEBREAK_ALWAYS_INLINE void searchRowsWith(const BatchContext& ctx, RowJob& job, HashGroup hash) {
    const int as = ctx.alphabetSize;

    // digit[i]: the row's i-th shared character (most significant first);
    // state1/2[d]: the hash state after the first d of them.
    std::vector<int> digit(ctx.prefixDigits);
    uint64_t rowValue = ctx.rows.firstRow + job.from;
    for (int i = ctx.prefixDigits - 1; i >= 0; --i) {
        digit[i] = (int) (rowValue % as);
        rowValue /= as;
    }
    std::vector<uint32_t> state1(ctx.prefixDigits + 1), state2(ctx.prefixDigits + 1);
    state1[0] = ctx.seed1Start;
    state2[0] = ctx.seed2Start;
    auto rehashFrom = [&](int d) {
        for (; d < ctx.prefixDigits; ++d) {
            state1[d + 1] = state1[d];
            state2[d + 1] = state2[d];
            mpqStep(state1[d + 1], state2[d + 1], ctx.key[digit[d]], ctx.ord[digit[d]]);
        }
    };
    rehashFrom(0);

    for (uint64_t i = job.from; i < job.to; ++i) {
        const uint64_t row = ctx.rows.firstRow + i;
        const uint32_t s1 = state1[ctx.prefixDigits];
        const uint32_t s2 = state2[ctx.prefixDigits];
        const int kBegin = (i == 0) ? ctx.rows.firstRowStartK : 0;
        const int kEnd = (i == ctx.rows.rowCount - 1) ? ctx.rows.lastRowEndK : as;

        if (kBegin == 0 && kEnd == as) {
            for (int k0 = 0; k0 < as; k0 += Group) {
                uint32_t a[Group];
                if (hash(k0, s1, s2, a)) {
                    for (int l = 0; l < Group && k0 + l < as; ++l) {
                        if (a[l] == ctx.targetA)
                            record(job, row * as + k0 + l);
                    }
                }
            }
        } else {
            // A partial row at the edge of the batch's range - at most two
            // per batch, so only correctness matters here.
            for (int k = kBegin; k < kEnd; ++k) {
                uint32_t a = s1, b = s2;
                mpqStep(a, b, ctx.key[k], ctx.ord[k]);
                for (int j = 0; j < ctx.suffixLen; ++j)
                    mpqStep(a, b, ctx.suffixKey[j], ctx.suffixOrd[j]);
                if (a == ctx.targetA)
                    record(job, row * as + k);
            }
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

#ifdef NAMEBREAK_AVX2
template <int Chains>
__attribute__((target("avx2"))) void searchRowsAvx2(const BatchContext& ctx, RowJob& job) {
    searchRowsWith<Chains * kAvx2Width>(ctx, job, [&](int k0, uint32_t s1, uint32_t s2, uint32_t* outA) __attribute__((always_inline)) {
        return hashLanes<U32x8, Chains>(ctx, k0, s1, s2, outA);
    });
}

// The whole row in one group: ceil(alphabetSize / 8) chains.
void searchRowsAvx2(const BatchContext& ctx, RowJob& job) {
    switch ((ctx.alphabetSize + kAvx2Width - 1) / kAvx2Width) {
        case 1: searchRowsAvx2<1>(ctx, job); break;
        case 2: searchRowsAvx2<2>(ctx, job); break;
        case 3: searchRowsAvx2<3>(ctx, job); break;
        case 4: searchRowsAvx2<4>(ctx, job); break;
        case 5: searchRowsAvx2<5>(ctx, job); break;
        case 6: searchRowsAvx2<6>(ctx, job); break;
        default: searchRowsAvx2<7>(ctx, job); break;
    }
}
#endif

void searchRows(const BatchContext& ctx, RowJob& job) {
#ifdef NAMEBREAK_AVX2
    static const bool hasAvx2 = __builtin_cpu_supports("avx2");
    if (hasAvx2) {
        searchRowsAvx2(ctx, job);
        return;
    }
#endif
#ifdef NAMEBREAK_VECTOR_TYPES
    searchRowsWith<kLanes>(ctx, job, [&](int k0, uint32_t s1, uint32_t s2, uint32_t* outA) {
        return hashLanes<U32x4, kLanes / 4>(ctx, k0, s1, s2, outA);
    });
#else
    searchRowsWith<kLanes>(ctx, job, [&](int k0, uint32_t s1, uint32_t s2, uint32_t* outA) {
        return hashLanesScalar(ctx, k0, s1, s2, outA);
    });
#endif
}

class CpuBackend : public SearchBackend {
public:
    CpuBackend() : threadCount_(std::max(1u, std::thread::hardware_concurrency())) {}

    const char* name() const override { return "cpu"; }
    std::vector<int> supportedAlphabetSizes() const override { return {}; }
    int windowChars() const override { return windowCharsOr(5); }
    // Row indices are 64-bit here, so any trailing length the engine keeps
    // within 64-bit indexing works.
    int maxTrailingLen() const override { return MAX_CANDIDATE_LEN; }
    // About a tenth of a second on a many-core CPU - short enough for pause
    // and abort to feel immediate, long enough to amortize starting threads.
    uint64_t batchSize(int alphabetSize) const override { return (uint64_t) alphabetSize * rowsPerBatchOr(1u << 22); }

    void beginSearch(const SearchConstants& constants) override {
        verifier_.begin(constants);
        alphabetSize_ = (int) constants.alphabet.size();
        targetA_ = constants.targetHashA;
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

    void endSearch() override {}

private:
    unsigned threadCount_;
    HitVerifier verifier_;
    int alphabetSize_ = 0;
    uint32_t targetA_ = 0;
    alignas(64) uint32_t key_[kPaddedAlphabet] = {};
    alignas(64) uint32_t ord_[kPaddedAlphabet] = {};
    std::vector<uint32_t> suffixKey_;
    std::vector<uint32_t> suffixOrd_;
};

BatchOutcome CpuBackend::runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) {
    BatchContext ctx;
    ctx.alphabetSize = alphabetSize_;
    ctx.suffixLen = (int) suffixKey_.size();
    ctx.targetA = targetA_;
    ctx.key = key_;
    ctx.ord = ord_;
    ctx.suffixKey = suffixKey_.data();
    ctx.suffixOrd = suffixOrd_.data();
    ctx.rows = rowRangeFor(start, count, alphabetSize_);
    ctx.prefixDigits = trailingLen - 1;
    ctx.seed1Start = params.seed1Start;
    ctx.seed2Start = params.seed2Start;

    // Contiguous slices of rows, one per thread (or all on this thread, for
    // a small batch).
    uint64_t threads = std::min<uint64_t>(threadCount_, std::max<uint64_t>(1, count / kMinCandidatesPerThread));
    threads = std::min<uint64_t>(threads, ctx.rows.rowCount);
    std::vector<RowJob> jobs(threads);
    const uint64_t rowsPerJob = (ctx.rows.rowCount + threads - 1) / threads;
    for (uint64_t t = 0; t < threads; ++t) {
        jobs[t].from = std::min(ctx.rows.rowCount, t * rowsPerJob);
        jobs[t].to = std::min(ctx.rows.rowCount, (t + 1) * rowsPerJob);
    }
    if (threads == 1) {
        searchRows(ctx, jobs[0]);
    } else {
        std::vector<std::thread> workers;
        for (uint64_t t = 1; t < threads; ++t)
            workers.emplace_back(searchRows, std::cref(ctx), std::ref(jobs[t]));
        searchRows(ctx, jobs[0]);
        for (std::thread& worker : workers)
            worker.join();
    }

    BatchOutcome outcome;
    std::vector<uint64_t> hits;
    for (const RowJob& job : jobs) {
        outcome.hitCount += (int) std::min<uint64_t>(job.hitCount, MAX_MATCHES + 1);
        hits.insert(hits.end(), job.hits.begin(), job.hits.end());
    }
    // As documented on BatchOutcome: past MAX_MATCHES, report only the count.
    if (outcome.hitCount <= MAX_MATCHES)
        verifier_.addHits(hits, trailingLen, params, outcome);
    return outcome;
}

} // namespace

std::unique_ptr<SearchBackend> makeCpuBackend(std::string&) {
    return std::make_unique<CpuBackend>();
}
