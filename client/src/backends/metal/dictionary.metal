// The Metal backend's dictionary kernel - the OpenCL one
// (backends/opencl/dictionary.cl, itself the CUDA kernel ported) in Metal's
// shading language. Embedded into the program at build time and compiled by
// Metal at runtime, once per combination of these (see metal_backend.mm):
//   SUFFIX_LEN        the suffix's length
//   HASHES            the hashes each candidate gets (dictionaryHashes in
//                     engine/backend.h): 0 hashA, 1 the basename hash, 2 both
//   WORDS_PER_THREAD  how many cells of a batch one thread hashes
//   TAILS             1 if the candidates have tails, 0 if their only tail
//                     is ""
//   TAILS_PER_CELL    how many tails a cell has at most
//   PAIR_SLOTS        kDictionaryPairSlots: how many pairs a segment has at
//                     most, whose states a threadgroup works out first
//   FILTER_BITS       kDictionaryFilterBits: how many low bits of each seed
//                     index the suffix filters (backends/common/dictionary_batch.h)
//   FILTER_WORDS      kDictionaryFilterWords: the uint32s of a filter
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                     in the tests' builds (see engine/hash_match.h)
//   BASENAME_MATCH_MASK  the same for the basename hash
//
// A launch is cut into segments, up to WORDS_PER_THREAD * the threadgroup's
// size cells of one batch - a cell a word and a chunk of up to
// TAILS_PER_CELL of the batch's tails - of one pair (a word and a separator
// before the last word, see DictionaryBatch in engine/backend.h) or of
// several whole ones, one segment per threadgroup (see DictionaryLaunchBatch
// and planDictionaryLaunch in backends/common/dictionary_batch.h). Its
// threads take neighbouring cells, a run each group of them
// (dictionaryLanesPerRun): neighbouring words with the same chunk of tails,
// the words in the word table's order (by length) if the batch has every
// word, in the list's if not. A thread hashes its word from the states
// after the batch's leading part and the cell's pair - which the
// threadgroup works out first - or without pairs, the batch's: hashA, the
// basename hash (which a '\' starts over), or both, as HASHES says - then
// each tail of its chunk from there, and then the suffix, where its filter
// lets it through. Every hashA hit, and every basename hit, is
// recorded as (batch, word, tail, pair): the host rebuilds and checks them
// - and, of a basename hit with the basename hash alone, its hashA and
// hashB (DictionaryHitVerifier).

#include <metal_stdlib>
using namespace metal;

#ifndef HASHA_MATCH_MASK
#define HASHA_MATCH_MASK 0xFFFFFFFFu
#endif
#ifndef BASENAME_MATCH_MASK
#define BASENAME_MATCH_MASK 0xFFFFFFFFu
#endif

#define HASHA_MATCHES(a, target) (((a) & (uint) (HASHA_MATCH_MASK)) == ((target) & (uint) (HASHA_MATCH_MASK)))
#define BASENAME_MATCHES(h, target) (((h) & (uint) (BASENAME_MATCH_MASK)) == ((target) & (uint) (BASENAME_MATCH_MASK)))

// Which hashes a candidate gets.
#define HASH_A (HASHES != 1)
#define HASH_BASENAME (HASHES != 0)

// lowBitsFilterIndex (backends/common/lowbits_filter.h) at FILTER_BITS,
// which this must match exactly: the low bits of seed1, then those of seed2.
#define FILTER_STATE_MASK ((1u << FILTER_BITS) - 1u)
#define FILTER_INDEX(seed1, seed2) (((seed1) & FILTER_STATE_MASK) | (((seed2) & FILTER_STATE_MASK) << FILTER_BITS))
#define FILTER_PASSES(filter, seed1, seed2) (((filter)[FILTER_INDEX(seed1, seed2) / 32] >> (FILTER_INDEX(seed1, seed2) % 32)) & 1u)

// One step of the hash, for a character whose key and value plus 3 are
// known - the usual step, its sums in another order (see
// dictionarySuffixKeys in backends/common/dictionary_batch.h).
#define STEP_PLUS3(seed1, seed2, key, ordPlus3)      \
    do {                                             \
        seed1 = (key) ^ (seed1 + seed2);             \
        seed2 = seed1 + 33u * seed2 + (ordPlus3);    \
    } while (0)

// Hashes character b (0: the lowest byte) of `four` into hashA's state, the
// basename hash's, or both.
#define STEP_CHAR(four, b)                                  \
    do {                                                    \
        const uint ch_ = ((four) >> (8 * (b))) & 0xFFu;     \
        const uint2 keys_ = lKeys[ch_];                     \
        if (HASH_A)                                         \
            STEP_PLUS3(seed1, seed2, keys_.x, ch_ + 3);     \
        if (HASH_BASENAME)                                  \
            STEP_PLUS3(key1, key2, keys_.y, ch_ + 3);       \
    } while (0)

// These two must match their namesakes in backends/common/dictionary_batch.h.
struct DictionaryLaunchBatch {
    uint seed1;
    uint seed2;
    uint basenameSeed1;
    uint basenameSeed2;
    uint firstWord;
    uint wordCount;
    uint firstSegment;
    uint firstTail;
    uint tailCount;
    uint tailChunks;
    uint firstPair;
    uint pairCount;
    uint segmentsPerPair;
    uint pairsPerSegment;
};
struct DictionaryHit {
    uint batch;
    uint word;
    uint tail;
    uint pair;
};

// Must match DictionaryArgs in metal_backend.mm.
struct DictionaryArgs {
    uint batchCount;
    uint wordCount;
    uint targetA;
    uint basenameKey;
    uint capacity;
    uint separatorCount;
};

// seed1 after the suffix, hashed on from (seed1, seed2) with its keys -
// suffixKeys[3 * i + which] for character i, which = 0 for hashA's, 1 for
// the basename hash's (see dictionarySuffixKeys).
static uint hashSuffix(uint seed1, uint seed2, constant uint* suffixKeys, int which) {
    for (int i = 0; i < SUFFIX_LEN; ++i)
        STEP_PLUS3(seed1, seed2, suffixKeys[3 * i + which], suffixKeys[3 * i + 2]);
    return seed1;
}

// The states after `batch`'s leading part and pair `pair` - its word, then
// its separator (see DictionaryBatch) - as (seed1, seed2, basenameSeed1,
// basenameSeed2): hashA's, the basename hash's (which a '\' starts over),
// or both, as HASHES says. Once a pair, not once a candidate, so a
// character at a time.
static uint4 pairStates(thread const DictionaryLaunchBatch& batch, device const uint* chars, device const uint4* entries,
                        device const uint* positions, uint wordCount, uint separatorCount, threadgroup const uint2* lKeys, uint pair) {
    uint seed1 = batch.seed1, seed2 = batch.seed2, key1 = batch.basenameSeed1, key2 = batch.basenameSeed2;
    const uint word = pair / separatorCount;
    const uint at[2] = {positions[word], wordCount + (pair - word * separatorCount)};
    for (int text = 0; text < 2; ++text) {
        const uint4 entry = entries[at[text]];
        uint four = 0;
        for (uint c = 0; c < entry.y; ++c) {
            if ((c & 3) == 0)
                four = chars[entry.x + c / 4];
            const uint ch = (four >> (8 * (c & 3))) & 0xFFu;
            const uint2 keys = lKeys[ch];
            if (HASH_A)
                STEP_PLUS3(seed1, seed2, keys.x, ch + 3);
            if (HASH_BASENAME) {
                if (ch == '\\') {
                    key1 = 0x7FED7FEDu;
                    key2 = 0xEEEEEEEEu;
                } else {
                    STEP_PLUS3(key1, key2, keys.y, ch + 3);
                }
            }
        }
    }
    return uint4(seed1, seed2, key1, key2);
}

// `entries` are DictionaryWordEntry's: (offset, length, basenameStart,
// index), the words' and then the separators'.
// `cryptKeys` is the crypt table's hashA part (0x100) and then its basename
// hash's part (0x300), 256 entries each, and `filters` the suffix filters,
// hashA's and then the basename hash's - the ones it uses copied into
// threadgroup memory. `tails` is the tails (dictionaryTailTable), which a
// SIMD-group's threads read together. `counts` is how many hits of each kind the launch
// had, every one of them, and the hits arrays the first args.capacity of
// each.
kernel void searchDictionary(constant DictionaryArgs& args [[buffer(0)]],
                             device const DictionaryLaunchBatch* batches [[buffer(1)]],
                             device const uint* chars [[buffer(2)]],
                             device const uint4* entries [[buffer(3)]],
                             device const uint* positions [[buffer(4)]],
                             device const uint* cryptKeys [[buffer(5)]],
                             device const uint* filters [[buffer(6)]],
                             constant uint* suffixKeys [[buffer(7)]],
                             device atomic_int* counts [[buffer(8)]],
                             device DictionaryHit* hashAHits [[buffer(9)]],
                             device DictionaryHit* basenameHits [[buffer(10)]],
                             device const uint4* tails [[buffer(11)]],
                             uint segment [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint lsize [[threads_per_threadgroup]]) {
    threadgroup uint2 lKeys[256];
    threadgroup uint lFilterA[HASH_A ? FILTER_WORDS : 1];
    threadgroup uint lFilterBasename[HASH_BASENAME ? FILTER_WORDS : 1];
    threadgroup uint4 lPairs[PAIR_SLOTS];
    // The threadgroup's batch, its segment's first pair (of the batch's,
    // counted from its first) and how many of its pairs it has, and its
    // cells, from its first pair's first: [lCellFirst, lCellEnd).
    threadgroup uint lBatch, lPairFirst, lPairCount, lCellFirst, lCellEnd;
    const uint cellsPerSegment = WORDS_PER_THREAD * lsize;
    for (uint i = lid; i < 256; i += lsize)
        lKeys[i] = uint2(cryptKeys[i], cryptKeys[256 + i]);
    for (uint i = lid; i < FILTER_WORDS; i += lsize) {
#if HASH_A
        lFilterA[i] = filters[i];
#endif
#if HASH_BASENAME
        lFilterBasename[i] = filters[FILTER_WORDS + i];
#endif
    }
    // The threadgroup's batch: the last whose first segment is at most its
    // own (dictionaryBatchOfSegment on the host) - and its segment of it.
    if (lid == 0) {
        uint lo = 0, hi = args.batchCount - 1;
        while (lo < hi) {
            const uint mid = lo + (hi - lo + 1) / 2;
            if (batches[mid].firstSegment <= segment)
                lo = mid;
            else
                hi = mid - 1;
        }
        const DictionaryLaunchBatch batch = batches[lo];
        const uint s = segment - batch.firstSegment, pairCells = batch.wordCount * batch.tailChunks, pairs = max(batch.pairCount, 1u);
        uint pairFirst, pairCount, cellFirst = 0, cellEnd;
        if (batch.segmentsPerPair > 1) {
            // A segment of a pair (see DictionaryLaunchBatch).
            pairFirst = s / batch.segmentsPerPair;
            pairCount = 1;
            cellFirst = (s - pairFirst * batch.segmentsPerPair) * cellsPerSegment;
            cellEnd = min(cellFirst + cellsPerSegment, pairCells);
        } else {
            // Pairs of a segment.
            pairFirst = s * batch.pairsPerSegment;
            pairCount = pairFirst < pairs ? min(batch.pairsPerSegment, pairs - pairFirst) : 0;
            cellEnd = pairCount * pairCells;
        }
        // A segment past the batch's last pair has none of its cells.
        if (pairFirst >= pairs) {
            pairCount = 0;
            cellEnd = cellFirst;
        }
        lBatch = lo;
        lPairFirst = pairFirst;
        lPairCount = pairCount;
        lCellFirst = cellFirst;
        lCellEnd = cellEnd;
        // Without pairs, the batch's states are those every cell's word is
        // hashed on from - as its one pair's.
        lPairs[0] = uint4(batch.seed1, batch.seed2, batch.basenameSeed1, batch.basenameSeed2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint batchIndex = lBatch;
    const DictionaryLaunchBatch batch = batches[batchIndex];
    const bool wholeList = batch.wordCount == args.wordCount;
    const uint pairCells = batch.wordCount * batch.tailChunks; // a pair's (all, without pairs)
    const uint cellFirst = lCellFirst, cellEnd = lCellEnd;
    if (batch.pairCount) {
        // The states after the leading part and each of the segment's pairs.
        const uint pairFirst = batch.firstPair + lPairFirst, pairs = lPairCount;
        for (uint p = lid; p < pairs; p += lsize)
            lPairs[p] = pairStates(batch, chars, entries, positions, args.wordCount, args.separatorCount, lKeys, pairFirst + p);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The segment's cells, in runs of perThread * lanes, each of `lanes`
    // threads (dictionaryLanesPerRun): the thread's are `left` cells, `lanes`
    // apart, from threadFirst on.
    const uint lanes = lsize % 32 == 0 ? 32 : (lsize & (0u - lsize));
    const uint perThread = (cellEnd - cellFirst + lsize - 1) / lsize;
    const uint threadFirst = cellFirst + lid / lanes * lanes * perThread + lid % lanes;
    uint left = threadFirst < cellEnd ? min(perThread, (cellEnd - threadFirst + lanes - 1) / lanes) : 0;
    // Its cells pair by pair: the pair's slot of lPairs, and the cell of
    // that pair's.
    uint slot = 0, within = threadFirst;
    if (within >= pairCells) {
        slot = within / pairCells;
        within -= slot * pairCells;
    }
    const uint hitPairFirst = batch.firstPair + lPairFirst;
    while (left) {
        const uint4 states = lPairs[slot];
        const uint hitPair = hitPairFirst + slot;
        // The thread's cells of the pair: `n` of them, from `within`.
        const uint n = min(left, (pairCells - within + lanes - 1) / lanes);
        left -= n;
        for (const uint stop = within + n * lanes; within < stop; within += lanes) {
            // The cell's word, and its chunk of the tails.
            uint i = within, tailFirst = batch.firstTail, tailEnd = batch.firstTail + batch.tailCount;
            if (TAILS && batch.tailChunks > 1) {
                const uint chunk = within / batch.wordCount;
                i = within - chunk * batch.wordCount;
                tailFirst += chunk * TAILS_PER_CELL;
                tailEnd = min(tailFirst + TAILS_PER_CELL, tailEnd);
            }
            const uint4 entry = entries[wholeList ? i : positions[batch.firstWord + i]];
            const uint length = entry.y, basenameStart = entry.z, index = entry.w;
            device const uint* wordChars = chars + entry.x;
            uint seed1 = states.x, seed2 = states.y;
            uint key1 = states.z, key2 = states.w;
            if (!HASH_BASENAME || basenameStart == 0) {
                // Four characters at a time, as they're stored, and then
                // the rest.
                const uint full = length / 4;
                for (uint q = 0; q < full; ++q) {
                    const uint four = wordChars[q];
                    STEP_CHAR(four, 0);
                    STEP_CHAR(four, 1);
                    STEP_CHAR(four, 2);
                    STEP_CHAR(four, 3);
                }
                const uint rest = length % 4;
                if (rest != 0) {
                    const uint four = wordChars[full];
                    STEP_CHAR(four, 0);
                    if (rest > 1)
                        STEP_CHAR(four, 1);
                    if (rest > 2)
                        STEP_CHAR(four, 2);
                }
            } else {
                // A word with a '\': its basename starts over after the
                // last one - and without hashA, what comes before that isn't
                // hashed at all.
                key1 = 0x7FED7FEDu;
                key2 = 0xEEEEEEEEu;
                const uint from = HASH_A ? 0 : basenameStart;
                uint four = 0;
                for (uint c = from; c < length; ++c) {
                    if (c == from || (c & 3) == 0)
                        four = wordChars[c / 4];
                    const uint ch = (four >> (8 * (c & 3))) & 0xFFu;
                    const uint2 keys = lKeys[ch];
                    if (HASH_A)
                        STEP_PLUS3(seed1, seed2, keys.x, ch + 3);
                    if (c >= basenameStart)
                        STEP_PLUS3(key1, key2, keys.y, ch + 3);
                }
            }
            // The suffix after tail `tailIndex`, which has left the states
            // at (s1, s2) and (k1, k2): a hit recorded where it matches.
#define CHECK_SUFFIX(s1, s2, k1, k2, tailIndex)                                                                            \
    do {                                                                                                                   \
        if (HASH_A && FILTER_PASSES(lFilterA, s1, s2) && HASHA_MATCHES(hashSuffix(s1, s2, suffixKeys, 0), args.targetA)) { \
            const int slot = atomic_fetch_add_explicit(&counts[0], 1, memory_order_relaxed);                               \
            if ((uint) slot < args.capacity) {                                                                             \
                hashAHits[slot].batch = batchIndex;                                                                        \
                hashAHits[slot].word = index;                                                                              \
                hashAHits[slot].tail = (tailIndex);                                                                        \
                hashAHits[slot].pair = hitPair;                                                                            \
            }                                                                                                              \
        }                                                                                                                  \
        if (HASH_BASENAME && FILTER_PASSES(lFilterBasename, k1, k2) &&                                                     \
            BASENAME_MATCHES(hashSuffix(k1, k2, suffixKeys, 1), args.basenameKey)) {                                       \
            const int slot = atomic_fetch_add_explicit(&counts[1], 1, memory_order_relaxed);                               \
            if ((uint) slot < args.capacity) {                                                                             \
                basenameHits[slot].batch = batchIndex;                                                                     \
                basenameHits[slot].word = index;                                                                           \
                basenameHits[slot].tail = (tailIndex);                                                                     \
                basenameHits[slot].pair = hitPair;                                                                         \
            }                                                                                                              \
        }                                                                                                                  \
    } while (0)
#if TAILS
            // Each tail on from the word's states: its characters, eight
            // at most, two uints of them.
            for (uint t = tailFirst; t < tailEnd; ++t) {
                const uint4 tail = tails[t];
                uint s1 = seed1, s2 = seed2, k1 = key1, k2 = key2;
                for (uint c = 0; c < tail.z; ++c) {
                    const uint ch = ((c < 4 ? tail.x : tail.y) >> (8 * (c & 3))) & 0xFFu;
                    const uint2 keys = lKeys[ch];
                    if (HASH_A)
                        STEP_PLUS3(s1, s2, keys.x, ch + 3);
                    if (HASH_BASENAME)
                        STEP_PLUS3(k1, k2, keys.y, ch + 3);
                }
                CHECK_SUFFIX(s1, s2, k1, k2, t);
            }
#else
            CHECK_SUFFIX(seed1, seed2, key1, key2, tailFirst);
#endif
#undef CHECK_SUFFIX
        }
        // On to the thread's next pair - or one after that, if a pair has
        // fewer cells than `lanes`.
        if (!left)
            break;
        within -= pairCells;
        ++slot;
        if (within >= pairCells) {
            const uint pairs = within / pairCells;
            slot += pairs;
            within -= pairs * pairCells;
        }
    }
}
