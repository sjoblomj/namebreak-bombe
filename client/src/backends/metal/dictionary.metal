// The Metal backend's dictionary kernel - the OpenCL one
// (backends/opencl/dictionary.cl, itself the CUDA kernel ported) in Metal's
// shading language. Embedded into the program at build time and compiled by
// Metal at runtime, once per combination of these (see metal_backend.mm):
//   SUFFIX_LEN        the suffix's length
//   BASENAMES         1 if the search compares each candidate's basename to
//                     a key (DictionaryHitVerifier::candidatesHaveBasenames)
//   WORDS_PER_THREAD  how many words of a batch one thread hashes
//   FILTER_BITS       kDictionaryFilterBits: how many low bits of each seed
//                     index the suffix filters (backends/common/dictionary_batch.h)
//   FILTER_WORDS      kDictionaryFilterWords: the uint32s of a filter
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                     in the tests' builds (see engine/hash_match.h)
//   BASENAME_MATCH_MASK  the same for the basename hash
//
// A launch is cut into segments, up to WORDS_PER_THREAD * the threadgroup's
// size consecutive words of one batch, one per threadgroup (see
// planDictionaryLaunch in backends/common/dictionary_batch.h). Its threads
// take neighbouring words - in the word table's order (by length) if the
// batch has every word, in the list's if not - and hash each from the
// batch's state, with one hash: hashA - or with BASENAMES the basename hash
// alone, which a word with a '\' starts over after the last one, compared to
// the key: the file's name has that basename, so the host checks hashA and
// hashB of those that match (DictionaryHitVerifier). Then the suffix, where
// its filter lets it through. Every hit is recorded as (batch, word) - a
// hashA hit, or with BASENAMES a basename hit: the host rebuilds and checks
// them.

#include <metal_stdlib>
using namespace metal;

#ifndef HASHA_MATCH_MASK
#define HASHA_MATCH_MASK 0xFFFFFFFFu
#endif
#ifndef BASENAME_MATCH_MASK
#define BASENAME_MATCH_MASK 0xFFFFFFFFu
#endif

// Which hash a candidate is hashed with: its crypt table part, filter, suffix
// keys and hits are the first of each (hashA's) or the second (the basename
// hash's), and what a hit must match.
#if BASENAMES
#define HASH_PART 1
#define HASH_MATCHES(h, target) (((h) & (uint) (BASENAME_MATCH_MASK)) == ((target) & (uint) (BASENAME_MATCH_MASK)))
#else
#define HASH_PART 0
#define HASH_MATCHES(h, target) (((h) & (uint) (HASHA_MATCH_MASK)) == ((target) & (uint) (HASHA_MATCH_MASK)))
#endif

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

// Hashes character b (0: the lowest byte) of `four` into the state.
#define STEP_CHAR(four, b)                                  \
    do {                                                    \
        const uint ch_ = ((four) >> (8 * (b))) & 0xFFu;     \
        STEP_PLUS3(seed1, seed2, lKeys[ch_], ch_ + 3);      \
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
    uint unused;
};
struct DictionaryHit {
    uint batch;
    uint word;
};

// Must match DictionaryArgs in metal_backend.mm.
struct DictionaryArgs {
    uint batchCount;
    uint wordCount;
    uint targetA;
    uint basenameKey;
    uint capacity;
};

// seed1 after the suffix, hashed on from (seed1, seed2) with its keys -
// suffixKeys[3 * i + which] for character i, which = 0 for hashA's, 1 for
// the basename hash's (see dictionarySuffixKeys).
static uint hashSuffix(uint seed1, uint seed2, constant uint* suffixKeys, int which) {
    for (int i = 0; i < SUFFIX_LEN; ++i)
        STEP_PLUS3(seed1, seed2, suffixKeys[3 * i + which], suffixKeys[3 * i + 2]);
    return seed1;
}

// `entries` are DictionaryWordEntry's: (offset, length, basenameStart, index).
// `cryptKeys` is the crypt table's hashA part (0x100) and then its basename
// hash's part (0x300), 256 entries each, and `filters` the suffix filters,
// hashA's and then the basename hash's - the hash's own of each copied into
// threadgroup memory. `counts` is how many hits of each kind the launch
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
                             uint segment [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint lsize [[threads_per_threadgroup]]) {
    threadgroup uint lKeys[256];
    threadgroup uint lFilter[FILTER_WORDS];
    threadgroup uint lBatch;
    for (uint i = lid; i < 256; i += lsize)
        lKeys[i] = cryptKeys[256 * HASH_PART + i];
    for (uint i = lid; i < FILTER_WORDS; i += lsize)
        lFilter[i] = filters[FILTER_WORDS * HASH_PART + i];
    // The threadgroup's batch: the last whose first segment is at most its
    // own (dictionaryBatchOfSegment on the host).
    if (lid == 0) {
        uint lo = 0, hi = args.batchCount - 1;
        while (lo < hi) {
            const uint mid = lo + (hi - lo + 1) / 2;
            if (batches[mid].firstSegment <= segment)
                lo = mid;
            else
                hi = mid - 1;
        }
        lBatch = lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint batchIndex = lBatch;
    const DictionaryLaunchBatch batch = batches[batchIndex];
    const bool wholeList = batch.wordCount == args.wordCount;
    const uint wordsPerSegment = WORDS_PER_THREAD * lsize;
    const uint first = (segment - batch.firstSegment) * wordsPerSegment;
    const uint end = min(first + wordsPerSegment, batch.wordCount);
    for (uint i = first + lid; i < end; i += lsize) {
        const uint4 entry = entries[wholeList ? i : positions[batch.firstWord + i]];
        const uint length = entry.y, basenameStart = entry.z, index = entry.w;
        device const uint* wordChars = chars + entry.x;
        uint seed1 = BASENAMES ? batch.basenameSeed1 : batch.seed1;
        uint seed2 = BASENAMES ? batch.basenameSeed2 : batch.seed2;
        if (!BASENAMES || basenameStart == 0) {
            // Four characters at a time, as they're stored, and then the
            // rest.
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
            // A word with a '\': its basename starts over after the last
            // one, and what comes before that isn't hashed at all.
            seed1 = 0x7FED7FEDu;
            seed2 = 0xEEEEEEEEu;
            uint four = 0;
            for (uint c = basenameStart; c < length; ++c) {
                if (c == basenameStart || (c & 3) == 0)
                    four = wordChars[c / 4];
                STEP_CHAR(four, c & 3);
            }
        }
        if (FILTER_PASSES(lFilter, seed1, seed2) &&
            HASH_MATCHES(hashSuffix(seed1, seed2, suffixKeys, HASH_PART), BASENAMES ? args.basenameKey : args.targetA)) {
            const int slot = atomic_fetch_add_explicit(&counts[HASH_PART], 1, memory_order_relaxed);
            if ((uint) slot < args.capacity) {
                device DictionaryHit* const hits = BASENAMES ? basenameHits : hashAHits;
                hits[slot].batch = batchIndex;
                hits[slot].word = index;
            }
        }
    }
}
