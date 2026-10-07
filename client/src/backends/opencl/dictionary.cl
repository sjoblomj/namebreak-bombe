// The OpenCL backend's dictionary kernel - the CUDA backend's
// dictionaryKernel (backends/cuda/cuda_backend.cu), ported. Embedded into the
// program at build time and compiled by the OpenCL driver at runtime, once
// per combination of these (see opencl_backend.cpp):
//   SUFFIX_LEN        the suffix's length
//   BASENAMES         1 if the search compares each candidate's basename to
//                     a key (DictionaryHitVerifier::candidatesHaveBasenames)
//   WORDS_PER_THREAD  how many words of a batch one work-item hashes
//   FILTER_BITS       kDictionaryFilterBits: how many low bits of each seed
//                     index the suffix filters (backends/common/dictionary_batch.h)
//   FILTER_WORDS      kDictionaryFilterWords: the uint32s of a filter
//   HASHA_MATCH_MASK  the bits of hashA a hit must match - all of them, except
//                     in the tests' builds (see engine/hash_match.h)
//
// A dictionary search (engine/dictionary_search.h) hands the backend batches:
// a leading part's hash states, followed by a run of the word list's words
// and then the suffix - one candidate per word. A launch is cut into
// segments, up to WORDS_PER_THREAD * the work-group's size consecutive words
// of one batch, one per work-group (see planDictionaryLaunch in
// backends/common/dictionary_batch.h). Its work-items take neighbouring
// words - in the word table's order (by length, so that a wavefront's words
// mostly have the same length) if the batch has every word, in the list's if
// not - and hash each from the batch's states: hashA, and with BASENAMES the
// basename hash too, which a word with a '\' starts over after the last one.
// Then the suffix, where its filter lets it through. Every hashA hit, and
// every basename hit, is recorded as (batch, word): the host rebuilds and
// checks them.

#ifndef HASHA_MATCH_MASK
#define HASHA_MATCH_MASK 0xFFFFFFFFu
#endif
#define HASHA_MATCHES(a, target) (((a) & HASHA_MATCH_MASK) == ((target) & HASHA_MATCH_MASK))

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

// Hashes character b (0: the lowest byte) of `four` into hashA's state -
// and with BASENAMES the basename hash's.
#if BASENAMES
#define STEP_CHAR(four, b)                                  \
    do {                                                    \
        const uint ch_ = ((four) >> (8 * (b))) & 0xFFu;     \
        const uint2 keys_ = lKeys[ch_];                     \
        STEP_PLUS3(seed1, seed2, keys_.x, ch_ + 3);         \
        STEP_PLUS3(key1, key2, keys_.y, ch_ + 3);           \
    } while (0)
#else
#define STEP_CHAR(four, b)                                  \
    do {                                                    \
        const uint ch_ = ((four) >> (8 * (b))) & 0xFFu;     \
        STEP_PLUS3(seed1, seed2, lKeys[ch_].x, ch_ + 3);    \
    } while (0)
#endif

// These two must match their namesakes in backends/common/dictionary_batch.h.
typedef struct {
    uint seed1;
    uint seed2;
    uint basenameSeed1;
    uint basenameSeed2;
    uint firstWord;
    uint wordCount;
    uint firstSegment;
    uint unused;
} DictionaryLaunchBatch;
typedef struct {
    uint batch;
    uint word;
} DictionaryHit;

// seed1 after the suffix, hashed on from (seed1, seed2) with its keys -
// suffixKeys[3 * i + which] for character i, which = 0 for hashA's, 1 for
// the basename hash's (see dictionarySuffixKeys).
uint hashSuffix(uint seed1, uint seed2, __constant const uint* suffixKeys, int which) {
    for (int i = 0; i < SUFFIX_LEN; ++i)
        STEP_PLUS3(seed1, seed2, suffixKeys[3 * i + which], suffixKeys[3 * i + 2]);
    return seed1;
}

// `entries` are DictionaryWordEntry's: (offset, length, basenameStart, index).
// `cryptKeys` is the crypt table's hashA part (0x100) and then its basename
// hash's part (0x300), 256 entries each, and `filters` the suffix filters,
// hashA's and then the basename hash's - all copied into local memory.
// `counts` is how many hits of each kind the launch had, every one of them,
// and the hits arrays the first `capacity` of each.
__kernel void searchDictionary(__global const DictionaryLaunchBatch* batches, uint batchCount, __global const uint* chars,
                               __global const uint4* entries, __global const uint* positions, uint wordCount,
                               __global const uint* cryptKeys, __global const uint* filters, __constant const uint* suffixKeys,
                               uint targetA, uint basenameKey, __global int* counts, __global DictionaryHit* hashAHits,
                               __global DictionaryHit* basenameHits, uint capacity) {
    __local uint2 lKeys[256];
    __local uint lFilterA[FILTER_WORDS];
#if BASENAMES
    __local uint lFilterBasename[FILTER_WORDS];
#endif
    __local uint lBatch;
    const uint lid = get_local_id(0), lsize = get_local_size(0);
    for (uint i = lid; i < 256; i += lsize)
        lKeys[i] = (uint2)(cryptKeys[i], cryptKeys[256 + i]);
    for (uint i = lid; i < FILTER_WORDS; i += lsize) {
        lFilterA[i] = filters[i];
#if BASENAMES
        lFilterBasename[i] = filters[FILTER_WORDS + i];
#endif
    }
    // The work-group's batch: the last whose first segment is at most its own
    // (dictionaryBatchOfSegment on the host).
    const uint segment = get_group_id(0);
    if (lid == 0) {
        uint lo = 0, hi = batchCount - 1;
        while (lo < hi) {
            const uint mid = lo + (hi - lo + 1) / 2;
            if (batches[mid].firstSegment <= segment)
                lo = mid;
            else
                hi = mid - 1;
        }
        lBatch = lo;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint batchIndex = lBatch;
    const DictionaryLaunchBatch batch = batches[batchIndex];
    const bool wholeList = batch.wordCount == wordCount;
    const uint wordsPerSegment = WORDS_PER_THREAD * lsize;
    const uint first = (segment - batch.firstSegment) * wordsPerSegment;
    const uint end = min(first + wordsPerSegment, batch.wordCount);
    for (uint i = first + lid; i < end; i += lsize) {
        const uint4 entry = entries[wholeList ? i : positions[batch.firstWord + i]];
        const uint length = entry.y, basenameStart = entry.z, index = entry.w;
        __global const uint* wordChars = chars + entry.x;
        uint seed1 = batch.seed1, seed2 = batch.seed2;
        uint key1 = batch.basenameSeed1, key2 = batch.basenameSeed2;
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
            // A word with a '\': its basename starts over after the last one.
            key1 = 0x7FED7FEDu;
            key2 = 0xEEEEEEEEu;
            uint four = 0;
            for (uint c = 0; c < length; ++c) {
                if ((c & 3) == 0)
                    four = wordChars[c / 4];
                const uint ch = (four >> (8 * (c & 3))) & 0xFFu;
                const uint2 keys = lKeys[ch];
                STEP_PLUS3(seed1, seed2, keys.x, ch + 3);
                if (c >= basenameStart)
                    STEP_PLUS3(key1, key2, keys.y, ch + 3);
            }
        }
        if (FILTER_PASSES(lFilterA, seed1, seed2) && HASHA_MATCHES(hashSuffix(seed1, seed2, suffixKeys, 0), targetA)) {
            const int slot = atomic_inc(&counts[0]);
            if ((uint) slot < capacity) {
                hashAHits[slot].batch = batchIndex;
                hashAHits[slot].word = index;
            }
        }
#if BASENAMES
        if (FILTER_PASSES(lFilterBasename, key1, key2) && hashSuffix(key1, key2, suffixKeys, 1) == basenameKey) {
            const int slot = atomic_inc(&counts[1]);
            if ((uint) slot < capacity) {
                basenameHits[slot].batch = batchIndex;
                basenameHits[slot].word = index;
            }
        }
#endif
    }
}
