// The backward half of a meet-in-the-middle search, on an OpenCL device: the
// twin of cuda_backend.cu's meetInTheMiddleKernel, and of MeetInTheMiddleWalk
// on the CPU (see engine/meet_in_the_middle.h for what a meet-in-the-middle
// search is and what a tail means). Compiled at runtime from this source,
// embedded in the program as a string.
//
// One work-item takes one *row*: the ALPHABET_SIZE tails that share every
// character but the one rewound last - the first character of the tail,
// which is the directory's own. Everything before it, and the pruning rules
// over it, is then done once for the whole row.
//
// Compiled in (see OpenClBackend::meetInTheMiddleKernel), as the rules and the
// alphabet don't change within a search:
//   ALPHABET_SIZE                how many characters a directory is made of
//   MEET_IN_THE_MIDDLE_SEED_MASK the bits of each seed that must match the
//                                meeting point (every one of them, outside
//                                the tests)
//   RULE_ANY, RULE_SYMBOL_RUNS, RULE_ADJACENT_BACKSLASHES,
//   RULE_MAX_BACKSLASH_COUNT     the pruning rules
//   ENTRY_SYMBOL_RUN, ENTRY_BACKSLASHES, ENTRY_NEXT_IS_BACKSLASH
//                                the rules' state to the directory's right

// What the rules make of one character - as the host's MeetInTheMiddleChar
// has it, in each character's uint4 (key, ord, flags, unused).
#define MEET_IN_THE_MIDDLE_SYMBOL 1u
#define MEET_IN_THE_MIDDLE_BACKSLASH 2u

// The rules' state over the characters rewound so far (TailPruneState).
typedef struct {
    int symbolRun;
    int backslashes;
    int nextIsBackslash;
} MeetInTheMiddlePrune;

// Steps `state` over a character, to the left of the ones so far; false if it
// breaks one of the rules. Must agree with tailPruneStep
// (engine/meet_in_the_middle.cpp) exactly, or this would search other tails
// than the CPU.
inline bool meetInTheMiddlePruneStep(MeetInTheMiddlePrune* state, uint flags) {
    // Each rule compiled in or out, rather than tested at runtime - which a
    // kernel compiled for this search can do, and which keeps the device
    // compiler from warning about the constants it would otherwise test.
    if (flags & MEET_IN_THE_MIDDLE_SYMBOL) {
        ++state->symbolRun;
#if RULE_SYMBOL_RUNS
        if (state->symbolRun >= 3)
            return false;
#endif
        state->symbolRun = min(state->symbolRun, 2);
    } else {
        state->symbolRun = 0;
    }
    if (flags & MEET_IN_THE_MIDDLE_BACKSLASH) {
#if RULE_ADJACENT_BACKSLASHES
        if (state->nextIsBackslash)
            return false;
#endif
        ++state->backslashes;
#if RULE_MAX_BACKSLASH_COUNT != 0
        if (state->backslashes > RULE_MAX_BACKSLASH_COUNT)
            return false;
#endif
    }
    state->nextIsBackslash = (flags & MEET_IN_THE_MIDDLE_BACKSLASH) != 0;
    return true;
}

// One step of the hash undone: the state the character `ord`, whose crypt
// table key is `key`, was hashed onto. Must match hashStepBack
// (engine/meet_in_the_middle.h) - 0x3E0F83E1 is 33's inverse modulo 2^32.
inline void mpqStepBack(uint* seed1, uint* seed2, uint key, uint ord) {
    const uint before2 = (*seed2 - ord - *seed1 - 3) * 0x3E0F83E1u;
    *seed1 = (*seed1 ^ key) - before2;
    *seed2 = before2;
}

// Whether the table holds the state (seed1, seed2): its filter first - all
// but about one lookup in eight stops there, without touching the keys - and
// then a binary search of the keys themselves.
inline bool meetInTheMiddleTableHolds(__global const ulong* keys, ulong keyCount, __global const ulong* filter, int filterShift, uint seed1,
                               uint seed2) {
    const ulong key = ((ulong) (seed2 & MEET_IN_THE_MIDDLE_SEED_MASK) << 32) | (seed1 & MEET_IN_THE_MIDDLE_SEED_MASK);
    const ulong bit = (key * 0x9E3779B97F4A7C15UL) >> filterShift;
    if (!((filter[bit >> 6] >> (bit & 63)) & 1))
        return false;
    ulong lo = 0, hi = keyCount;
    while (lo < hi) {
        const ulong mid = lo + (hi - lo) / 2;
        if (keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < keyCount && keys[lo] == key;
}

__kernel void searchMeetInTheMiddle(__global const ulong* keys, const ulong keyCount, __global const ulong* filter, const int filterShift,
                             __global const uint4* chars, const uint meetingPointSeed1, const uint meetingPointSeed2, const int tailLength,
                             const ulong firstRow, const ulong lastRow, const uint firstRowStartK, const uint lastRowEndK,
                             __global uint* counts, __global ulong* hits, const uint capacity) {
    // The alphabet in local memory: every work-item of the group reads the
    // same entry in the loop below, which local memory broadcasts.
    __local uint4 sChars[ALPHABET_SIZE];
    for (uint i = get_local_id(0); i < ALPHABET_SIZE; i += get_local_size(0))
        sChars[i] = chars[i];
    // The launch's tails looked up, counted per group and added to the global
    // count once - an atomic per tail would cost more than the lookups do.
    __local uint sProbed;
    if (get_local_id(0) == 0)
        sProbed = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const ulong row = firstRow + (ulong) get_global_id(0);
    uint probed = 0;
    if (row <= lastRow) {
        uint seed1 = meetingPointSeed1, seed2 = meetingPointSeed2;
        MeetInTheMiddlePrune prune = {ENTRY_SYMBOL_RUN, ENTRY_BACKSLASHES, ENTRY_NEXT_IS_BACKSLASH};
        bool pruned = false;
        // The characters rewound before the row's own: the row's digits, most
        // significant first - the directory's last character first (see
        // MeetInTheMiddleSpace for the numbering).
        if (tailLength > 1) {
            uint digit[MAX_CANDIDATE_LEN];
            ulong rest = row;
            for (int i = tailLength - 2; i >= 0; --i) {
                digit[i] = (uint) (rest % ALPHABET_SIZE);
                rest /= ALPHABET_SIZE;
            }
            for (int i = 0; i < tailLength - 1; ++i) {
                const uint4 c = sChars[digit[i]];
                if (RULE_ANY && !meetInTheMiddlePruneStep(&prune, c.z)) {
                    // Every tail of the row breaks the rules.
                    pruned = true;
                    break;
                }
                mpqStepBack(&seed1, &seed2, c.x, c.y);
            }
        }
        if (!pruned && tailLength == 0) {
            // The one tail of no characters: the meeting point itself, which is
            // where a directory of exactly the table's length ends.
            probed = 1;
            if (meetInTheMiddleTableHolds(keys, keyCount, filter, filterShift, seed1, seed2)) {
                const uint slot = atomic_inc(&counts[1]);
                if (slot < capacity)
                    hits[slot] = 0;
            }
        } else if (!pruned) {
            const uint fromK = row == firstRow ? firstRowStartK : 0;
            const uint toK = row == lastRow ? lastRowEndK : ALPHABET_SIZE;
            for (uint k = fromK; k < toK; ++k) {
                const uint4 c = sChars[k];
                MeetInTheMiddlePrune last = prune;
                if (RULE_ANY && !meetInTheMiddlePruneStep(&last, c.z))
                    continue;
                uint s1 = seed1, s2 = seed2;
                mpqStepBack(&s1, &s2, c.x, c.y);
                ++probed;
                if (meetInTheMiddleTableHolds(keys, keyCount, filter, filterShift, s1, s2)) {
                    const uint slot = atomic_inc(&counts[1]);
                    if (slot < capacity)
                        hits[slot] = row * ALPHABET_SIZE + k;
                }
            }
        }
    }
    atomic_add(&sProbed, probed);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0 && sProbed != 0)
        atomic_add(&counts[0], sProbed);
}
