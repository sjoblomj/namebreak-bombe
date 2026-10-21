// The Metal backend's meet-in-the-middle kernel - the OpenCL one
// (backends/opencl/meet_in_the_middle.cl, itself the CUDA kernel ported) in
// Metal's shading language, and the twin of MeetInTheMiddleWalk on the CPU
// (see engine/meet_in_the_middle.h for what a meet-in-the-middle search is
// and what a tail means). Embedded into the program at build time and
// compiled by Metal at runtime, once per combination of these (see
// metal_backend.mm), as the rules and the alphabet don't change within a
// search:
//   ALPHABET_SIZE                how many characters a directory is made of
//   MAX_CANDIDATE_LEN            the longest tail a batch can have
//   MEET_IN_THE_MIDDLE_SEED_MASK the bits of each seed that must match the
//                                meeting point (every one of them, outside
//                                the tests)
//   RULE_ANY, RULE_SYMBOL_RUNS, RULE_ADJACENT_BACKSLASHES,
//   RULE_MAX_BACKSLASH_COUNT     the pruning rules
//   ENTRY_SYMBOL_RUN, ENTRY_BACKSLASHES, ENTRY_NEXT_IS_BACKSLASH
//                                the rules' state to the directory's right
//
// One thread takes one *row*: the ALPHABET_SIZE tails that share every
// character but the one rewound last - the first character of the tail,
// which is the directory's own. Everything before it, and the pruning rules
// over it, is then done once for the whole row.

#include <metal_stdlib>
using namespace metal;

// What the rules make of one character - as the host's has it, in each
// character's uint4 (key, ord, flags, unused).
#define MEET_IN_THE_MIDDLE_SYMBOL 1u
#define MEET_IN_THE_MIDDLE_BACKSLASH 2u

// Must match MeetInTheMiddleArgs in metal_backend.mm.
struct MeetInTheMiddleArgs {
    ulong keyCount;
    ulong firstRow;
    ulong lastRow;
    int filterShift;
    uint meetingPointSeed1;
    uint meetingPointSeed2;
    int tailLength;
    uint firstRowStartK;
    uint lastRowEndK;
    uint capacity;
    uint unused;
};

// The rules' state over the characters rewound so far (TailPruneState).
struct MeetInTheMiddlePrune {
    int symbolRun;
    int backslashes;
    int nextIsBackslash;
};

// Steps `state` over a character, to the left of the ones so far; false if it
// breaks one of the rules. Must agree with tailPruneStep
// (engine/meet_in_the_middle.cpp) exactly, or this would search other tails
// than the CPU.
static bool meetInTheMiddlePruneStep(thread MeetInTheMiddlePrune& state, uint flags) {
    // Each rule compiled in or out, rather than tested at runtime, as in the
    // OpenCL kernel.
    if (flags & MEET_IN_THE_MIDDLE_SYMBOL) {
        ++state.symbolRun;
#if RULE_SYMBOL_RUNS
        if (state.symbolRun >= 3)
            return false;
#endif
        state.symbolRun = min(state.symbolRun, 2);
    } else {
        state.symbolRun = 0;
    }
    if (flags & MEET_IN_THE_MIDDLE_BACKSLASH) {
#if RULE_ADJACENT_BACKSLASHES
        if (state.nextIsBackslash)
            return false;
#endif
        ++state.backslashes;
#if RULE_MAX_BACKSLASH_COUNT != 0
        if (state.backslashes > RULE_MAX_BACKSLASH_COUNT)
            return false;
#endif
    }
    state.nextIsBackslash = (flags & MEET_IN_THE_MIDDLE_BACKSLASH) != 0;
    return true;
}

// One step of the hash undone: the state the character `ord`, whose crypt
// table key is `key`, was hashed onto. Must match hashStepBack
// (engine/meet_in_the_middle.h) - 0x3E0F83E1 is 33's inverse modulo 2^32.
static void mpqStepBack(thread uint& seed1, thread uint& seed2, uint key, uint ord) {
    const uint before2 = (seed2 - ord - seed1 - 3) * 0x3E0F83E1u;
    seed1 = (seed1 ^ key) - before2;
    seed2 = before2;
}

// Whether the table holds the state (seed1, seed2): its filter first - all
// but about one lookup in eight stops there, without touching the keys - and
// then a binary search of the keys themselves.
static bool meetInTheMiddleTableHolds(device const ulong* keys, ulong keyCount, device const ulong* filter, int filterShift, uint seed1,
                                      uint seed2) {
    const ulong key = ((ulong) (seed2 & (uint) (MEET_IN_THE_MIDDLE_SEED_MASK)) << 32) | (seed1 & (uint) (MEET_IN_THE_MIDDLE_SEED_MASK));
    const ulong bit = (key * 0x9E3779B97F4A7C15ul) >> filterShift;
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

// `chars` is the alphabet: each character's (hashA crypt table key, value,
// flags, unused). `counts` is how many tails the launch looked up and how
// many of them the table holds, every one of them, and `hits` the first
// args.capacity of those, as tail indices.
kernel void searchMeetInTheMiddle(constant MeetInTheMiddleArgs& args [[buffer(0)]],
                                  device const ulong* keys [[buffer(1)]],
                                  device const ulong* filter [[buffer(2)]],
                                  device const uint4* chars [[buffer(3)]],
                                  device atomic_uint* counts [[buffer(4)]],
                                  device ulong* hits [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint lsize [[threads_per_threadgroup]]) {
    // The alphabet in threadgroup memory: every thread of the group reads the
    // same entry in the loop below.
    threadgroup uint4 sChars[ALPHABET_SIZE];
    for (uint i = lid; i < ALPHABET_SIZE; i += lsize)
        sChars[i] = chars[i];
    // The launch's tails looked up, counted per threadgroup and added to the
    // global count once - an atomic per tail would cost more than the lookups
    // do.
    threadgroup atomic_uint sProbed;
    if (lid == 0)
        atomic_store_explicit(&sProbed, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong row = args.firstRow + (ulong) gid;
    uint probed = 0;
    if (row <= args.lastRow) {
        uint seed1 = args.meetingPointSeed1, seed2 = args.meetingPointSeed2;
        MeetInTheMiddlePrune prune = {ENTRY_SYMBOL_RUN, ENTRY_BACKSLASHES, ENTRY_NEXT_IS_BACKSLASH};
        bool pruned = false;
        // The characters rewound before the row's own: the row's digits, most
        // significant first - the directory's last character first (see
        // MeetInTheMiddleSpace for the numbering).
        if (args.tailLength > 1) {
            uint digit[MAX_CANDIDATE_LEN];
            ulong rest = row;
            for (int i = args.tailLength - 2; i >= 0; --i) {
                digit[i] = (uint) (rest % ALPHABET_SIZE);
                rest /= ALPHABET_SIZE;
            }
            for (int i = 0; i < args.tailLength - 1; ++i) {
                const uint4 c = sChars[digit[i]];
                if (RULE_ANY && !meetInTheMiddlePruneStep(prune, c.z)) {
                    // Every tail of the row breaks the rules.
                    pruned = true;
                    break;
                }
                mpqStepBack(seed1, seed2, c.x, c.y);
            }
        }
        if (!pruned && args.tailLength == 0) {
            // The one tail of no characters: the meeting point itself, which is
            // where a directory of exactly the table's length ends.
            probed = 1;
            if (meetInTheMiddleTableHolds(keys, args.keyCount, filter, args.filterShift, seed1, seed2)) {
                const uint slot = atomic_fetch_add_explicit(&counts[1], 1u, memory_order_relaxed);
                if (slot < args.capacity)
                    hits[slot] = 0;
            }
        } else if (!pruned) {
            const uint fromK = row == args.firstRow ? args.firstRowStartK : 0;
            const uint toK = row == args.lastRow ? args.lastRowEndK : ALPHABET_SIZE;
            for (uint k = fromK; k < toK; ++k) {
                const uint4 c = sChars[k];
                MeetInTheMiddlePrune last = prune;
                if (RULE_ANY && !meetInTheMiddlePruneStep(last, c.z))
                    continue;
                uint s1 = seed1, s2 = seed2;
                mpqStepBack(s1, s2, c.x, c.y);
                ++probed;
                if (meetInTheMiddleTableHolds(keys, args.keyCount, filter, args.filterShift, s1, s2)) {
                    const uint slot = atomic_fetch_add_explicit(&counts[1], 1u, memory_order_relaxed);
                    if (slot < args.capacity)
                        hits[slot] = row * ALPHABET_SIZE + k;
                }
            }
        }
    }
    if (probed != 0)
        atomic_fetch_add_explicit(&sProbed, probed, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        const uint total = atomic_load_explicit(&sProbed, memory_order_relaxed);
        if (total != 0)
            atomic_fetch_add_explicit(&counts[0], total, memory_order_relaxed);
    }
}
