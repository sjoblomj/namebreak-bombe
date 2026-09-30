// Correctness test for the lookup filter (backends/common/lowbits_filter.h) on
// its own, independent of any search: the filter must never leave out a
// candidate that matches, and every other test only ever sees the few matches
// its cases plant. Pure CPU. What it checks:
//
//  1. The property the filter rests on, for the real hash step: the low n
//     bits of the result never depend on higher bits of the state (for every
//     n, for arbitrary characters and suffixes, bytes >= 0x80 included).
//  2. Every bit of every entry of the table, for many alphabets, suffixes and
//     targets (and the real search's), against its definition - from states
//     with random high bits, so this also checks the index function and the
//     table's layout, not just the builder.
//  3. Soundness head-on: a random state and last character, and a target set
//     to exactly the hashA they produce - a true match by construction - must
//     always get past the filter.
//  4. That the check in (2) - which beginSearch also runs on a sample of
//     every real search's table - really catches a wrong table: a flipped
//     bit, a bit past the alphabet, a table built for another target, or with
//     seed1 and seed2 swapped.
//  5. That the filter does filter: about alphabetSize / 2^n bits per entry.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backends/common/lowbits_filter.h"
#include "engine/mpq_hash.h"

static uint32_t g_cryptTable[0x500];
static int g_failures = 0;

static void fail(const std::string& what) {
    fprintf(stderr, "FAILED: %s\n", what.c_str());
    ++g_failures;
}

static void step(uint32_t& seed1, uint32_t& seed2, unsigned char ch) {
    seed1 = g_cryptTable[0x100 + ch] ^ (seed1 + seed2);
    seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
}

static uint32_t finish(uint32_t seed1, uint32_t seed2, unsigned char last, const std::string& suffix) {
    step(seed1, seed2, last);
    for (unsigned char ch : suffix)
        step(seed1, seed2, ch);
    return seed1;
}

static std::string randomBytes(std::mt19937_64& rng, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i)
        s += (char) (1 + rng() % 255); // any byte but NUL
    return s;
}

// `n` distinct bytes (no NUL), in random order.
static std::string randomAlphabet(std::mt19937_64& rng, int n) {
    std::vector<char> pool;
    for (int b = 1; b < 256; ++b)
        pool.push_back((char) b);
    std::shuffle(pool.begin(), pool.end(), rng);
    return std::string(pool.begin(), pool.begin() + n);
}

static SearchConstants constantsFor(const std::string& alphabet, const std::string& suffix, uint32_t targetA) {
    SearchConstants c;
    c.alphabet = alphabet;
    c.suffix = suffix;
    c.cryptTable = g_cryptTable;
    c.targetHashA = targetA;
    c.targetHashB = 0;
    return c;
}

static void testProperty(std::mt19937_64& rng) {
    printf("=== 1: low bits depend only on low bits ===\n");
    long checks = 0;
    for (int n = 1; n <= 32; ++n) {
        const uint32_t mask = 0xFFFFFFFFu >> (32 - n);
        for (int trial = 0; trial < 20000; ++trial) {
            const std::string suffix = randomBytes(rng, rng() % 24);
            const unsigned char last = (unsigned char) (1 + rng() % 255);
            const uint32_t s1 = (uint32_t) rng(), s2 = (uint32_t) rng();
            const uint32_t t1 = (s1 & mask) | ((uint32_t) rng() & ~mask), t2 = (s2 & mask) | ((uint32_t) rng() & ~mask);
            ++checks;
            if ((finish(s1, s2, last, suffix) & mask) != (finish(t1, t2, last, suffix) & mask)) {
                fail("the low " + std::to_string(n) + " bits of the hash depend on a higher bit of the state");
                return;
            }
        }
    }
    printf("    %ld checks passed\n", checks);
}

static void testWholeTables(std::mt19937_64& rng) {
    printf("=== 2: every bit of every entry ===\n");
    struct Config { std::string alphabet, suffix; uint32_t target; };
    std::vector<Config> configs = {
        {" !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_", ".WAV", 0xF60F5D90}, // the real search
        {" !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_#", "", 0x00000000},
        {"A", ".TXT", 0xFFFFFFFF},
    };
    // 24 random ones at the default 7 bits (16,384 entries each), fewer for
    // the wider tables CMakeLists.txt also builds this for - the same number
    // of entries in all, and at least 2 tables.
    const int randomTables = std::max(2, std::min(24, (int) (24 * 16384ull / kLowBitsFilterEntries)));
    for (int i = 0; i < randomTables; ++i) {
        const int size = 1 + (int) (rng() % MAX_ALPHABET_SIZE);
        const size_t suffixLen = (i % 6 == 0) ? 20 + rng() % 44 : rng() % 12;
        configs.push_back({randomAlphabet(rng, size), randomBytes(rng, suffixLen), (uint32_t) rng()});
    }
    for (const Config& config : configs) {
        const SearchConstants c = constantsFor(config.alphabet, config.suffix, config.target);
        const std::vector<uint64_t> table = buildLowBitsFilterTable(c);
        std::string error;
        if (!checkLowBitsFilterTable(table, c, 0, 2, rng(), error))
            fail("alphabet size " + std::to_string(config.alphabet.size()) + ", suffix length " + std::to_string(config.suffix.size()) + ": " + error);
    }
    printf("    %zu tables, %u entries each, checked\n", configs.size(), kLowBitsFilterEntries);
}

static void testPlantedMatches(std::mt19937_64& rng) {
    printf("=== 3: a true match always gets past the filter ===\n");
    const int kPlanted = 300;
    for (int i = 0; i < kPlanted; ++i) {
        const std::string alphabet = randomAlphabet(rng, 1 + (int) (rng() % MAX_ALPHABET_SIZE));
        const std::string suffix = randomBytes(rng, rng() % 16);
        const uint32_t s1 = (uint32_t) rng(), s2 = (uint32_t) rng();
        const int k = (int) (rng() % alphabet.size());
        const uint32_t target = finish(s1, s2, (unsigned char) alphabet[k], suffix);
        const std::vector<uint64_t> table = buildLowBitsFilterTable(constantsFor(alphabet, suffix, target));
        if (!((table[lowBitsFilterIndex(s1, s2)] >> k) & 1)) {
            char msg[200];
            snprintf(msg, sizeof(msg), "the filter leaves out a true match: state (0x%08X, 0x%08X), last character %d of %zu, suffix length %zu",
                     s1, s2, k, alphabet.size(), suffix.size());
            fail(msg);
            return;
        }
    }
    printf("    %d planted matches, all let through\n", kPlanted);
}

static void testCheckCatchesWrongTables(std::mt19937_64& rng) {
    printf("=== 4: the table check catches a wrong table ===\n");
    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
    const SearchConstants c = constantsFor(alphabet, ".WAV", 0xF60F5D90);
    const std::vector<uint64_t> good = buildLowBitsFilterTable(c);
    std::string error;
    auto mustFail = [&](const std::vector<uint64_t>& table, const char* what) {
        // A "wrong" table the same as the right one isn't wrong: at the
        // narrowest widths CMakeLists.txt builds this for (1 bit), another
        // suffix or target can give the same table. At the widths searches
        // use, each of these must differ, or this test tests nothing.
        if (table == good) {
            if (kLowBitsFilterBits >= 4)
                fail(std::string("TEST BUG: ") + what + " gives the same table");
            else
                printf("    (%s gives the same table at %d bits - nothing to catch)\n", what, kLowBitsFilterBits);
            return;
        }
        if (checkLowBitsFilterTable(table, c, 0, 1, rng(), error))
            fail(std::string("the table check doesn't notice ") + what);
    };
    for (int i = 0; i < 20; ++i) {
        std::vector<uint64_t> table = good;
        table[rng() % table.size()] ^= uint64_t(1) << (rng() % alphabet.size());
        mustFail(table, "one flipped bit");
    }
    {
        std::vector<uint64_t> table = good;
        table[rng() % table.size()] |= uint64_t(1) << alphabet.size();
        mustFail(table, "a bit past the alphabet");
    }
    mustFail(buildLowBitsFilterTable(constantsFor(alphabet, ".WAV", 0xF60F5D91)), "a table built for another target");
    mustFail(buildLowBitsFilterTable(constantsFor(alphabet, ".WAX", 0xF60F5D90)), "a table built for another suffix");
    if (kLowBitsFilterBits > 0) {
        std::vector<uint64_t> swapped(good.size());
        for (uint32_t lo1 = 0; lo1 <= kLowBitsFilterStateMask; ++lo1)
            for (uint32_t lo2 = 0; lo2 <= kLowBitsFilterStateMask; ++lo2)
                swapped[lowBitsFilterIndex(lo2, lo1)] = good[lowBitsFilterIndex(lo1, lo2)];
        mustFail(swapped, "seed1 and seed2 swapped");
    }
    // Sampled, as beginSearch runs it: a table that's wrong everywhere.
    std::vector<uint64_t> empty(good.size(), 0);
    if (checkLowBitsFilterTable(empty, c, 1024, 2, rng(), error))
        fail("the sampled table check doesn't notice an all-zero table");
    printf("    %s\n", g_failures ? "FAILURES above" : "every wrong table caught");
}

static void testSelectivity() {
    printf("=== 5: the filter filters ===\n");
    const std::string alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
    const std::vector<uint64_t> table = buildLowBitsFilterTable(constantsFor(alphabet, ".WAV", 0xF60F5D90));
    uint64_t bits = 0;
    for (uint64_t entry : table)
        for (uint64_t e = entry; e; e &= e - 1)
            ++bits;
    const double perEntry = (double) bits / table.size();
    const double expected = (double) alphabet.size() / (1u << kLowBitsFilterBits);
    printf("    %.3f candidates per row get past it (expected about %.3f)\n", perEntry, expected);
    if (perEntry > expected * 1.1 + 0.05 || perEntry < expected * 0.9 - 0.05)
        fail("the filter lets through " + std::to_string(perEntry) + " candidates per row, not about " + std::to_string(expected));
}

int main() {
    prepareCryptTable(g_cryptTable);
    std::mt19937_64 rng(20260929);
    printf("kLowBitsFilterBits=%d, kLowBitsFilterHashMask=0x%08X\n", kLowBitsFilterBits, kLowBitsFilterHashMask);
    testProperty(rng);
    testWholeTables(rng);
    testPlantedMatches(rng);
    testCheckCatchesWrongTables(rng);
    testSelectivity();
    if (g_failures) {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
