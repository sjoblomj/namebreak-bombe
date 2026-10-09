// Correctness test for the parts of a dictionary search that don't hash
// candidates on a backend (engine/dictionary.h, engine/dictionary_search.h):
// how candidates are numbered, the bounds' verdicts on leading parts, the
// hash helpers, counting the candidates within the bounds, the fingerprint
// and the progress file - against brute force, and against hash values
// computed independently (Python's mpq_unknowns.py). Pure CPU.
//
// Writes its files under ./dictionary_test/ - ctest runs this from its own
// directory under build/testrun/.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "engine/dictionary.h"
#include "engine/dictionary_search.h"
#include "engine/mpq_hash.h"

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

static const std::string kDir = "dictionary_test";

static DictionaryPattern pattern(std::vector<std::string> words, std::vector<std::string> separators, int minWords, int maxWords) {
    DictionaryPattern p;
    p.words = std::move(words);
    p.separators = std::move(separators);
    p.minWords = minWords;
    p.maxWords = maxWords;
    return p;
}

// Every candidate's text in number order, made by plain nested loops - the
// first word turning slowest, then the separator after it, and so on - not
// by DictionarySpace.
static std::vector<std::string> allCandidates(const DictionaryPattern& p) {
    std::vector<std::string> out;
    for (int k = p.minWords; k <= p.maxWords; ++k) {
        std::function<void(int, const std::string&)> gen = [&](int level, const std::string& text) {
            for (const std::string& word : p.words) {
                if (level == k - 1) {
                    out.push_back(text + word);
                    continue;
                }
                for (const std::string& separator : p.separators)
                    gen(level + 1, text + word + separator);
            }
        };
        gen(0, "");
    }
    return out;
}

static std::string randomString(std::mt19937& rng, const std::string& alphabet, int minLen, int maxLen) {
    std::string s;
    const int len = std::uniform_int_distribution<int>(minLen, maxLen)(rng);
    for (int i = 0; i < len; ++i)
        s += alphabet[std::uniform_int_distribution<size_t>(0, alphabet.size() - 1)(rng)];
    return s;
}

static std::vector<std::string> randomWords(std::mt19937& rng, const std::string& alphabet, int count, int maxLen) {
    std::vector<std::string> words;
    for (int i = 0; i < count; ++i)
        words.push_back(randomString(rng, alphabet, 1, maxLen));
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    return words;
}

static const char* verdictName(FilenameBounds::Verdict v) {
    switch (v) {
    case FilenameBounds::Verdict::Outside:
        return "outside";
    case FilenameBounds::Verdict::Inside:
        return "inside";
    default:
        return "mixed";
    }
}

static FilenameBounds bounds(const char* lower, const char* upper) {
    FilenameBounds b;
    b.hasLower = lower != nullptr;
    b.lower = lower ? lower : "";
    b.hasUpper = upper != nullptr;
    b.upper = upper ? upper : "";
    return b;
}

static void testSpace() {
    printf("--- DictionarySpace::create ---\n");
    DictionarySpace space;
    std::string error;
    check(DictionarySpace::create(pattern({"A", "B", "C"}, {"", "_"}, 1, 3), space, error), "a valid pattern");
    check(space.size() == 3 + 3 * 2 * 3 + 3 * 2 * 3 * 2 * 3, "size: W + W^2 S + W^3 S^2");
    check(space.blockStart(1) == 0 && space.blockSize(1) == 3, "one word: numbers 0-2");
    check(space.blockStart(2) == 3 && space.blockSize(2) == 18, "two words: from 3, 18 of them");
    check(space.blockStart(3) == 21 && space.blockSize(3) == 108, "three words: from 21, 108 of them");
    check(DictionarySpace::create(pattern({"A", "B"}, {""}, 2, 2), space, error) && space.size() == 4 && space.blockStart(2) == 0,
          "min_words 2: the two-word block starts at 0");

    auto fails = [&](const DictionaryPattern& p, const std::string& expectedInError, const std::string& what) {
        std::string err;
        DictionarySpace s;
        const bool failed = !DictionarySpace::create(p, s, err);
        check(failed && err.find(expectedInError) != std::string::npos, what + " (error: '" + err + "')");
    };
    fails(pattern({}, {""}, 1, 1), "no words", "no words: an error");
    fails(pattern({"B", "A"}, {""}, 1, 1), "sorted", "unsorted words: an error");
    fails(pattern({"A", "A"}, {""}, 1, 1), "sorted", "duplicate words: an error");
    fails(pattern({"A"}, {}, 1, 2), "no separators", "no separators: an error");
    fails(pattern({"A"}, {"_", "_"}, 1, 2), "more than once", "a separator twice: an error");
    fails(pattern({"A"}, {""}, 0, 1), "1 <= min_words", "min_words 0: an error");
    fails(pattern({"A"}, {""}, 2, 1), "1 <= min_words", "max_words < min_words: an error");
    fails(pattern({"A"}, {""}, 1, kMaxDictionaryWords + 1), "1 <= min_words", "max_words over the limit: an error");
    check(DictionarySpace::create(pattern({"A"}, {""}, 1, kMaxDictionaryWords), space, error) && space.size() == kMaxDictionaryWords,
          "max_words at the limit: fine");
    {
        // 65536^4 * 4^3 overflows 64 bits; 65536^3 * 4^2 = 2^52 doesn't.
        std::vector<std::string> many;
        for (int i = 0; i < 65536; ++i) {
            char buf[8];
            snprintf(buf, sizeof buf, "%05d", i);
            many.push_back(buf);
        }
        fails(pattern(many, {"", "_", "-", " "}, 1, 4), "64 bits", "more candidates than 64 bits: an error");
        check(DictionarySpace::create(pattern(many, {"", "_", "-", " "}, 1, 3), space, error) &&
                  space.size() == 65536ull + (65536ull * 65536 * 4) + (65536ull * 65536 * 65536 * 16),
              "65536 words, 4 separators, up to three words: fits");
        check(space.text(space.decode(space.size() - 1)) == "65535 65535 65535", "... and its last candidate decodes");
    }

    printf("--- numbering ---\n");
    std::mt19937 rng(1234);
    bool allMatch = true, roundTrips = true;
    for (int round = 0; round < 40; ++round) {
        const DictionaryPattern p = pattern(randomWords(rng, "AB\\_.", 1 + round % 6, 3),
                                            round % 3 == 0 ? std::vector<std::string>{""} : std::vector<std::string>{"", "_", "\\"},
                                            1 + round % 2, 1 + round % 2 + round % 3);
        DictionarySpace s;
        if (!DictionarySpace::create(p, s, error)) {
            check(false, "random pattern: " + error);
            continue;
        }
        const std::vector<std::string> expected = allCandidates(p);
        allMatch = allMatch && expected.size() == s.size();
        for (uint64_t n = 0; n < s.size() && allMatch; ++n) {
            const DictionaryChoice choice = s.decode(n);
            allMatch = allMatch && s.text(choice) == expected[n] && s.textOf(n) == expected[n] &&
                       choice.separators.size() + 1 == choice.words.size();
            roundTrips = roundTrips && s.encode(choice) == n;
        }
    }
    check(allMatch, "every number decodes to the candidate nested loops give, in the same order (40 random patterns)");
    check(roundTrips, "encode(decode(n)) == n for every one");
    {
        DictionarySpace s;
        DictionarySpace::create(pattern({"A", "B", "C"}, {"", "_"}, 1, 2), s, error);
        check(s.textOf(0) == "A" && s.textOf(2) == "C" && s.textOf(3) == "AA" && s.textOf(4) == "AB" && s.textOf(6) == "A_A" &&
                  s.textOf(9) == "BA" && s.textOf(20) == "C_C",
              "worked example: A, B, C; then AA AB AC A_A A_B A_C BA ... C_C - the last word fastest, then the separator");
        DictionaryChoice choice;
        choice.words = {1, 2};
        choice.separators = {1};
        check(s.encode(choice) == 3 + (1 * 2 + 1) * 3 + 2 && s.text(choice) == "B_C", "encode: ((w1 * S + s1) * W + w2) after the one-word block");
    }
}

static void testBounds() {
    printf("--- FilenameBounds ---\n");
    const FilenameBounds b = bounds("REZ\\CRDT_LST.TXT", "REZ\\GLUCMPGN.BIN");
    check(b.contains("REZ\\CRDT_LST.TXT") && b.contains("REZ\\GLUCMPGN.BIN"), "contains: both bounds inclusive");
    check(b.contains("REZ\\DEATH.TXT") && !b.contains("REZ\\CRDT_LSS.TXT") && !b.contains("REZ\\GLUCMPGN.BINX"), "contains: in between only");
    check(bounds(nullptr, nullptr).contains("") && bounds(nullptr, nullptr).contains("~~~"), "no bounds: everything");
    check(bounds("", nullptr).contains("") && bounds(nullptr, "").contains("") && !bounds(nullptr, "").contains("A"),
          "an empty bound is a bound: everything is >= \"\", only \"\" is <= it");
    check(!bounds("B", nullptr).contains("A") && bounds("B", nullptr).contains("B"), "only a lower bound");
    check(bounds(nullptr, "B").contains("A") && !bounds(nullptr, "B").contains("BA"), "only an upper bound");

    auto verdict = [&](const FilenameBounds& fb, const std::string& start) { return std::string(verdictName(fb.classifyStart(start))); };
    check(verdict(b, "REZ\\DEATH") == "inside", "REZ\\DEATH...: wholly inside");
    check(verdict(b, "REZ\\ZERG") == "outside", "REZ\\ZERG...: wholly outside (after the upper bound)");
    check(verdict(b, "REZ\\ABC") == "outside", "REZ\\ABC...: wholly outside (before the lower bound)");
    check(verdict(b, "REZ\\CRDT") == "mixed", "REZ\\CRDT...: mixed - REZ\\CRDT_MAP is inside, REZ\\CRDTAARDVARK isn't");
    check(verdict(b, "REZ\\CRD") == "mixed", "REZ\\CRD...: mixed - REZ\\CRDZERG is inside, though CRD sorts before CRDT");
    check(verdict(b, "REZ\\CRE") == "inside", "REZ\\CRE...: wholly inside");
    check(verdict(b, "REZ\\GLU") == "mixed", "REZ\\GLU...: mixed (a prefix of the upper bound)");
    check(verdict(b, "REZ\\GLUCMPGN.BIN") == "mixed", "the upper bound itself: mixed - it's inside, anything longer isn't");
    check(verdict(b, "REZ\\GLUCMPGN.BINX") == "outside", "longer than the upper bound, which starts it: outside");
    check(verdict(b, "REZ\\CRDT_LST.TXT") == "inside", "the lower bound itself: inside (it and everything longer is >= it)");
    check(verdict(b, "") == "mixed" && verdict(b, "REZ\\") == "mixed", "a prefix of both bounds: mixed");
    check(verdict(bounds(nullptr, nullptr), "") == "inside" && verdict(bounds(nullptr, nullptr), "ANYTHING") == "inside", "no bounds: inside");
    check(verdict(bounds("", ""), "") == "mixed" && verdict(bounds("", ""), "A") == "outside", "both bounds empty: only \"\" is inside");

    // Against brute force: whatever the verdict says about the filenames
    // that start with a leading part must hold for every one of them.
    std::mt19937 rng(99);
    const std::string alphabet = "AB_\\.";
    bool sound = true;
    int inside = 0, outside = 0, mixed = 0, mixedProven = 0;
    for (int round = 0; round < 3000 && sound; ++round) {
        FilenameBounds fb;
        fb.hasLower = rng() % 4 != 0;
        fb.lower = randomString(rng, alphabet, 0, 4);
        fb.hasUpper = rng() % 4 != 0;
        fb.upper = randomString(rng, alphabet, 0, 4);
        if (fb.hasLower && fb.hasUpper && fb.upper < fb.lower)
            std::swap(fb.lower, fb.upper);
        const std::string start = randomString(rng, alphabet, 0, 4);
        // Every filename `start` + up to 3 more characters.
        int in = 0, total = 0;
        std::function<void(const std::string&, int)> extend = [&](const std::string& s, int more) {
            ++total;
            in += fb.contains(s) ? 1 : 0;
            if (more > 0) {
                for (char c : alphabet)
                    extend(s + c, more - 1);
            }
        };
        extend(start, 3);
        const FilenameBounds::Verdict v = fb.classifyStart(start);
        if (v == FilenameBounds::Verdict::Inside) {
            sound = in == total;
            ++inside;
        } else if (v == FilenameBounds::Verdict::Outside) {
            sound = in == 0;
            ++outside;
        } else {
            ++mixed;
            mixedProven += in > 0 && in < total ? 1 : 0;
        }
        if (!sound)
            fprintf(stderr, "  start '%s', lower %s'%s', upper %s'%s': %s, but %d of %d inside\n", start.c_str(), fb.hasLower ? "" : "(none) ",
                    fb.lower.c_str(), fb.hasUpper ? "" : "(none) ", fb.upper.c_str(), verdictName(v), in, total);
    }
    check(sound, "every inside/outside verdict holds for every filename that starts with it (3000 random cases)");
    check(inside > 100 && outside > 100 && mixed > 100,
          "... each verdict given often (inside " + std::to_string(inside) + ", outside " + std::to_string(outside) + ", mixed " +
              std::to_string(mixed) + ")");
    check(mixedProven * 10 > mixed * 7, "... and most mixed ones really are mixed (" + std::to_string(mixedProven) + " of " +
                                            std::to_string(mixed) + ") - it isn't just saying mixed to be safe");
}

static void testHashes() {
    printf("--- hash helpers ---\n");
    uint32_t table[0x500];
    prepareCryptTable(table);
    // Computed with scripts/mpq_unknowns.py's hash_string.
    check(continueHash(kInitialHashState, "REZ\\CRDT_LST.TXT", kHashAOffset, table).first == 0x339EFE07, "hashA of REZ\\CRDT_LST.TXT");
    check(continueHash(kInitialHashState, "REZ\\CRDT_LST.TXT", kHashBOffset, table).first == 0xE6B2B01C, "hashB of REZ\\CRDT_LST.TXT");
    check(continueHash(kInitialHashState, "CRDT_LST.TXT", kFileKeyOffset, table).first == 0x4565C467,
          "hash type 3 of CRDT_LST.TXT - the Portuguese StarDat's unknown credits file's raw key");
    check(continueHash(kInitialHashState, "", kHashAOffset, table) == kInitialHashState, "nothing hashed: the initial state");
    check(continueHash(continueHash(kInitialHashState, "REZ\\", kHashAOffset, table), "CRDT_LST.TXT", kHashAOffset, table) ==
              continueHash(kInitialHashState, "REZ\\CRDT_LST.TXT", kHashAOffset, table),
          "continued over two pieces: as over both at once");
    check(continueHash(kInitialHashState, "REZ\\CRDT_LST.TXT", kHashAOffset, table).first ==
              mpqHashWithPrefixCache_CPU("REZ\\CRDT_LST.TXT", table).first,
          "the same as mpqHashWithPrefixCache_CPU");

    check(continueBasenameHash(kInitialHashState, "REZ\\CRDT_LST.TXT", table).first == 0x4565C467, "basename key of REZ\\CRDT_LST.TXT");
    check(continueBasenameHash(kInitialHashState, "A\\B\\CRDT_LST.TXT", table).first == 0x4565C467, "... of A\\B\\CRDT_LST.TXT too");
    check(continueBasenameHash(kInitialHashState, "CRDT_LST.TXT", table).first == 0x4565C467, "... and of CRDT_LST.TXT, with no directory");
    check(continueBasenameHash(kInitialHashState, "MUSIC\\ZERG3.PK", table).first == 0x37237E62, "basename key of MUSIC\\ZERG3.PK");
    check(continueBasenameHash(kInitialHashState, "REZ\\", table) == kInitialHashState, "ending in '\\': the initial state");
    check(continueBasenameHash(continueBasenameHash(kInitialHashState, "RE", table), "Z\\CRDT_L", table) ==
              continueBasenameHash(kInitialHashState, "REZ\\CRDT_L", table),
          "continued over pieces split anywhere, the '\\' in the second: as at once");

    check(basenameOf("REZ\\CRDT_LST.TXT") == "CRDT_LST.TXT" && basenameOf("A\\B\\C") == "C", "basenameOf: after the last '\\'");
    check(basenameOf("CRDT_LST.TXT") == "CRDT_LST.TXT" && basenameOf("REZ\\") == "" && basenameOf("") == "", "basenameOf: no '\\', or nothing after it");
}

static void testCounting() {
    printf("--- countDictionaryCandidates ---\n");
    std::mt19937 rng(7);
    bool allMatch = true;
    int withBounds = 0;
    for (int round = 0; round < 400 && allMatch; ++round) {
        const DictionaryPattern p = pattern(randomWords(rng, "AB_\\", 1 + round % 7, 3),
                                            round % 2 ? std::vector<std::string>{"", "_"} : std::vector<std::string>{"\\"},
                                            1 + round % 2, 1 + round % 2 + round % 3);
        DictionarySpace space;
        std::string error;
        if (!DictionarySpace::create(p, space, error)) {
            check(false, "random pattern: " + error);
            return;
        }
        const std::vector<std::string> all = allCandidates(p);
        const std::string prefix = round % 3 ? "P\\" : "", suffix = round % 4 ? ".X" : "";
        // Bounds from the candidates themselves, cut short or not, or none.
        auto randomBound = [&]() {
            std::string f = prefix + all[rng() % all.size()] + suffix;
            if (rng() % 2)
                f = f.substr(0, rng() % (f.size() + 1));
            return f;
        };
        FilenameBounds fb;
        fb.hasLower = rng() % 3 != 0;
        fb.lower = fb.hasLower ? randomBound() : "";
        fb.hasUpper = rng() % 3 != 0;
        fb.upper = fb.hasUpper ? randomBound() : "";
        if (fb.hasLower && fb.hasUpper && fb.upper < fb.lower)
            std::swap(fb.lower, fb.upper);
        withBounds += fb.hasLower || fb.hasUpper;
        const uint64_t start = rng() % 3 ? rng() % (all.size() + 2) : 0;
        uint64_t expected = 0;
        for (uint64_t n = start; n < all.size(); ++n)
            expected += fb.contains(prefix + all[n] + suffix) ? 1 : 0;
        const uint64_t counted = countDictionaryCandidates(space, prefix, suffix, fb, start);
        if (counted != expected) {
            allMatch = false;
            fprintf(stderr, "  round %d: counted %llu, brute force %llu\n", round, (unsigned long long) counted, (unsigned long long) expected);
        }
    }
    check(allMatch, "the count within the bounds from a start number: as brute force (400 random patterns, bounds and starts)");
    check(withBounds > 300, "... most of them with a bound");
}

static void testFingerprint() {
    printf("--- dictionaryFingerprint ---\n");
    DictionaryRequest base;
    base.pattern = pattern({"A", "B"}, {"", "_"}, 1, 2);
    base.prefix = "REZ\\";
    base.suffix = ".TXT";
    base.bounds = bounds("REZ\\A", "REZ\\Z");
    base.targetHashA = 1;
    base.targetHashB = 2;
    base.checkBasename = true;
    base.basenameKey = 3;
    const std::string fp = dictionaryFingerprint(base);
    check(fp.size() == 16 && fp.find_first_not_of("0123456789abcdef") == std::string::npos, "16 lowercase hex digits");
    check(dictionaryFingerprint(base) == fp, "the same request: the same fingerprint");
    // Worked out by hand (well, in Python) from the format the fingerprint
    // has had since dictionary mode began: a search with a key, its
    // basenames recorded, keeps resuming from the progress files it wrote.
    check(fp == "e0f01a98635c367c", "a recorded key's fingerprint as it always was: " + fp);

    std::vector<std::pair<std::string, std::function<void(DictionaryRequest&)>>> changes = {
        {"a word", [](DictionaryRequest& r) { r.pattern.words = {"A", "C"}; }},
        {"a separator", [](DictionaryRequest& r) { r.pattern.separators = {"", "-"}; }},
        {"the separators' order", [](DictionaryRequest& r) { r.pattern.separators = {"_", ""}; }},
        {"min_words", [](DictionaryRequest& r) { r.pattern.minWords = 2; }},
        {"max_words", [](DictionaryRequest& r) { r.pattern.maxWords = 3; }},
        {"the prefix", [](DictionaryRequest& r) { r.prefix = "REZ"; }},
        {"the suffix", [](DictionaryRequest& r) { r.suffix = ".TBL"; }},
        {"prefix and suffix trading a character", [](DictionaryRequest& r) { r.prefix = "REZ\\."; r.suffix = "TXT"; }},
        {"the lower bound", [](DictionaryRequest& r) { r.bounds.lower = "REZ\\B"; }},
        {"no lower bound", [](DictionaryRequest& r) { r.bounds.hasLower = false; }},
        {"the upper bound", [](DictionaryRequest& r) { r.bounds.upper = "REZ\\Y"; }},
        {"no upper bound", [](DictionaryRequest& r) { r.bounds.hasUpper = false; }},
        {"hashA", [](DictionaryRequest& r) { r.targetHashA = 9; }},
        {"hashB", [](DictionaryRequest& r) { r.targetHashB = 9; }},
        {"the encryption key", [](DictionaryRequest& r) { r.basenameKey = 9; }},
        {"to no encryption key", [](DictionaryRequest& r) { r.checkBasename = false; }},
        {"to the basenames not recorded", [](DictionaryRequest& r) { r.recordBasenames = false; }},
        {"to every hashA hit recorded", [](DictionaryRequest& r) { r.recordHashAMatches = true; }},
    };
    for (auto& change : changes) {
        DictionaryRequest changed = base;
        change.second(changed);
        check(dictionaryFingerprint(changed) != fp, "changing " + change.first + " changes it");
    }
    DictionaryRequest same = base;
    same.startNumber = 12345;
    same.wordSource = "other";
    same.outputFilePath = "x";
    same.basenamesFilePath = "y";
    same.progressFilePath = "z";
    same.progressInterval = std::chrono::milliseconds(1);
    same.statusInterval = std::chrono::milliseconds(1);
    check(dictionaryFingerprint(same) == fp, "the start, the files, the intervals and where the words came from don't");
}

static void writeFile(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

static void testProgressFile() {
    printf("--- progress file ---\n");
    std::filesystem::create_directories(kDir);
    const std::string path = kDir + "/wordnumber.txt";
    std::string error;
    DictionaryProgress progress;
    bool exists = true;
    std::filesystem::remove(path);
    check(readDictionaryProgress(path, progress, exists, error) && !exists, "no file: true, and exists false");

    check(writeDictionaryProgress(path, {"0123456789abcdef", 18446744073709551615ull}, 7, "CRDT_LST", error), "written");
    DictionaryProgress back;
    check(readDictionaryProgress(path, back, exists, error) && exists && back.fingerprint == "0123456789abcdef" &&
              back.next == 18446744073709551615ull,
          "read back: fingerprint and next, up to the largest 64-bit number");
    check(!std::filesystem::exists(path + ".tmp"), "no temporary file left behind");
    {
        std::ifstream in(path);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        check(text.find("total = 7\n") != std::string::npos && text.find("candidate = CRDT_LST\n") != std::string::npos &&
                  text[0] == '#',
              "with the total and the candidate for a human, and a comment first");
    }

    writeFile(path, "# comment\r\n\r\n  fingerprint =  fedcba9876543210 \r\nnext=42\r\nsomething = else\r\n");
    check(readDictionaryProgress(path, back, exists, error) && back.fingerprint == "fedcba9876543210" && back.next == 42,
          "CRLF, spaces, comments, blank lines and other keys: fine");

    auto rejects = [&](const std::string& text, const std::string& what) {
        writeFile(path, text);
        DictionaryProgress p;
        bool e = false;
        std::string err;
        check(!readDictionaryProgress(path, p, e, err) && e && !err.empty(), what + " (error: '" + err + "')");
    };
    rejects("next = 42\n", "no fingerprint: an error");
    rejects("fingerprint = 0123456789abcdef\n", "no next: an error");
    rejects("", "an empty file: an error");
    rejects("fingerprint = 0123456789ABCDEF\nnext = 1\n", "uppercase fingerprint: an error");
    rejects("fingerprint = 0123456789abcde\nnext = 1\n", "a short fingerprint: an error");
    rejects("fingerprint = 0123456789abcdef\nnext = -1\n", "a negative next: an error");
    rejects("fingerprint = 0123456789abcdef\nnext = 12x\n", "next with trailing junk: an error");
    rejects("fingerprint = 0123456789abcdef\nnext =\n", "an empty next: an error");
    rejects("fingerprint = 0123456789abcdef\nnext = 18446744073709551616\n", "next beyond 64 bits: an error");
    rejects("fingerprint = 0123456789abcdef\nnext = 1\njunk\n", "a line that isn't key = value: an error");
}

static void testFormatDuration() {
    printf("--- formatDuration ---\n");
    check(formatDuration(0) == "0s" && formatDuration(6.4) == "6s" && formatDuration(59.6) == "1m00s", "seconds, rounded");
    check(formatDuration(245) == "4m05s", "minutes");
    check(formatDuration(3723) == "1h02m03s", "hours");
    check(formatDuration(2 * 86400 + 3 * 3600 + 4 * 60 + 5) == "2d03h04m", "days");
    check(formatDuration(-1) == "?" && formatDuration(1e13) == "?", "negative or absurd: ?");
}

int main() {
    testFormatDuration();
    testSpace();
    testBounds();
    testHashes();
    testCounting();
    testFingerprint();
    testProgressFile();

    std::filesystem::remove_all(kDir);
    if (g_failures) {
        fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL CHECKS PASSED\n");
    return 0;
}
