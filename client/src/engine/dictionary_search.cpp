#include "engine/dictionary_search.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <vector>

#include "engine/match_writer.h"
#include "engine/mpq_hash.h"
#include "engine/wordlist.h"

// The most candidates one runDictionaryBatches() call gets, whatever the
// backend asks for - set small at compile time by the tests (see
// CMakeLists.txt), so that even their small searches are split into many
// calls, and a leading part's words across them.
#ifndef NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL
#define NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL UINT64_MAX
#endif
// Likewise the most batches one call gets.
#ifndef NAMEBREAK_DICTIONARY_BATCHES_PER_CALL
#define NAMEBREAK_DICTIONARY_BATCHES_PER_CALL SIZE_MAX
#endif

void SearchBackend::beginDictionarySearch(const DictionaryConstants&) {
    fprintf(stderr, "BUG: the %s backend can't search dictionaries - check supportsDictionary() first\n", name());
    std::abort();
}

DictionaryOutcome SearchBackend::runDictionaryBatches(const std::vector<DictionaryBatch>&) {
    fprintf(stderr, "BUG: the %s backend can't search dictionaries - check supportsDictionary() first\n", name());
    std::abort();
}

HashState continueHash(HashState state, const std::string& text, int offset, const uint32_t* cryptTable) {
    uint32_t seed1 = state.first, seed2 = state.second;
    for (unsigned char ch : text) {
        seed1 = cryptTable[offset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return {seed1, seed2};
}

HashState continueBasenameHash(HashState state, const std::string& text, const uint32_t* cryptTable) {
    uint32_t seed1 = state.first, seed2 = state.second;
    for (unsigned char ch : text) {
        if (ch == '\\') {
            seed1 = kInitialHashState.first;
            seed2 = kInitialHashState.second;
            continue;
        }
        seed1 = cryptTable[kFileKeyOffset + ch] ^ (seed1 + seed2);
        seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3;
    }
    return {seed1, seed2};
}

std::string basenameOf(const std::string& filename) {
    const size_t slash = filename.rfind('\\');
    return slash == std::string::npos ? filename : filename.substr(slash + 1);
}

namespace {

// A run of candidates the walk below hands on: `leading` followed by each of
// the words [firstWord, firstWord + wordCount), numbered from firstNumber.
struct Leaf {
    const std::string* leading;
    HashState hashA;
    HashState basename;
    uint64_t firstNumber;
    uint32_t firstWord;
    uint32_t wordCount;
};

// Walks a dictionary search's candidates from a start number up to (not
// including) an end number, in number order, as the runs of them within the
// bounds: depth first, one level per word and the separator after it, down
// to the last word. A leading part the bounds put wholly outside is skipped,
// and one they put wholly inside isn't compared again below it. Without
// onLeaf, it only counts the candidates (and doesn't descend into a part
// wholly inside at all); without a crypt table, it doesn't hash.
class Walker {
public:
    Walker(const DictionarySpace& space, const std::string& prefix, const std::string& suffix, const FilenameBounds& bounds,
           uint64_t start, uint64_t end, const uint32_t* cryptTable, std::function<bool(const Leaf&)> onLeaf)
        : space_(space), prefix_(prefix), suffix_(suffix), bounds_(bounds), start_(start), end_(end), cryptTable_(cryptTable),
          onLeaf_(std::move(onLeaf)) {}

    // False if onLeaf stopped the walk.
    bool run() {
        const FilenameBounds::Verdict verdict = bounds_.classifyStart(prefix_);
        if (verdict == FilenameBounds::Verdict::Outside)
            return true;
        HashState hashA = kInitialHashState, basename = kInitialHashState;
        if (cryptTable_) {
            hashA = continueHash(hashA, prefix_, kHashAOffset, cryptTable_);
            basename = continueBasenameHash(basename, prefix_, cryptTable_);
        }
        for (int k = space_.minWords(); k <= space_.maxWords() && space_.blockStart(k) < end_; ++k) {
            if (!visit(space_.blockStart(k), space_.blockSize(k), k - 1, prefix_, hashA, basename,
                       verdict == FilenameBounds::Verdict::Inside))
                return false;
        }
        return true;
    }

    uint64_t counted() const { return counted_; }

private:
    // The candidates numbered [nodeStart, nodeStart + nodeSize): `text`
    // (the prefix and the words and separators chosen so far) followed by
    // `pairsLeft` more words and separators, and the last word.
    bool visit(uint64_t nodeStart, uint64_t nodeSize, int pairsLeft, const std::string& text, HashState hashA, HashState basename,
               bool inside) {
        if (nodeStart + nodeSize <= start_ || nodeStart >= end_)
            return true;
        if (!onLeaf_ && inside) {
            counted_ += std::min(nodeStart + nodeSize, end_) - std::max(nodeStart, start_);
            return true;
        }
        if (pairsLeft == 0)
            return leaf(nodeStart, text, hashA, basename, inside);

        const std::vector<std::string>& words = space_.words();
        const std::vector<std::string>& separators = space_.separators();
        const uint64_t pairs = (uint64_t) words.size() * separators.size();
        const uint64_t childSize = nodeSize / pairs;
        // A child is a word and the separator after it - the separator
        // turning faster.
        for (uint64_t c = start_ > nodeStart ? (start_ - nodeStart) / childSize : 0; c < pairs && nodeStart + c * childSize < end_; ++c) {
            const std::string& word = words[c / separators.size()];
            const std::string& separator = separators[c % separators.size()];
            const std::string childText = text + word + separator;
            bool childInside = inside;
            if (!inside) {
                const FilenameBounds::Verdict verdict = bounds_.classifyStart(childText);
                if (verdict == FilenameBounds::Verdict::Outside)
                    continue;
                childInside = verdict == FilenameBounds::Verdict::Inside;
            }
            HashState childA = hashA, childBasename = basename;
            if (cryptTable_) {
                childA = continueHash(continueHash(hashA, word, kHashAOffset, cryptTable_), separator, kHashAOffset, cryptTable_);
                childBasename = continueBasenameHash(continueBasenameHash(basename, word, cryptTable_), separator, cryptTable_);
            }
            if (!visit(nodeStart + c * childSize, childSize, pairsLeft - 1, childText, childA, childBasename, childInside))
                return false;
        }
        return true;
    }

    // The candidates `leading` + each word + the suffix, numbered from
    // nodeStart.
    bool leaf(uint64_t nodeStart, const std::string& leading, HashState hashA, HashState basename, bool inside) {
        const std::vector<std::string>& words = space_.words();
        const uint32_t wordCount = (uint32_t) words.size();
        const uint32_t first = start_ > nodeStart ? (uint32_t) (start_ - nodeStart) : 0;
        const uint32_t last = end_ - nodeStart < wordCount ? (uint32_t) (end_ - nodeStart) : wordCount; // exclusive
        if (inside)
            return emit(nodeStart, leading, hashA, basename, first, last - first);
        // Cut through by a bound: each candidate is compared, and every run
        // of them within the bounds handed on.
        uint32_t runStart = first;
        for (uint32_t i = first; i <= last; ++i) {
            const bool within = i < last && bounds_.contains(leading + words[i] + suffix_);
            if (within)
                continue;
            if (i > runStart && !emit(nodeStart, leading, hashA, basename, runStart, i - runStart))
                return false;
            runStart = i + 1;
        }
        return true;
    }

    // Hands on a run of (at least one) candidates.
    bool emit(uint64_t nodeStart, const std::string& leading, HashState hashA, HashState basename, uint32_t firstWord,
              uint32_t wordCount) {
        if (!onLeaf_) {
            counted_ += wordCount;
            return true;
        }
        return onLeaf_(Leaf{&leading, hashA, basename, nodeStart + firstWord, firstWord, wordCount});
    }

    const DictionarySpace& space_;
    const std::string& prefix_;
    const std::string& suffix_;
    const FilenameBounds& bounds_;
    uint64_t start_;
    uint64_t end_;
    const uint32_t* cryptTable_;
    std::function<bool(const Leaf&)> onLeaf_;
    uint64_t counted_ = 0;
};

// Keeps the basenames file: every basename that matched, once each - those
// already in the file from an earlier search included. Without a file (an
// empty path), only remembers which it has had.
class BasenameWriter {
public:
    ~BasenameWriter() {
        if (file_)
            fclose(file_);
    }

    bool open(const std::string& path, std::string& error) {
        path_ = path;
        if (path.empty())
            return true;
        std::filesystem::path dir = std::filesystem::path(path).parent_path();
        if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec) {
                error = "cannot create " + dir.string() + ": " + ec.message();
                return false;
            }
        }
        std::ifstream in(path, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (!line.empty())
                seen_.insert(line);
        }
        file_ = fopen(path.c_str(), "a");
        if (!file_) {
            error = "fopen " + path + ": " + strerror(errno);
            return false;
        }
        return true;
    }

    // Writes `basename` unless it's there already. True if it was new.
    bool add(const std::string& basename) {
        if (!seen_.insert(basename).second)
            return false;
        if (file_ && (fprintf(file_, "%s\n", basename.c_str()) < 0 || fflush(file_) != 0))
            fprintf(stderr, "ERROR: cannot record the basename %s in %s: %s\n", basename.c_str(), path_.c_str(), strerror(errno));
        return true;
    }

private:
    std::string path_;
    FILE* file_ = nullptr;
    std::set<std::string> seen_;
};

// `text` as a field of the fingerprint: its length, then itself - so that no
// two different requests can make the same string.
std::string field(const std::string& name, const std::string& text) {
    return name + "=" + std::to_string(text.size()) + ":" + text + "\n";
}

std::string trimmed(const std::string& s) {
    const size_t first = s.find_first_not_of(" \t\r");
    if (first == std::string::npos)
        return "";
    const size_t last = s.find_last_not_of(" \t\r");
    return s.substr(first, last - first + 1);
}

} // namespace

std::string formatDuration(double seconds) {
    if (seconds < 0 || seconds > 1e12)
        return "?";
    uint64_t s = (uint64_t) (seconds + 0.5);
    char buf[64];
    if (s >= 86400)
        snprintf(buf, sizeof buf, "%llud%02lluh%02llum", (unsigned long long) (s / 86400), (unsigned long long) (s / 3600 % 24),
                 (unsigned long long) (s / 60 % 60));
    else if (s >= 3600)
        snprintf(buf, sizeof buf, "%lluh%02llum%02llus", (unsigned long long) (s / 3600), (unsigned long long) (s / 60 % 60),
                 (unsigned long long) (s % 60));
    else if (s >= 60)
        snprintf(buf, sizeof buf, "%llum%02llus", (unsigned long long) (s / 60), (unsigned long long) (s % 60));
    else
        snprintf(buf, sizeof buf, "%llus", (unsigned long long) s);
    return buf;
}

uint64_t countDictionaryCandidates(const DictionarySpace& space, const std::string& prefix, const std::string& suffix,
                                   const FilenameBounds& bounds, uint64_t startNumber, uint64_t endNumber) {
    Walker walker(space, prefix, suffix, bounds, startNumber, endNumber, nullptr, nullptr);
    walker.run();
    return walker.counted();
}

std::string dictionaryFingerprint(const DictionaryRequest& req) {
    std::string text = "namebreak dictionary search 1\n";
    text += field("words", hex64(wordListChecksum(req.pattern.words)) + " " + std::to_string(req.pattern.words.size()));
    for (const std::string& separator : req.pattern.separators)
        text += field("separator", separator);
    text += field("words_per_candidate", std::to_string(req.pattern.minWords) + "-" + std::to_string(req.pattern.maxWords));
    text += field("prefix", req.prefix);
    text += field("suffix", req.suffix);
    text += req.bounds.hasLower ? field("lower", req.bounds.lower) : "lower=none\n";
    text += req.bounds.hasUpper ? field("upper", req.bounds.upper) : "upper=none\n";
    text += field("hashes", std::to_string(req.targetHashA) + " " + std::to_string(req.targetHashB));
    // The key, whenever it's known - it decides which candidates are compared
    // to the hashes - and whether the basenames are recorded, and every
    // hashA hit (a search whose basenames weren't recorded, or that only
    // compared the candidates matching the key to the hashes, mustn't be
    // resumed as one that did).
    if (!req.checkBasename) {
        text += "basename_key=none\n";
    } else {
        text += field("basename_key", std::to_string(req.basenameKey));
        if (!req.recordBasenames)
            text += "basenames=unrecorded\n";
        if (req.recordHashAMatches)
            text += "hasha_matches=every\n";
    }
    return hex64(fnv1a64(text));
}

std::vector<std::string> dictionaryResumableFingerprints(const DictionaryRequest& req) {
    // Whether a search compares every candidate to the hashes - rather than
    // only those whose basename matches its key - and records the basenames.
    auto comparesEvery = [](const DictionaryRequest& r) {
        DictionaryConstants constants;
        constants.suffix = r.suffix;
        constants.checkBasename = r.checkBasename;
        constants.recordBasenames = r.recordBasenames;
        constants.recordHashAMatches = r.recordHashAMatches;
        return dictionaryHashes(constants) != DictionaryHashes::Basename;
    };
    auto recordsBasenames = [](const DictionaryRequest& r) { return r.checkBasename && r.recordBasenames; };

    // Every search that differs from this one in those only: without a key,
    // and with this one's, its basenames recorded or not, every hashA hit
    // recorded or not.
    struct Variant {
        bool checkBasename, recordBasenames, recordHashAMatches;
    };
    std::vector<Variant> variants = {{false, false, false}};
    if (req.checkBasename) {
        for (bool record : {false, true}) {
            for (bool everyHashA : {false, true})
                variants.push_back({true, record, everyHashA});
        }
    }
    std::vector<std::string> fingerprints;
    DictionaryRequest earlier = req;
    for (const Variant& v : variants) {
        earlier.checkBasename = v.checkBasename;
        earlier.recordBasenames = v.recordBasenames;
        earlier.recordHashAMatches = v.recordHashAMatches;
        // It compared at least the candidates this one compares to the
        // hashes - all of them, or with the same key, those whose basename
        // matches it - and recorded the basenames if this one does.
        const bool compared = comparesEvery(earlier) || (!comparesEvery(req) && earlier.checkBasename);
        const bool recorded = !recordsBasenames(req) || recordsBasenames(earlier);
        const std::string fingerprint = dictionaryFingerprint(earlier);
        if (compared && recorded && std::find(fingerprints.begin(), fingerprints.end(), fingerprint) == fingerprints.end())
            fingerprints.push_back(fingerprint);
    }
    return fingerprints;
}

bool writeDictionaryProgress(const std::string& path, const DictionaryProgress& progress, uint64_t total,
                             const std::string& candidate, std::string& error) {
    std::string text = "# namebreak dictionary search progress - every candidate numbered below `next` has been searched\n";
    text += "fingerprint = " + progress.fingerprint + "\n";
    text += "next = " + std::to_string(progress.next) + "\n";
    text += "total = " + std::to_string(total) + "\n";
    text += "candidate = " + candidate + "\n";
    return replaceFileContents(path, text, error);
}

bool readDictionaryProgress(const std::string& path, DictionaryProgress& out, bool& exists, std::string& error) {
    std::error_code ec;
    exists = std::filesystem::exists(path, ec);
    if (!exists)
        return true;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    bool hasFingerprint = false, hasNext = false;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        const std::string t = trimmed(line);
        if (t.empty() || t[0] == '#')
            continue;
        const size_t eq = t.find('=');
        if (eq == std::string::npos) {
            error = path + ":" + std::to_string(lineNo) + ": expected 'key = value'";
            return false;
        }
        const std::string key = trimmed(t.substr(0, eq)), value = trimmed(t.substr(eq + 1));
        if (key == "fingerprint") {
            if (value.size() != 16 || value.find_first_not_of("0123456789abcdef") != std::string::npos) {
                error = path + ":" + std::to_string(lineNo) + ": the fingerprint must be 16 lowercase hex digits";
                return false;
            }
            out.fingerprint = value;
            hasFingerprint = true;
        } else if (key == "next") {
            errno = 0;
            char* end = nullptr;
            const unsigned long long next = strtoull(value.c_str(), &end, 10);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || errno == ERANGE || *end != '\0') {
                error = path + ":" + std::to_string(lineNo) + ": 'next' must be a number";
                return false;
            }
            out.next = next;
            hasNext = true;
        }
    }
    if (!hasFingerprint || !hasNext) {
        error = path + " isn't a dictionary search's progress file - it has no " + (hasFingerprint ? "'next'" : "'fingerprint'");
        return false;
    }
    return true;
}

DictionaryResult runDictionarySearch(SearchBackend& backend, const DictionaryRequest& req, std::atomic<bool>* abortRequested,
                                     std::function<void(const std::string&)> onPartialMatch, const std::atomic<bool>* pauseRequested) {
    DictionarySearchHooks hooks;
    hooks.onPartialMatch = std::move(onPartialMatch);
    return runDictionarySearch(backend, req, abortRequested, hooks, pauseRequested);
}

DictionaryResult runDictionarySearch(SearchBackend& backend, const DictionaryRequest& req, std::atomic<bool>* abortRequested,
                                     const DictionarySearchHooks& hooks, const std::atomic<bool>* pauseRequested) {
    DictionaryResult result;
    auto fail = [&](const std::string& error) {
        result.ok = false;
        result.error = error;
        return result;
    };
    if (!backend.supportsDictionary())
        return fail(std::string("the ") + backend.name() + " backend can't search dictionaries yet - use the cpu backend");
    DictionarySpace space;
    std::string error;
    if (!DictionarySpace::create(req.pattern, space, error))
        return fail(error);

    MatchWriter matches;
    if (!matches.open(req.outputFilePath, error))
        return fail(error);
    const bool recordBasenames = req.checkBasename && req.recordBasenames;
    BasenameWriter basenames;
    if (recordBasenames && !basenames.open(req.basenamesFilePath, error))
        return fail(error);

    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);

    const std::string fingerprint = dictionaryFingerprint(req);
    const uint64_t end = std::min(req.endNumber, space.size());
    const uint64_t start = std::min(req.startNumber, end);
    const uint64_t toSearch = countDictionaryCandidates(space, req.prefix, req.suffix, req.bounds, start, end);

    auto describeNumber = [&](uint64_t number) {
        return number < space.size() ? "'" + space.textOf(number) + "'" : std::string("(the end)");
    };
    std::string separators;
    for (const std::string& separator : req.pattern.separators)
        separators += (separators.empty() ? "\"" : ", \"") + separator + "\"";
    printf("dictionary search, backend: %s\n", backend.name());
    printf("words: %zu (%s, checksum %s)\n", space.words().size(), req.wordSource.c_str(),
           hex64(wordListChecksum(space.words())).c_str());
    printf("separators: %s\n", separators.c_str());
    printf("words per candidate: %d to %d\n", space.minWords(), space.maxWords());
    printf("prefix: '%s'\n", req.prefix.c_str());
    printf("suffix: '%s'\n", req.suffix.c_str());
    printf("lower: %s\n", req.bounds.hasLower ? ("'" + req.bounds.lower + "'").c_str() : "(none)");
    printf("upper: %s\n", req.bounds.hasUpper ? ("'" + req.bounds.upper + "'").c_str() : "(none)");
    printf("hashA: '%X'\n", req.targetHashA);
    printf("hashB: '%X'\n", req.targetHashB);
    DictionaryConstants constants;
    constants.words = space.words();
    constants.suffix = req.suffix;
    constants.cryptTable = cryptTable;
    constants.targetHashA = req.targetHashA;
    constants.targetHashB = req.targetHashB;
    constants.checkBasename = req.checkBasename;
    constants.basenameKey = req.basenameKey;
    constants.recordBasenames = req.recordBasenames;
    constants.recordHashAMatches = req.recordHashAMatches;
    if (req.checkBasename) {
        if (recordBasenames) {
            printf("basename key: '%X' - matching basenames go to %s\n", req.basenameKey,
                   req.basenamesFilePath.empty() ? "the coordinator" : req.basenamesFilePath.c_str());
            printf("basename matches expected by chance: %.3g (one per 4,294,967,296 candidates)\n", (double) toSearch / 4294967296.0);
        } else {
            printf("basename key: '%X' - matching basenames aren't recorded\n", req.basenameKey);
        }
        if (dictionaryHashes(constants) == DictionaryHashes::Basename)
            printf("only candidates whose basename matches the key are compared to hashA and hashB\n");
        else if (req.recordHashAMatches)
            printf("every candidate is compared to hashA and hashB (record_hasha_matches)\n");
    }
    printf("candidates: %llu in all, numbered 0 to %llu\n", (unsigned long long) space.size(),
           (unsigned long long) (space.size() - 1));
    if (end < space.size())
        printf("searching numbers %llu to %llu %s\n", (unsigned long long) start, (unsigned long long) (end - (end > start)),
               describeNumber(end - (end > start)).c_str());
    printf("starting at number %llu %s - %llu candidates to search within the bounds\n", (unsigned long long) start,
           describeNumber(start).c_str(), (unsigned long long) toSearch);
    if (!req.progressFilePath.empty())
        printf("progress file: %s\n", req.progressFilePath.c_str());
    fflush(stdout);

    backend.beginDictionarySearch(constants);

    result.nextNumber = start;
    const auto startTime = std::chrono::steady_clock::now();
    auto lastProgress = startTime, lastStatus = startTime;
    bool progressWarned = false;
    auto saveProgress = [&]() {
        if (req.progressFilePath.empty())
            return;
        std::string writeError;
        if (writeDictionaryProgress(req.progressFilePath, {fingerprint, result.nextNumber}, space.size(),
                                    result.nextNumber < space.size() ? space.textOf(result.nextNumber) : "", writeError)) {
            progressWarned = false;
        } else if (!progressWarned) {
            fprintf(stderr, "warning: cannot save the search's progress (%s) - trying again later\n", writeError.c_str());
            progressWarned = true;
        }
        lastProgress = std::chrono::steady_clock::now();
    };
    auto printStatus = [&]() {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        const double rate = elapsed > 0 ? (double) result.candidatesSearched / elapsed : 0;
        const double percent = toSearch ? 100.0 * (double) result.candidatesSearched / (double) toSearch : 100.0;
        printf("[%.2f%%] %llu of %llu candidates, %.2f M/s, about %s left - at number %llu %s\n", percent,
               (unsigned long long) result.candidatesSearched, (unsigned long long) toSearch, rate / 1e6,
               rate > 0 ? formatDuration((double) (toSearch - result.candidatesSearched) / rate).c_str() : "?",
               (unsigned long long) result.nextNumber, describeNumber(result.nextNumber).c_str());
        fflush(stdout);
        lastStatus = std::chrono::steady_clock::now();
    };

    const uint64_t perCall =
        std::max<uint64_t>(1, std::min<uint64_t>(backend.dictionaryCandidatesPerCall(), NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL));
    const size_t batchesPerCall = std::max<size_t>(1, std::min<size_t>(backend.dictionaryBatchesPerCall(), NAMEBREAK_DICTIONARY_BATCHES_PER_CALL));
    std::vector<DictionaryBatch> pending;
    uint64_t pendingCount = 0, pendingEnd = start;

    // Searches the batches gathered so far. False if the search is to stop:
    // aborted, or found.
    auto searchPending = [&]() -> bool {
        if (pending.empty())
            return true;
        if (abortRequested && abortRequested->load(std::memory_order_relaxed)) {
            result.aborted = true;
            return false;
        }
        if (pauseRequested && pauseRequested->load(std::memory_order_relaxed)) {
            saveProgress();
            if (!req.progressFilePath.empty())
                printf("[paused] progress saved to %s\n", req.progressFilePath.c_str());
            fflush(stdout);
            while (pauseRequested->load(std::memory_order_relaxed)) {
                if (abortRequested && abortRequested->load(std::memory_order_relaxed)) {
                    result.aborted = true;
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }

        const DictionaryOutcome outcome = backend.runDictionaryBatches(pending);
        result.candidatesSearched += pendingCount;
        result.nextNumber = pendingEnd;
        pending.clear();
        pendingCount = 0;

        for (const std::string& hit : outcome.hits) {
            printf("%s\n", hit.c_str());
            if (hooks.onPartialMatch)
                hooks.onPartialMatch(hit);
        }
        for (const std::string& hit : outcome.basenameHits) {
            const std::string basename = basenameOf(hit);
            if (basenames.add(basename)) {
                ++result.basenameHits;
                printf("BASENAME MATCH: %s (in %s)\n", basename.c_str(), hit.c_str());
                fflush(stdout);
                if (hooks.onBasenameMatch)
                    hooks.onBasenameMatch(basename, hit);
            }
        }
        if (hooks.onProgress)
            hooks.onProgress(result.nextNumber);
        if (outcome.found) {
            matches.found(outcome.foundFilename);
            printf("%s\n", outcome.foundFilename.c_str());
            printf("BOTH HASHES MATCH: %s\n", outcome.foundFilename.c_str());
            result.found = true;
            result.filename = outcome.foundFilename;
            return false;
        }
        if (!outcome.hits.empty())
            matches.hit(outcome.hits.back());
        matches.writeIfDue();

        const auto now = std::chrono::steady_clock::now();
        if (now - lastProgress >= req.progressInterval)
            saveProgress();
        if (now - lastStatus >= req.statusInterval)
            printStatus();
        return true;
    };

    Walker walker(space, req.prefix, req.suffix, req.bounds, start, end, cryptTable, [&](const Leaf& leaf) {
        uint32_t done = 0;
        while (done < leaf.wordCount) {
            const uint32_t take = (uint32_t) std::min<uint64_t>(leaf.wordCount - done, perCall - pendingCount);
            DictionaryBatch batch;
            batch.leading = *leaf.leading;
            batch.seed1 = leaf.hashA.first;
            batch.seed2 = leaf.hashA.second;
            batch.basenameSeed1 = leaf.basename.first;
            batch.basenameSeed2 = leaf.basename.second;
            batch.firstWord = leaf.firstWord + done;
            batch.wordCount = take;
            pending.push_back(std::move(batch));
            pendingCount += take;
            done += take;
            pendingEnd = leaf.firstNumber + done;
            if ((pendingCount >= perCall || pending.size() >= batchesPerCall) && !searchPending())
                return false;
        }
        return true;
    });
    if (walker.run() && searchPending()) {
        // Every candidate searched - and those after the last one within the
        // bounds count as searched too.
        result.nextNumber = end;
        if (hooks.onProgress)
            hooks.onProgress(result.nextNumber);
    }
    backend.endDictionarySearch();
    matches.flush();
    saveProgress();
    printStatus();
    if (!result.found && !result.aborted)
        printf("Searched every candidate - no match of both hashes\n");
    return result;
}
