#include "engine/meet_in_the_middle_search.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "engine/duration.h"
#include "engine/match_writer.h"
#include "engine/mpq_hash.h"
#include "engine/wordlist.h"

// The most tails one runMeetInTheMiddleBatch() call gets, whatever the
// backend asks for - set small at compile time by the tests (see
// CMakeLists.txt), so that even their small searches are split into many
// calls, and a tail length across several of them.
#ifndef NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_CALL
#define NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_CALL UINT64_MAX
#endif

void SearchBackend::beginMeetInTheMiddleSearch(const MeetInTheMiddleConstants&) {
    fprintf(stderr, "BUG: the %s backend can't search meet-in-the-middle targets - check supportsMeetInTheMiddle() first\n", name());
    std::abort();
}

MeetInTheMiddleOutcome SearchBackend::runMeetInTheMiddleBatch(const MeetInTheMiddleBatch&) {
    fprintf(stderr, "BUG: the %s backend can't search meet-in-the-middle targets - check supportsMeetInTheMiddle() first\n", name());
    std::abort();
}

namespace {

std::string trimmed(const std::string& s) {
    const size_t first = s.find_first_not_of(" \t\r");
    if (first == std::string::npos)
        return "";
    const size_t last = s.find_last_not_of(" \t\r");
    return s.substr(first, last - first + 1);
}

// `text` as a field of the fingerprint: its length, then itself - so that no
// two different requests can make the same string.
std::string field(const std::string& name, const std::string& text) {
    return name + "=" + std::to_string(text.size()) + ":" + text + "\n";
}

std::string hex32(uint32_t value) {
    char buf[16];
    snprintf(buf, sizeof buf, "%08X", value);
    return buf;
}

std::string stateText(HashState state) {
    return hex32(state.first) + " " + hex32(state.second);
}

} // namespace

bool meetingPointFitsHashes(const MeetInTheMiddleRequest& req, const uint32_t* cryptTable, std::string& error) {
    const uint32_t hashA = continueHash(req.meetingPointA, req.suffix, kHashAOffset, cryptTable).first;
    // Without hashB's meeting point there is only hashA's to check - see
    // MeetInTheMiddleRequest::hasMeetingPointB.
    const uint32_t hashB =
        req.hasMeetingPointB ? continueHash(req.meetingPointB, req.suffix, kHashBOffset, cryptTable).first : req.targetHashB;
    if (hashA == req.targetHashA && hashB == req.targetHashB)
        return true;
    // Whichever of the two doesn't fit - hashA first, if both don't.
    const bool aFits = hashA == req.targetHashA;
    const char* which = aFits ? "hashB" : "hashA";
    error = std::string("the meeting point for ") + which + " (" + stateText(aFits ? req.meetingPointB : req.meetingPointA) +
            ") doesn't fit the file: hashing the suffix '" + req.suffix + "' onto it gives " + hex32(aFits ? hashB : hashA) + ", not the " +
            which + " of " + hex32(aFits ? req.targetHashB : req.targetHashA) +
            " this search is for - the meeting point must be the state after the directory, right where the suffix begins";
    return false;
}

uint64_t meetInTheMiddleTailsToWalk(int alphabetSize, int maxLength, int tableLength) {
    MeetInTheMiddleSpace space;
    std::string error;
    if (!MeetInTheMiddleSpace::create(std::string(alphabetSize, 'x'), 1, maxLength, tableLength, space, error))
        return UINT64_MAX;
    return space.size();
}

std::string meetInTheMiddleFingerprint(const MeetInTheMiddleRequest& req) {
    std::string text = "namebreak meet-in-the-middle search 1\n";
    text += field("alphabet", req.alphabet);
    text += field("prefix", req.prefix);
    text += field("suffix", req.suffix);
    text += field("meeting_point_a", stateText(req.meetingPointA));
    text += field("meeting_point_b", req.hasMeetingPointB ? stateText(req.meetingPointB) : "none");
    text += field("hashes", std::to_string(req.targetHashA) + " " + std::to_string(req.targetHashB));
    text += field("lengths", std::to_string(req.minLength) + "-" + std::to_string(req.maxLength));
    // The table's length is part of it: it decides where the walk's nodes
    // are cut, so a `next` from a search with another one means something
    // else entirely.
    text += field("table_length", std::to_string(req.tableLength));
    text += field("rules", std::to_string(req.rules.symbolRuns) + " " + std::to_string(req.rules.unopenedBrackets) + " " +
                               std::to_string(req.rules.maxBackslashCount) + " " + std::to_string(req.rules.minBackslashCount) + " " +
                               std::to_string(req.rules.adjacentBackslashes));
    return hex64(fnv1a64(text));
}

bool writeMeetInTheMiddleProgress(const std::string& path, const MeetInTheMiddleProgress& progress, uint64_t total, const std::string& tail,
                           std::string& error) {
    std::string text = "# namebreak meet-in-the-middle search progress - every node numbered below `next` has been searched\n";
    text += "fingerprint = " + progress.fingerprint + "\n";
    text += "next = " + std::to_string(progress.next) + "\n";
    text += "total = " + std::to_string(total) + "\n";
    text += "tail = " + tail + "\n";
    return replaceFileContents(path, text, error);
}

bool readMeetInTheMiddleProgress(const std::string& path, MeetInTheMiddleProgress& out, bool& exists, std::string& error) {
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
        error = path + " isn't a meet-in-the-middle search's progress file - it has no " + (hasFingerprint ? "'next'" : "'fingerprint'");
        return false;
    }
    return true;
}

MeetInTheMiddleResult runMeetInTheMiddleSearch(SearchBackend& backend, const MeetInTheMiddleRequest& req, std::atomic<bool>* abortRequested,
                                 std::function<void(const std::string&)> onPartialMatch, const std::atomic<bool>* pauseRequested) {
    MeetInTheMiddleSearchHooks hooks;
    hooks.onPartialMatch = std::move(onPartialMatch);
    return runMeetInTheMiddleSearch(backend, req, abortRequested, hooks, pauseRequested);
}

MeetInTheMiddleResult runMeetInTheMiddleSearch(SearchBackend& backend, const MeetInTheMiddleRequest& req, std::atomic<bool>* abortRequested,
                                 const MeetInTheMiddleSearchHooks& hooks, const std::atomic<bool>* pauseRequested) {
    MeetInTheMiddleResult result;
    auto fail = [&](const std::string& error) {
        result.ok = false;
        result.error = error;
        return result;
    };
    if (!backend.supportsMeetInTheMiddle())
        return fail(std::string("the ") + backend.name() + " backend can't search meet-in-the-middle targets yet - use the cpu backend");

    uint32_t cryptTable[0x500];
    prepareCryptTable(cryptTable);

    std::string error;
    if (!meetingPointFitsHashes(req, cryptTable, error))
        return fail(error);
    if ((int) (req.prefix.size() + req.maxLength + req.suffix.size()) > MAX_FILENAME_LEN)
        return fail("the longest filename (" + std::to_string(req.prefix.size() + req.maxLength + req.suffix.size()) +
                    " characters) is longer than the " + std::to_string(MAX_FILENAME_LEN) + " a search allows");

    // The table's length: as asked for, or as much as its memory allows.
    int tableLength = req.tableLength;
    if (tableLength < 0) {
        tableLength = meetInTheMiddleTableLengthForBytes((int) req.alphabet.size(), req.maxLength, req.tableBytes);
    } else if (tableLength > req.maxLength) {
        printf("the table's %d characters are more than the longest directory's %d - using %d\n", tableLength, req.maxLength,
               req.maxLength);
        tableLength = req.maxLength;
    }

    MeetInTheMiddleSpace space;
    if (!MeetInTheMiddleSpace::create(req.alphabet, req.minLength, req.maxLength, tableLength, space, error))
        return fail(error);

    MatchWriter matches;
    if (!matches.open(req.outputFilePath, error))
        return fail(error);

    const std::string fingerprint = meetInTheMiddleFingerprint(req);
    const uint64_t end = std::min(req.endNumber, space.size());
    const uint64_t start = std::min(req.startNumber, end);
    const HashState startState = continueHash(kInitialHashState, req.prefix, kHashAOffset, cryptTable);
    const PruneState prefixEntry = dirPruneStart(req.prefix);
    // The directory, with the table's characters as '?' - what a node is
    // reported as.
    auto describeNumber = [&](uint64_t number) {
        return number < space.size() ? std::string(space.tableLength(), '?') + space.tailOf(number) : std::string();
    };

    printf("meet-in-the-middle search, backend: %s\n", backend.name());
    printf("prefix: '%s'\n", req.prefix.c_str());
    printf("suffix: '%s'\n", req.suffix.c_str());
    printf("alphabet: '%s' (%zu characters)\n", req.alphabet.c_str(), req.alphabet.size());
    printf("directory: %d to %d characters\n", space.minLength(), space.maxLength());
    printf("meeting point after the directory: hashA %s, hashB %s\n", stateText(req.meetingPointA).c_str(),
           req.hasMeetingPointB ? stateText(req.meetingPointB).c_str() : "(not given)");
    if (!req.hasMeetingPointB)
        printf("no meeting point for hashB: the one for hashA is all that was checked against the file's hashes - if it is for the "
               "wrong place in the name, this search will walk everything and find nothing\n");
    printf("hashA: '%X'\n", req.targetHashA);
    printf("hashB: '%X'\n", req.targetHashB);
    printf("table: the first %d characters of a directory, %llu of them at most, %llu MB\n", space.tableLength(),
           (unsigned long long) meetInTheMiddleTableCount((int) req.alphabet.size(), space.tableLength()),
           (unsigned long long) MeetInTheMiddleTable::bytesFor((int) req.alphabet.size(), space.tableLength()) / (1024 * 1024));
    printf("walk: the other %d characters, %llu tails in all, numbered 0 to %llu\n", space.maxTailLength(),
           (unsigned long long) space.size(), (unsigned long long) (space.size() - 1));
    if (end < space.size() || start > 0)
        printf("searching nodes %llu to %llu\n", (unsigned long long) start, (unsigned long long) (end - (end > start)));
    if (!req.progressFilePath.empty())
        printf("progress file: %s\n", req.progressFilePath.c_str());
    fflush(stdout);

    // Building the table also checks every directory shorter than it - the
    // walk never sees those, since it only ever adds characters to a
    // full-length prefix.
    const auto buildStart = std::chrono::steady_clock::now();
    std::vector<std::string> shortMatches;
    MeetInTheMiddleTable table;
    if (!MeetInTheMiddleTable::build(space, startState, prefixEntry, req.rules, cryptTable, req.meetingPointA, shortMatches, table, error))
        return fail(error);
    result.tableLength = space.tableLength();
    result.tableEntries = table.size();
    result.tableBytes = table.bytes();
    printf("table built in %s: %llu directories, %llu MB\n",
           formatDuration(std::chrono::duration<double>(std::chrono::steady_clock::now() - buildStart).count()).c_str(),
           (unsigned long long) table.size(), (unsigned long long) (table.bytes() / (1024 * 1024)));
    // How many directories are expected to match the meeting point without
    // being the file: one in 2^64 of the prefix-and-tail pairs the search
    // compares (fewer bits in the tests' dense-hit builds - see
    // NAMEBREAK_MEET_IN_THE_MIDDLE_MATCH_BITS, hash_match.h).
    const double pairs = (double) table.size() * (double) (end - start);
    printf("directories expected to match the meeting point by chance: %.3g\n", pairs / std::ldexp(1.0, 2 * NAMEBREAK_MEET_IN_THE_MIDDLE_MATCH_BITS));
    fflush(stdout);

    MeetInTheMiddleConstants constants;
    constants.alphabet = req.alphabet;
    constants.meetingPoint = req.meetingPointA;
    constants.cryptTable = cryptTable;
    constants.table = &table;
    constants.rules = req.rules;
    constants.tailEntry = tailPruneStart(req.suffix);

    result.nextNumber = start;
    const auto startTime = std::chrono::steady_clock::now();
    auto lastProgress = startTime, lastStatus = startTime;
    bool progressWarned = false;
    auto saveProgress = [&]() {
        if (req.progressFilePath.empty())
            return;
        std::string writeError;
        if (writeMeetInTheMiddleProgress(req.progressFilePath, {fingerprint, result.nextNumber}, space.size(), describeNumber(result.nextNumber),
                                  writeError)) {
            progressWarned = false;
        } else if (!progressWarned) {
            fprintf(stderr, "warning: cannot save the search's progress (%s) - trying again later\n", writeError.c_str());
            progressWarned = true;
        }
        lastProgress = std::chrono::steady_clock::now();
    };
    auto printStatus = [&]() {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        const uint64_t done = result.nextNumber - start, toSearch = end - start;
        const double rate = elapsed > 0 ? (double) done / elapsed : 0;
        const double percent = toSearch ? 100.0 * (double) done / (double) toSearch : 100.0;
        printf("[%.2f%%] %llu of %llu tails, %.2f M/s, about %s left - at %s\n", percent, (unsigned long long) done,
               (unsigned long long) toSearch, rate / 1e6, rate > 0 ? formatDuration((double) (toSearch - done) / rate).c_str() : "?",
               result.nextNumber < end ? describeNumber(result.nextNumber).c_str() : "(the end)");
        fflush(stdout);
        if (hooks.onStatus)
            hooks.onStatus(result.nextNumber < end ? describeNumber(result.nextNumber) : "");
        lastStatus = std::chrono::steady_clock::now();
    };

    // One directory that matched the meeting point: checked in full, reported,
    // and - if it's the file - the end of the search. False to stop.
    auto checkDir = [&](const std::string& dir) {
        // Within the lengths the search is for: the table holds directories
        // of its own length, which may be shorter than the shortest one
        // wanted - a tail of none then leaves one of those.
        if ((int) dir.size() < space.minLength() || (int) dir.size() > space.maxLength())
            return true;
        // The rules in full: each half of the search skips only what it can
        // tell breaks them on its own (see dirPasses).
        if (!dirPasses(req.rules, req.prefix, req.suffix, dir))
            return true;
        const std::string filename = req.prefix + dir + req.suffix;
        // Hashed from scratch, not from the walk's own state: a backend that
        // rewinds wrongly, or a table that holds the wrong directory, is
        // caught here rather than reported as a match.
        const HashState afterDir = continueHash(kInitialHashState, req.prefix + dir, kHashAOffset, cryptTable);
        if (meetingPointKey(afterDir) != meetingPointKey(req.meetingPointA)) {
            printf("WARNING: '%s' doesn't reach the meeting point after all - hashA's state after '%s%s' is %s, not %s\n", filename.c_str(),
                   req.prefix.c_str(), dir.c_str(), stateText(afterDir).c_str(), stateText(req.meetingPointA).c_str());
            return true;
        }
        ++result.hits;
        printf("%s\n", filename.c_str());
        if (hooks.onPartialMatch)
            hooks.onPartialMatch(filename);
        const uint32_t hashA = continueHash(afterDir, req.suffix, kHashAOffset, cryptTable).first;
        const uint32_t hashB = continueHash(kInitialHashState, filename, kHashBOffset, cryptTable).first;
        if (hashAMatches(hashA, req.targetHashA) && hashB == req.targetHashB) {
            matches.found(filename);
            printf("BOTH HASHES MATCH: %s\n", filename.c_str());
            fflush(stdout);
            result.found = true;
            result.filename = filename;
            return false;
        }
        matches.hit(filename);
        return true;
    };

    // One call's hits: which directories of the table's length end up where
    // each tail begins, and whether any of them is the file.
    auto checkHits = [&](const std::vector<MeetInTheMiddleHit>& hits) {
        std::vector<uint64_t> keys;
        std::vector<std::string> tails;
        keys.reserve(hits.size());
        tails.reserve(hits.size());
        for (const MeetInTheMiddleHit& hit : hits) {
            tails.push_back(space.tailOf(space.numberOf(hit.tailLength, hit.index)));
            keys.push_back(meetingPointKey(rewindHash(req.meetingPointA, tails.back(), kHashAOffset, cryptTable)));
        }
        // One walk of the table's space answers for every hit of the call.
        std::vector<std::vector<std::string>> prefixes;
        table.prefixesFor(keys, prefixes);

        for (size_t i = 0; i < hits.size(); ++i) {
            for (const std::string& prefix : prefixes[i]) {
                if (!checkDir(prefix + tails[i]))
                    return false;
            }
        }
        return true;
    };

    // The directories shorter than the table's own length, which the build
    // checked on the way - they are whole directories already, not a prefix
    // and a tail. They belong to the search that has the walk's first node
    // in its range, so that a resumed search, or one range of several,
    // doesn't report them all over again.
    bool stop = false;
    if (start == 0) {
        for (const std::string& dir : shortMatches) {
            if (!checkDir(dir)) {
                stop = true;
                break;
            }
        }
    }

    // Nothing to walk if one of those was the file: a backend would upload
    // the whole table for a search that's already over.
    if (!stop)
        backend.beginMeetInTheMiddleSearch(constants);
    const uint64_t perCall =
        std::max<uint64_t>(1, std::min<uint64_t>(backend.meetInTheMiddleTailsPerCall(), NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_CALL));
    uint64_t number = start;
    while (!stop && number < end) {
        if (abortRequested && abortRequested->load(std::memory_order_relaxed)) {
            result.aborted = true;
            break;
        }
        if (pauseRequested && pauseRequested->load(std::memory_order_relaxed)) {
            saveProgress();
            if (!req.progressFilePath.empty())
                printf("[paused] progress saved to %s\n", req.progressFilePath.c_str());
            fflush(stdout);
            while (pauseRequested->load(std::memory_order_relaxed)) {
                if (abortRequested && abortRequested->load(std::memory_order_relaxed)) {
                    result.aborted = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            if (result.aborted)
                break;
        }

        // One call: tails of one length only, since a batch is a range of
        // one length's indices.
        MeetInTheMiddleBatch batch;
        batch.tailLength = space.tailLengthOf(number);
        batch.start = space.indexOf(number);
        batch.count = std::min(std::min(end - number, space.countOf(batch.tailLength) - batch.start), perCall);
        MeetInTheMiddleOutcome outcome = backend.runMeetInTheMiddleBatch(batch);
        number += batch.count;
        result.nextNumber = number;
        result.tailsProbed += outcome.probed;

        // In the order the walk reached them, whichever order the backend
        // reports them in.
        std::sort(outcome.hits.begin(), outcome.hits.end(), [](const MeetInTheMiddleHit& a, const MeetInTheMiddleHit& b) {
            return a.tailLength != b.tailLength ? a.tailLength < b.tailLength : a.index < b.index;
        });
        if (!checkHits(outcome.hits))
            break;

        if (hooks.onProgress)
            hooks.onProgress(result.nextNumber);
        if (hooks.onCount)
            hooks.onCount(result.nextNumber - start, end - start);
        matches.writeIfDue();
        const auto now = std::chrono::steady_clock::now();
        if (now - lastProgress >= req.progressInterval)
            saveProgress();
        if (now - lastStatus >= req.statusInterval)
            printStatus();
    }
    if (!stop)
        backend.endMeetInTheMiddleSearch();

    if (!result.found && !result.aborted) {
        result.nextNumber = end;
        if (hooks.onProgress)
            hooks.onProgress(result.nextNumber);
    }
    matches.flush();
    saveProgress();
    printStatus();
    printf("%llu of %llu tails looked up in the table - the rules skipped the rest\n", (unsigned long long) result.tailsProbed,
           (unsigned long long) (result.nextNumber - start));
    if (!result.found && !result.aborted)
        printf("Searched every directory - no match of both hashes\n");
    return result;
}
