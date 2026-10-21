#include "net/coordinator_runner.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>

#include "backends/backends.h"
#include "backends/self_test.h"
#include "common/config.h"
#include "common/matches_file.h"
#include "common/version.h"
#include "net/basename_outbox.h"
#include "net/coordinator_client.h"
#include "net/word_lists.h"
#include "common/string_util.h"
#include "engine/candidate.h"
#include "common/platform.h"
#include "engine/dictionary_search.h"
#include "engine/meet_in_the_middle.h"
#include "engine/meet_in_the_middle_search.h"
#include "engine/match_writer.h"
#include "engine/search.h"
#include "engine/wordlist.h"

namespace {

// Fixed rather than derived from a range's lease, so progress checkpoints
// (and the liveness signal the server's reclaim sweep relies on) land at a
// steady, predictable cadence regardless of how big a range is or how fast a
// client is. Set to a second at compile time by a test build (see
// CMakeLists.txt), to see heartbeats fail and recover in a test's time.
#ifndef NAMEBREAK_HEARTBEAT_INTERVAL_SECONDS
#define NAMEBREAK_HEARTBEAT_INTERVAL_SECONDS 60
#endif
constexpr int kHeartbeatIntervalSeconds = NAMEBREAK_HEARTBEAT_INTERVAL_SECONDS;

// Ceiling for RetryBackoff below, so a prolonged outage doesn't leave a
// client waiting arbitrarily long once the server comes back.
constexpr std::chrono::seconds kMaxRetryBackoff{600};

// Exponential backoff, capped at kMaxRetryBackoff, so a real outage doesn't
// get hammered at the normal poll rate. Jittered (sleep a random fraction of
// the current backoff rather than its full length) so a fleet of clients
// that all started failing at once - e.g. the coordinator itself going down -
// doesn't retry in lockstep once it recovers.
class RetryBackoff {
public:
    // Never below 1s, so even a poll_interval_secs of 0 still backs off.
    explicit RetryBackoff(std::chrono::seconds initial) : initial_(std::max(initial, std::chrono::seconds(1))), current_(initial_) {}

    // The ceiling the next call to nextSleep() draws from - for logging.
    std::chrono::seconds cap() const { return current_; }

    // How long to sleep before the next attempt; doubles the cap for the one after.
    std::chrono::seconds nextSleep() {
        std::uniform_int_distribution<long long> jitter(0, current_.count());
        auto sleepFor = std::chrono::seconds(jitter(rng_));
        current_ = std::min(current_ * 2, kMaxRetryBackoff);
        return sleepFor;
    }

    void reset() { current_ = initial_; }

private:
    std::chrono::seconds initial_;
    std::chrono::seconds current_;
    std::mt19937 rng_{std::random_device{}()};
};

// Only libcurl needs pairing at process scope - calling curl_global_init()
// once up front (before the heartbeat thread starts making concurrent
// requests) and curl_global_cleanup() on every path back out of
// runCoordinator, including early failure, rather than leaving the init
// unmatched.
struct CurlGlobalGuard {
    CurlGlobalGuard()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobalGuard() { curl_global_cleanup(); }
};

// Sleeps for `duration`, but returns early (in at most ~500ms) once
// `quitRequested` becomes true - so a caller wanting to quit promptly (e.g.
// a GUI's Quit button) doesn't have to wait out a long claim-backoff or
// no-work poll interval once it's requested. A no-op wait (just the plain
// full-duration sleep) when `quitRequested` is null, matching every existing
// call site's behavior before this was introduced.
void interruptibleSleep(std::chrono::milliseconds duration, const std::atomic<bool>* quitRequested) {
    constexpr auto kTick = std::chrono::milliseconds(500);
    while (duration.count() > 0) {
        if (quitRequested && quitRequested->load(std::memory_order_relaxed))
            return;
        auto step = std::min(duration, kTick);
        std::this_thread::sleep_for(step);
        duration -= step;
    }
}

} // namespace

// Interactively confirms (or lets the user override) auto-discovered values
// for whichever of `args`'s username/hostname were left unset by
// args.configPath, then persists the final values back into that same file -
// so this only ever prompts once per machine (future runs find both keys
// already set there). Falls back to silently using the discovered defaults
// with no prompt/write when stdin isn't a terminal (e.g. a cron job or
// systemd service), so a missing config never hangs a non-interactive run.
// Declared in coordinator_runner.h - see there for why main() calls it too.
void resolveMissingIdentity(CoordinatorArgs& args) {
    struct Field {
        std::string key;
        std::string* value;
    };
    std::vector<Field> missing;
    if (args.username.empty()) missing.push_back({"username", &args.username});
    if (args.hostname.empty()) missing.push_back({"hostname", &args.hostname});
    if (missing.empty()) return;

    for (Field& f : missing) {
        *f.value = (f.key == "username") ? resolveUsername() : resolveHostname();
    }

    if (!isInteractiveTerminal()) {
        for (const Field& f : missing) {
            fprintf(stderr, "[coordinator] no %s configured - using detected value '%s' (run interactively to save this to %s)\n",
                    f.key.c_str(), f.value->c_str(), args.configPath.c_str());
        }
        return;
    }

    printf("No %s configured in %s. Detected:\n", missing.size() == 2 ? "username/hostname" : missing[0].key.c_str(), args.configPath.c_str());
    for (const Field& f : missing)
        printf("  %s = %s\n", f.key.c_str(), f.value->c_str());
    printf("Use %s? [Y/n]: ", missing.size() == 2 ? "these" : "this");
    fflush(stdout);

    std::string line;
    std::getline(std::cin, line);
    line = trim(line);
    bool accepted = line.empty() || line == "y" || line == "Y" || line == "yes" || line == "Yes";

    for (Field& f : missing) {
        if (!accepted) {
            printf("Enter %s [%s]: ", f.key.c_str(), f.value->c_str());
            fflush(stdout);
            std::string entered;
            std::getline(std::cin, entered);
            entered = trim(entered);
            if (!entered.empty())
                *f.value = entered;
        }
        if (appendKeyToConfigSection(args.configPath, "coordinator", f.key, *f.value)) {
            printf("[coordinator] saved %s = %s to %s\n", f.key.c_str(), f.value->c_str(), args.configPath.c_str());
        } else {
            fprintf(stderr, "[coordinator] warning: could not save %s to %s - you'll be asked again next time\n", f.key.c_str(), args.configPath.c_str());
        }
    }
}

namespace {

// What a claim says is known about its target's file beyond its hashes (see
// ClaimResponse::encryptionKeyHex) - what a match of both hashes is checked
// against. False, with `error` set, if the key isn't hex.
bool toMatchCheck(const ClaimResponse& claim, MatchCheck& check, std::string& error) {
    check = MatchCheck();
    if (!claim.encryptionKeyHex.empty()) {
        if (!hexToU32(claim.encryptionKeyHex, check.key)) {
            error = "malformed encryption key in claim response (" + claim.encryptionKeyHex + ")";
            return false;
        }
        check.checkKey = true;
    }
    // parseClaimResponse has made sure they're all there or none, and fit.
    if (claim.hashTableSize) {
        check.checkSlot = true;
        check.hashTableSize = (uint32_t) *claim.hashTableSize;
        check.firstStartSlot = (uint32_t) *claim.hashTableFirstStartSlot;
        check.lastStartSlot = (uint32_t) *claim.hashTableLastStartSlot;
    }
    return true;
}

SearchRequest toSearchRequest(const ClaimResponse& claim, const std::string& matchesDir, std::string& error) {
    SearchRequest req;
    req.alphabet = claim.alphabet;
    req.maxBackslashCount = (int) claim.maxBackslashCount;
    req.minBackslashCount = (int) claim.minBackslashCount;
    req.prefix = claim.prefix;
    req.suffix = claim.suffix;
    req.lowerBound = removePrefixAndSuffix(claim.lowerBoundFilename, req.prefix, req.suffix);
    req.upperBound = removePrefixAndSuffix(claim.upperBoundFilename, req.prefix, req.suffix);
    // A claimed range always starts exactly at its own lower bound (unlike
    // local "continuous" mode's arbitrary mid-space resume point).
    req.startCandidate = req.lowerBound;
    req.pruneSymbolRuns = claim.pruneSymbolRuns;
    req.pruneUnopenedBrackets = claim.pruneUnopenedBrackets;
    req.pruneWholeCandidate = claim.pruneWholeCandidate;
    req.pruneAdjacentBackslashes = claim.pruneAdjacentBackslashes;
    req.insertFromStart = claim.insertFromStart;
    req.insertFromEnd = claim.insertFromEnd;
    req.continuous = false; // a coordinator range is always run "bounded"
    req.outputFilePath = matchesFilePath(matchesDir, claim.targetName);
    if (!hexToU32(claim.hashAHex, req.targetHashA) || !hexToU32(claim.hashBHex, req.targetHashB)) {
        error = "malformed target hash in claim response (hashA=" + claim.hashAHex + " hashB=" + claim.hashBHex + ")";
    }
    if (error.empty())
        toMatchCheck(claim, req.matchCheck, error);
    return req;
}

// The backend dictionary ranges are searched with: the one this client
// searches everything with, once it has passed the dictionary self-test
// (selfTestDictionaryBackend) - which the first dictionary range runs, so a
// client that never gets one never waits for it - or else, should it fail
// it, the cpu backend.
class DictionaryBackend {
public:
    explicit DictionaryBackend(SearchBackend& main) : main_(main) {}

    // Null, with `error` set, if no backend can search dictionaries here.
    SearchBackend* get(std::string& error) {
        if (chosen_)
            return chosen_;
        std::string testError;
        if (main_.supportsDictionary() && selfTestDictionaryBackend(main_, testError)) {
            chosen_ = &main_;
            return chosen_;
        }
        fprintf(stderr, "[coordinator] the %s backend %s - searching dictionary targets with the cpu backend instead\n", main_.name(),
                main_.supportsDictionary() ? ("failed the dictionary self-test: " + testError).c_str() : "can't search dictionaries");
        cpu_ = createDictionaryBackend("cpu", error);
        chosen_ = cpu_.get();
        return chosen_;
    }

private:
    SearchBackend& main_;
    std::unique_ptr<SearchBackend> cpu_;
    SearchBackend* chosen_ = nullptr;
};

// The backend a meet-in-the-middle target's range is searched on - the main
// one if it can and passes the meet-in-the-middle self-test, else the cpu
// backend, as a dictionary range's is.
class MeetInTheMiddleBackend {
public:
    explicit MeetInTheMiddleBackend(SearchBackend& main) : main_(main) {}

    // Null, with `error` set, if no backend can search meet-in-the-middle targets here.
    SearchBackend* get(std::string& error) {
        if (chosen_)
            return chosen_;
        std::string testError;
        if (main_.supportsMeetInTheMiddle() && selfTestMeetInTheMiddleBackend(main_, testError)) {
            chosen_ = &main_;
            return chosen_;
        }
        fprintf(stderr, "[coordinator] the %s backend %s - searching meet-in-the-middle targets with the cpu backend instead\n", main_.name(),
                main_.supportsMeetInTheMiddle() ? ("failed the meet-in-the-middle self-test: " + testError).c_str()
                                                : "can't search meet-in-the-middle targets");
        cpu_ = createMeetInTheMiddleBackend("cpu", error);
        chosen_ = cpu_.get();
        return chosen_;
    }

private:
    SearchBackend& main_;
    std::unique_ptr<SearchBackend> cpu_;
    SearchBackend* chosen_ = nullptr;
};

// The false matches a range's search has found that the server hasn't said
// it has (see HeartbeatRequest::falseMatches) - added to from the search's
// thread, and sent and removed from the heartbeat thread's.
class FalseMatchOutbox {
public:
    void add(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        filenames_.push_back(filename);
    }
    // Every one waiting - a report carries them all.
    std::vector<std::string> peek() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return filenames_;
    }
    // The first `count`, which a report (from peek) carried and got an
    // answer to - any found since come after them.
    void remove(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        filenames_.erase(filenames_.begin(), filenames_.begin() + (ptrdiff_t) std::min(count, filenames_.size()));
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::string> filenames_;
};

// How a range's search went, whichever kind of target it's of.
struct RangeResult {
    bool ok = true;
    std::string error;
    bool found = false;
    std::string filename;
    bool aborted = false;
};

// One claimed range's search, whichever kind of target it's of - what
// runRange needs to run it and report on it.
struct RangeSearch {
    // Runs the search, on the calling thread; `abort` stops it.
    std::function<RangeResult(std::atomic<bool>& abort)> run;
    // An alphabet range's progress, as a report says it (see
    // HeartbeatRequest): its latest Hash A match's filename. Called from the
    // heartbeat thread while `run` runs.
    std::function<void(std::optional<std::string>& lastHashAMatch)> progress;
    // A dictionary range's progress and the basenames waiting to be sent -
    // null for any other kind of range.
    BasenameOutbox* outbox = nullptr;
    // A meet-in-the-middle range's progress: every tail numbered below this
    // has been looked up - null for any other kind of range. Called from the
    // heartbeat thread while `run` runs.
    std::function<std::optional<int64_t>()> nextTailIndex;
    // Where a match it finds is recorded (see MatchWriter).
    std::string outputFilePath;
    // Whatever kind of range it is, the false matches it found and hasn't
    // had delivered yet - every report carries them.
    FalseMatchOutbox falseMatches;
};

// What a report (a HeartbeatRequest or QuitRequest) of `search` says: a
// dictionary range's progress and the basenames it carries - never further
// than the first basename it leaves waiting (see BasenameOutbox) - or an
// alphabet range's latest Hash A match.
template <class Report>
void fillReport(const RangeSearch& search, Report& req) {
    req.falseMatches = search.falseMatches.peek();
    if (search.nextTailIndex) {
        req.nextTailIndex = search.nextTailIndex();
    } else if (search.outbox) {
        BasenameOutbox::Report report = search.outbox->peek(kMaxBasenamesPerReport);
        req.nextCandidateNumber = (int64_t) report.next;
        req.basenames = std::move(report.basenames);
    } else {
        search.progress(req.lastHashAMatchFilename);
    }
}

// After a report that carried `sent` got an answer that means the server
// has them - see HeartbeatRequest::basenames.
void basenamesDelivered(BasenameOutbox* outbox, const std::vector<std::string>& sent) {
    if (outbox && !sent.empty())
        outbox->remove(sent.size());
}

// The same for the false matches a report carried - the server keeps them
// whatever else it answers, as it does the basenames.
void falseMatchesDelivered(FalseMatchOutbox* outbox, const std::vector<std::string>& sent) {
    if (outbox && !sent.empty())
        outbox->remove(sent.size());
}

// "searched up to <filename>" or "searched up to number <n>", as a quit
// report gives it.
std::string describeProgress(const QuitRequest& req) {
    if (req.nextCandidateNumber)
        return "searched every candidate numbered below " + std::to_string(*req.nextCandidateNumber);
    if (req.nextTailIndex)
        return "looked up every tail numbered below " + std::to_string(*req.nextTailIndex);
    return "searched up to " + *req.lastHashAMatchFilename;
}

// Tells the server this client is quitting with `rangeId` unfinished, so
// that what's left of it is handed out again straight away rather than once
// its lease expires - see QuitRequest. `req` says how far the search got, if
// it got anywhere, and carries the basenames waiting in `outbox` (see
// fillReport). One attempt only: the user asked to quit, and if the server
// can't be reached the lease still runs out as before. Either way, any
// basename it didn't get is in what's handed out again. So are the false
// matches in `falseMatches` it carries - found again when that part is
// searched again, if they're past how far it says the search got.
void reportQuit(const CoordinatorArgs& args, const std::string& token, int64_t rangeId, const QuitRequest& req, BasenameOutbox* outbox,
                FalseMatchOutbox* falseMatches) {
    CoordinatorClient quitClient(args.serverUrl);
    quitClient.setToken(token);
    const bool progressed = req.lastHashAMatchFilename || req.nextCandidateNumber || req.nextTailIndex;
    std::string err;
    switch (quitClient.quit(rangeId, req, err)) {
        case CoordinatorClient::QuitOutcome::Ok:
            basenamesDelivered(outbox, req.basenames);
            falseMatchesDelivered(falseMatches, req.falseMatches);
            if (progressed)
                printf("[coordinator] range %lld: told the coordinator it's %s - the rest goes back to be handed out\n", (long long) rangeId,
                       describeProgress(req).c_str());
            else
                printf("[coordinator] range %lld: told the coordinator - no progress to report, so the whole range goes back to be handed out\n",
                       (long long) rangeId);
            break;
        case CoordinatorClient::QuitOutcome::Conflict:
            basenamesDelivered(outbox, req.basenames);
            falseMatchesDelivered(falseMatches, req.falseMatches);
            printf("[coordinator] range %lld: no longer assigned to us - nothing to report\n", (long long) rangeId);
            break;
        case CoordinatorClient::QuitOutcome::Error:
            fprintf(stderr, "[coordinator] range %lld: couldn't tell the coordinator we're quitting (%s) - its lease will expire instead\n",
                    (long long) rangeId, err.c_str());
            break;
    }
}

// Sends the basenames of a finished dictionary range that are more than its
// completion can carry, in heartbeats of kMaxBasenamesPerReport - each saying
// the search has got no further than the first basename it leaves waiting
// (see fillReport) - retried until they get an answer, as a completion is.
// False if the range can't be completed after all: the server says it isn't
// this client's any more, or released it (its target solved, say), or the
// client is quitting. Whatever wasn't sent then is in what the server hands
// out again.
bool sendBasenamesBeforeCompleting(const CoordinatorArgs& args, CoordinatorClient& client, const ClaimResponse& claim, BasenameOutbox& outbox,
                                   const std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    if (outbox.size() <= kMaxBasenamesPerReport)
        return true;
    printf("[coordinator] range %lld: sending the %zu basenames found before completing it, %zu at a time\n", (long long) claim.rangeId, outbox.size(),
           kMaxBasenamesPerReport);
    RetryBackoff backoff(std::chrono::seconds(args.pollIntervalSecs));
    while (outbox.size() > kMaxBasenamesPerReport) {
        HeartbeatRequest req;
        BasenameOutbox::Report report = outbox.peek(kMaxBasenamesPerReport);
        req.nextCandidateNumber = (int64_t) report.next;
        req.basenames = std::move(report.basenames);
        HeartbeatResponse resp;
        std::string err;
        switch (client.heartbeat(claim.rangeId, req, resp, err)) {
            case CoordinatorClient::HeartbeatOutcome::Ok:
                outbox.remove(req.basenames.size());
                backoff.reset();
                if (resp.rangeReleased) {
                    printf("[coordinator] range %lld: released by the coordinator before it could be completed\n", (long long) claim.rangeId);
                    return false;
                }
                break;
            case CoordinatorClient::HeartbeatOutcome::Conflict:
                outbox.remove(req.basenames.size());
                printf("[coordinator] range %lld: no longer assigned to us - the rest of it is someone else's to search\n", (long long) claim.rangeId);
                return false;
            case CoordinatorClient::HeartbeatOutcome::Error: {
                if (quitRequested && quitRequested->load(std::memory_order_relaxed)) {
                    fprintf(stderr, "[coordinator] range %lld: quitting with %zu basenames not sent - the part of the range they're in is handed out again\n",
                            (long long) claim.rangeId, outbox.size());
                    return false;
                }
                auto cap = backoff.cap();
                auto sleepFor = backoff.nextSleep();
                std::string msg = "[coordinator] range " + std::to_string(claim.rangeId) + ": failed to send basenames, retrying in " +
                                  std::to_string(sleepFor.count()) + "s (backoff cap " + std::to_string(cap.count()) + "s): " + err;
                fprintf(stderr, "%s\n", msg.c_str());
                if (callbacks && callbacks->onStatus)
                    callbacks->onStatus(msg);
                interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(sleepFor), quitRequested);
                break;
            }
        }
    }
    return true;
}

// Runs one claimed range: spawns the heartbeat thread, runs `search` on the
// calling thread, then reports completion - or, if quitting, how far it got.
// `quitRequested`/`callbacks`, if given, are as documented on runCoordinator
// (coordinator_runner.h).
void runRange(const CoordinatorArgs& args, const std::string& token, const ClaimResponse& claim, RangeSearch& search,
              const std::atomic<bool>* pauseRequested, std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    std::atomic<bool> abortRequested{false};
    // Only ever written by the heartbeat thread below, right before it
    // breaks out of its loop, and only ever read after heartbeatThread.join()
    // - that join() is itself a synchronizes-with edge, so a plain bool
    // (unlike abortRequested, which the search thread polls concurrently
    // while the heartbeat thread is still running) needs no atomic here.
    // Captured at the moment the search is signaled to stop, not re-read
    // later, to avoid any raciness against the user unpausing in between.
    // Set by either stopping reason below (an ordinary release, or losing
    // ownership outright); the messages below just word it differently
    // depending on which one it was (see lostOwnership).
    bool wasPausedWhenStopped = false;
    // Set (alongside wasPausedWhenStopped, not instead of it) specifically
    // when a heartbeat finds out this range's ownership had already moved on
    // by the time it was sent - see the HeartbeatOutcome::Conflict handling
    // below - as opposed to an ordinary rangeReleased response.
    bool lostOwnership = false;

    std::mutex stopMutex;
    std::condition_variable stopCv;
    bool stopRequested = false;

    // Watches `quitRequested` (a caller-owned atomic - e.g. a GUI's Quit
    // button, see coordinator_runner.h) and relays it into the same
    // `abortRequested` the heartbeat thread above already uses for a
    // server-signaled release, so the search needs no separate awareness of
    // it. Simple polling rather than a condition_variable like the heartbeat
    // thread's wait, since there's no event to wait on here - just
    // `quitRequested` occasionally flipping true from another thread - and
    // ~200ms is frequent enough to keep a GUI's Quit button feeling
    // responsive without busy-waiting.
    std::atomic<bool> quitWatcherStop{false};
    std::thread quitWatcherThread;
    if (quitRequested) {
        quitWatcherThread = std::thread([&]() {
            while (!quitWatcherStop.load(std::memory_order_relaxed)) {
                if (quitRequested->load(std::memory_order_relaxed)) {
                    abortRequested.store(true, std::memory_order_relaxed);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }

    std::thread heartbeatThread([&]() {
        CoordinatorClient hbClient(args.serverUrl);
        hbClient.setToken(token);
        std::unique_lock<std::mutex> lock(stopMutex);
        // Set when more basenames are waiting than a heartbeat carries: the
        // next one goes at once, rather than hold the range's progress back
        // (see fillReport) for another interval.
        bool sendAgainNow = false;
        while (!stopCv.wait_for(lock, std::chrono::seconds(sendAgainNow ? 0 : kHeartbeatIntervalSeconds), [&] { return stopRequested; })) {
            HeartbeatRequest req;
            fillReport(search, req);
            lock.unlock();
            sendAgainNow = false;
            HeartbeatResponse resp;
            std::string err;
            switch (hbClient.heartbeat(claim.rangeId, req, resp, err)) {
                case CoordinatorClient::HeartbeatOutcome::Ok:
                    basenamesDelivered(search.outbox, req.basenames);
                    falseMatchesDelivered(&search.falseMatches, req.falseMatches);
                    sendAgainNow = req.basenames.size() == kMaxBasenamesPerReport && search.outbox->size() > 0;
                    if (resp.rangeReleased) {
                        // The server doesn't say why (target solved elsewhere, or
                        // this range stalled too long - see HeartbeatResponse's
                        // doc comment in protocol.h) - our own pause state is
                        // enough to pick the right message and, more importantly,
                        // is what runCoordinator's claim loop already keys off to
                        // decide whether to go claim a new range or stay idle.
                        wasPausedWhenStopped = pauseRequested && pauseRequested->load(std::memory_order_relaxed);
                        if (wasPausedWhenStopped) {
                            printf("[coordinator] range %lld: claim released - no progress was reported while paused. "
                                   "Still paused; no new work will be claimed until resumed.\n",
                                   (long long) claim.rangeId);
                        } else {
                            printf("[coordinator] range %lld: target already solved elsewhere - signaling abort\n", (long long) claim.rangeId);
                        }
                        abortRequested.store(true, std::memory_order_relaxed);
                        lock.lock();
                        goto stopHeartbeating;
                    }
                    break;
                case CoordinatorClient::HeartbeatOutcome::Conflict:
                    // The server kept the basenames all the same.
                    basenamesDelivered(search.outbox, req.basenames);
                    falseMatchesDelivered(&search.falseMatches, req.falseMatches);
                    // Not a failure, and not something retrying will ever fix: by
                    // the time this heartbeat reached the server, it had already
                    // stopped considering this range ours - almost always because
                    // the lease (kHeartbeatIntervalSeconds-ish) lapsed while this
                    // client wasn't heartbeating at all (paused, or the machine
                    // itself was suspended/hibernated), so the server released or
                    // reassigned the range before we ever got a chance to say
                    // otherwise. Stop heartbeating (every next attempt would just
                    // get the same answer) and signal the search to stop too,
                    // exactly like an ordinary release above.
                    lostOwnership = true;
                    wasPausedWhenStopped = pauseRequested && pauseRequested->load(std::memory_order_relaxed);
                    if (wasPausedWhenStopped) {
                        printf("[coordinator] range %lld: no longer assigned to us - unsurprising, since this client "
                               "has been paused (and therefore not heartbeating) for a while. Still paused; no new "
                               "work will be claimed until resumed.\n",
                               (long long) claim.rangeId);
                    } else {
                        printf("[coordinator] range %lld: no longer assigned to us (the lease likely lapsed while "
                               "this client was inactive) - stopping\n",
                               (long long) claim.rangeId);
                    }
                    abortRequested.store(true, std::memory_order_relaxed);
                    lock.lock();
                    goto stopHeartbeating;
                case CoordinatorClient::HeartbeatOutcome::Error:
                    // The basenames stay, for the next report.
                    fprintf(stderr, "[coordinator] range %lld: heartbeat failed: %s\n", (long long) claim.rangeId, err.c_str());
                    break;
            }
            lock.lock();
        }
    stopHeartbeating:;
    });

    auto started = std::chrono::steady_clock::now();
    const RangeResult result = search.run(abortRequested);
    double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    // Read once, right as the search stops, rather than re-checked later -
    // same reasoning as wasPausedWhenStopped above (avoids raciness against
    // quitRequested changing again in between).
    bool externallyQuit = quitRequested && quitRequested->load(std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(stopMutex);
        stopRequested = true;
    }
    stopCv.notify_all();
    heartbeatThread.join();
    quitWatcherStop.store(true, std::memory_order_relaxed);
    if (quitWatcherThread.joinable())
        quitWatcherThread.join();

    if (!result.ok) {
        fprintf(stderr, "[coordinator] range %lld: search failed (%s) - letting the lease expire so it gets reassigned\n",
                (long long) claim.rangeId, result.error.c_str());
        if (callbacks && callbacks->onRangeFinished)
            callbacks->onRangeFinished(false);
        return;
    }

    if (result.aborted) {
        // Checked first and independently of lostOwnership/wasPausedWhenStopped
        // above - those two both describe reasons the *server* ended this
        // range, which externallyQuit (a caller-owned atomic, e.g. a GUI's
        // Quit button - see coordinator_runner.h) is not, even though it
        // reaches the same abortRequested flag the search polls.
        if (externallyQuit && !lostOwnership && !wasPausedWhenStopped) {
            // Ours until now, so tell the server we're done with it - see
            // reportQuit.
            printf("[coordinator] range %lld: aborted - quit requested\n", (long long) claim.rangeId);
            QuitRequest req;
            fillReport(search, req);
            reportQuit(args, token, claim.rangeId, req, search.outbox, &search.falseMatches);
        } else if (externallyQuit) {
            printf("[coordinator] range %lld: aborted - quit requested\n", (long long) claim.rangeId);
        } else if (lostOwnership) {
            printf("[coordinator] range %lld: aborted - no longer assigned to us%s\n", (long long) claim.rangeId,
                   wasPausedWhenStopped ? " (still paused)" : "");
        } else if (wasPausedWhenStopped) {
            printf("[coordinator] range %lld: aborted - claim released while paused\n", (long long) claim.rangeId);
        } else {
            printf("[coordinator] range %lld: aborted - target was already solved by someone else\n", (long long) claim.rangeId);
        }
        if (search.outbox && search.outbox->size() > 0)
            printf("[coordinator] range %lld: %zu basename(s) not sent - the server hasn't been told the search got past them, so "
                   "they'll be found again when that part of the range is searched again\n",
                   (long long) claim.rangeId, search.outbox->size());
        if (callbacks && callbacks->onRangeFinished)
            callbacks->onRangeFinished(false);
        return;
    }

    if (result.found) {
        printf("[coordinator] range %lld: MATCH FOUND: %s\n", (long long) claim.rangeId, result.filename.c_str());
    } else {
        printf("[coordinator] range %lld: exhausted, no match\n", (long long) claim.rangeId);
    }
    if (callbacks && callbacks->onRangeFinished)
        callbacks->onRangeFinished(result.found);

    CoordinatorClient completeClient(args.serverUrl);
    completeClient.setToken(token);
    // A completion says the whole range has been searched, so it can only
    // carry the last of its basenames: more than a report holds go first,
    // in heartbeats. A find is reported all the same - the server takes it
    // from anyone.
    if (search.outbox && !sendBasenamesBeforeCompleting(args, completeClient, claim, *search.outbox, quitRequested, callbacks) && !result.found)
        return;
    CompleteRequest completeReq;
    completeReq.found = result.found;
    if (result.found)
        completeReq.filename = result.filename;
    completeReq.elapsedSeconds = elapsedSeconds;
    completeReq.candidatesProcessed = claim.candidateCount;
    if (search.outbox)
        completeReq.basenames = search.outbox->peek(kMaxBasenamesPerReport).basenames;
    completeReq.falseMatches = search.falseMatches.peek();

    // Retried until it gets a real answer: a lost report either leaves the
    // range to wait out its whole lease before being redone, or - worse -
    // drops a found match on the floor. And there's no point moving on to
    // claim new work while the server can't be reached anyway. Deliberately
    // not gated on pauseRequested: reporting finished work isn't new work.
    RetryBackoff backoff(std::chrono::seconds(args.pollIntervalSecs));
    std::string err;
    CoordinatorClient::CompleteOutcome outcome;
    while ((outcome = completeClient.complete(claim.rangeId, completeReq, err)) == CoordinatorClient::CompleteOutcome::TransientError) {
        if (quitRequested && quitRequested->load(std::memory_order_relaxed)) {
            fprintf(stderr, "[coordinator] range %lld: quitting without having reported completion%s\n", (long long) claim.rangeId,
                    result.found ? (" - the match (" + result.filename + ") is only in " + foundFilePath(search.outputFilePath)).c_str() : "");
            return;
        }
        auto cap = backoff.cap();
        auto sleepFor = backoff.nextSleep();
        std::string msg = "[coordinator] range " + std::to_string(claim.rangeId) + ": failed to report completion, retrying in " +
                          std::to_string(sleepFor.count()) + "s (backoff cap " + std::to_string(cap.count()) + "s): " + err;
        fprintf(stderr, "%s\n", msg.c_str());
        if (callbacks && callbacks->onStatus)
            callbacks->onStatus(msg);
        interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(sleepFor), quitRequested);
    }
    switch (outcome) {
        case CoordinatorClient::CompleteOutcome::Ok:
            basenamesDelivered(search.outbox, completeReq.basenames);
            falseMatchesDelivered(&search.falseMatches, completeReq.falseMatches);
            break;
        case CoordinatorClient::CompleteOutcome::Conflict:
            basenamesDelivered(search.outbox, completeReq.basenames);
            falseMatchesDelivered(&search.falseMatches, completeReq.falseMatches);
            // This range's ownership moved on before we could report in -
            // almost certainly a network outage during heartbeating that
            // outlasted the lease, so the server already reassigned it.
            if (result.found) {
                fprintf(stderr,
                        "[coordinator] range %lld: found a match but lost ownership of this range before reporting it - "
                        "It has been reported now\n",
                        (long long) claim.rangeId);
            } else {
                printf("[coordinator] range %lld: lost ownership of this range before reporting completion - "
                       "the server had already reassigned it, so this GPU time was redundant but nothing is lost\n",
                       (long long) claim.rangeId);
            }
            break;
        case CoordinatorClient::CompleteOutcome::TransientError: // unreachable - retried above
        case CoordinatorClient::CompleteOutcome::Error:
            fprintf(stderr, "[coordinator] range %lld: failed to report completion: %s\n", (long long) claim.rangeId, err.c_str());
            break;
    }
}

// An alphabet target's range: runSearch over its candidates, its progress
// the latest Hash A match.
void runAlphabetRange(SearchBackend& backend, const CoordinatorArgs& args, const std::string& token, const ClaimResponse& claim,
                      const std::atomic<bool>* pauseRequested, std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    std::string reqError;
    SearchRequest req = toSearchRequest(claim, args.matchesDir, reqError);
    if (!reqError.empty()) {
        fprintf(stderr, "[coordinator] range %lld: %s - skipping, letting the lease expire so it gets reassigned\n",
                (long long) claim.rangeId, reqError.c_str());
        return;
    }

    if (callbacks && callbacks->onRangeClaimed)
        callbacks->onRangeClaimed(claim, req);

    std::mutex lastMatchMutex;
    std::optional<std::string> lastHashAMatch;
    RangeSearch search;
    search.outputFilePath = req.outputFilePath;
    search.run = [&](std::atomic<bool>& abort) {
        SearchResult found = runSearch(
            backend, req, &abort,
            [&](const std::string& filename) {
                std::lock_guard<std::mutex> lock(lastMatchMutex);
                lastHashAMatch = filename;
            },
            pauseRequested, [&](const FalseMatch& falseMatch) { search.falseMatches.add(falseMatch.filename); });
        return RangeResult{found.ok, found.error, found.found, found.filename, found.aborted};
    };
    search.progress = [&](std::optional<std::string>& last) {
        std::lock_guard<std::mutex> lock(lastMatchMutex);
        last = lastHashAMatch;
    };
    runRange(args, token, claim, search, pauseRequested, quitRequested, callbacks);
}

// The search of a dictionary target's range (see ClaimResponse::dictionary)
// - false, with `error` set, if the claim can't make one.
bool toDictionaryRequest(const ClaimResponse& claim, const std::vector<std::string>& words, const std::string& matchesDir, DictionaryRequest& req,
                         std::string& error) {
    req.pattern.words = words;
    req.pattern.separators.clear();
    for (const std::string& separator : claim.separators)
        req.pattern.separators.push_back(normalizeMpqName(separator));
    req.pattern.minWords = (int) claim.minWords;
    req.pattern.maxWords = (int) claim.maxWords;
    // The tails, made here as the server made them - and checked against
    // its checksum, rather than search other candidates than it numbered.
    if (!expandDictionaryTails(claim.tails, req.pattern.tails, error)) {
        error = "the claim's tails: " + error;
        return false;
    }
    if (!claim.tails.empty() && hex64(wordListChecksum(req.pattern.tails)) != claim.tailsChecksum) {
        error = "the claim's tails make other tails than the server's (checksum " + hex64(wordListChecksum(req.pattern.tails)) + ", not " +
                claim.tailsChecksum + ")";
        return false;
    }
    DictionarySpace space;
    if (claim.minWords < 1 || claim.maxWords > kMaxDictionaryWords || !DictionarySpace::create(req.pattern, space, error)) {
        if (error.empty())
            error = "the claim's word counts can't be searched";
        return false;
    }
    if ((uint64_t) claim.endCandidateNumber > space.size()) {
        error = "the claim's range ends at number " + std::to_string(claim.endCandidateNumber) + ", past the last candidate (" +
                std::to_string(space.size() - 1) + ")";
        return false;
    }
    req.prefix = normalizeMpqName(claim.prefix);
    req.suffix = normalizeMpqName(claim.suffix);
    req.bounds = FilenameBounds();
    if (claim.filenameLowerBound) {
        req.bounds.hasLower = true;
        req.bounds.lower = normalizeMpqName(*claim.filenameLowerBound);
    }
    if (claim.filenameUpperBound) {
        req.bounds.hasUpper = true;
        req.bounds.upper = normalizeMpqName(*claim.filenameUpperBound);
    }
    if (!hexToU32(claim.hashAHex, req.targetHashA) || !hexToU32(claim.hashBHex, req.targetHashB)) {
        error = "malformed target hash in claim response (hashA=" + claim.hashAHex + " hashB=" + claim.hashBHex + ")";
        return false;
    }
    // The key, whenever the target has one: only the candidates whose
    // basename matches it are compared to the hashes - and with
    // send_basenames, those basenames are sent.
    // It's also part of what a match of both hashes is checked against -
    // with the hash table position, if that's known too.
    if (!toMatchCheck(claim, req.matchCheck, error))
        return false;
    req.checkBasename = req.matchCheck.checkKey;
    req.basenameKey = req.matchCheck.key;
    req.recordBasenames = claim.sendBasenames;
    req.startNumber = (uint64_t) claim.firstCandidateNumber;
    req.endNumber = (uint64_t) claim.endCandidateNumber;
    req.wordSource.clear();
    for (const std::string& name : claim.wordLists)
        req.wordSource += (req.wordSource.empty() ? "" : " + ") + name;
    req.outputFilePath = matchesFilePath(matchesDir, claim.targetName);
    // The basenames go to the server, by way of the outbox (see
    // BasenameOutbox) - never to a file; the server keeps the progress.
    req.basenamesFilePath.clear();
    req.progressFilePath.clear();
    return true;
}

// A dictionary target's range: its words (downloaded if need be), and
// runDictionarySearch over its numbers, its progress the number it has got
// to, and the basenames it finds sent with every report - kept in memory
// until then (see BasenameOutbox). False if it can't be searched at all -
// the range is handed back, and the caller should wait a while before
// claiming again, rather than be handed the same one.
bool runDictionaryRange(DictionaryBackend& backends, const CoordinatorArgs& args, const std::string& token, const ClaimResponse& claim,
                        const std::atomic<bool>* pauseRequested, std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    std::string error;
    CoordinatorClient downloadClient(args.serverUrl);
    downloadClient.setToken(token);
    std::vector<std::string> words, downloaded;
    DictionaryRequest req;
    SearchBackend* backend = nullptr;
    const std::string cacheDir = (std::filesystem::path(args.matchesDir) / kWordListCacheDirName).string();
    const bool ready = resolveClaimWords(
                           claim, cacheDir,
                           [&](const std::string& name, std::string& text, std::string& downloadError) {
                               printf("[coordinator] downloading word list %s\n", name.c_str());
                               fflush(stdout);
                               return downloadClient.downloadWordList(name, text, downloadError);
                           },
                           words, downloaded, error) &&
                       toDictionaryRequest(claim, words, args.matchesDir, req, error) && (backend = backends.get(error)) != nullptr;
    if (!ready) {
        fprintf(stderr, "[coordinator] range %lld: can't search it (%s) - handing it back\n", (long long) claim.rangeId, error.c_str());
        reportQuit(args, token, claim.rangeId, QuitRequest(), nullptr, nullptr);
        return false;
    }
    for (const std::string& name : downloaded)
        printf("[coordinator] word list %s kept in %s\n", name.c_str(), cacheDir.c_str());

    if (callbacks && callbacks->onRangeClaimed) {
        // What a caller can show of it: no lower bound - the numbers aren't a
        // candidate's place in an alphabet, so how far it has got comes from
        // onDictionaryCount instead.
        SearchRequest shown;
        shown.prefix = req.prefix;
        shown.suffix = req.suffix;
        shown.outputFilePath = req.outputFilePath;
        callbacks->onRangeClaimed(claim, shown);
    }

    // The search hands on every basename found in a backend call before
    // saying it has got past the call - see BasenameOutbox.
    BasenameOutbox outbox(req.startNumber);
    RangeSearch search;
    search.outbox = &outbox;
    search.outputFilePath = req.outputFilePath;
    search.run = [&](std::atomic<bool>& abort) {
        DictionarySearchHooks hooks;
        hooks.onBasenameMatch = [&](const std::string& basename, const std::string&) {
            outbox.add(basename);
            if (callbacks && callbacks->onDictionaryBasename)
                callbacks->onDictionaryBasename(basename);
        };
        hooks.onProgress = [&](uint64_t next) { outbox.setProgress(next); };
        hooks.onFalseMatch = [&](const FalseMatch& falseMatch) { search.falseMatches.add(falseMatch.filename); };
        if (callbacks) {
            hooks.onCount = callbacks->onDictionaryCount;
            hooks.onStatus = callbacks->onDictionaryStatus;
        }
        DictionaryResult found = runDictionarySearch(*backend, req, &abort, hooks, pauseRequested);
        return RangeResult{found.ok, found.error, found.found, found.filename, found.aborted};
    };
    runRange(args, token, claim, search, pauseRequested, quitRequested, callbacks);
    return true;
}

// The search of a meet-in-the-middle target's range (see
// ClaimResponse::meetInTheMiddle) - false, with `error` set, if the claim
// can't make one. `base` is the node number its first tail is, which turns
// the walk's progress back into a tail index for a report.
bool toMeetInTheMiddleRequest(const ClaimResponse& claim, const CoordinatorArgs& args, MeetInTheMiddleRequest& req, uint64_t& base,
                              std::string& error) {
    req.alphabet = claim.alphabet;
    req.prefix = claim.prefix;
    req.suffix = claim.suffix;
    // One length a range: a tail is the directory without the table's
    // characters, so the tails of two lengths aren't the same thing.
    req.minLength = (int) claim.directoryLength;
    req.maxLength = (int) claim.directoryLength;
    req.tableLength = (int) claim.meetInTheMiddleTableLength;
    req.rules.symbolRuns = claim.pruneSymbolRuns;
    req.rules.unopenedBrackets = claim.pruneUnopenedBrackets;
    req.rules.maxBackslashCount = (int) claim.maxBackslashCount;
    req.rules.minBackslashCount = (int) claim.minBackslashCount;
    req.rules.adjacentBackslashes = claim.pruneAdjacentBackslashes;
    if (!hexToU32(claim.hashAHex, req.targetHashA) || !hexToU32(claim.hashBHex, req.targetHashB)) {
        error = "malformed target hash in claim response (hashA=" + claim.hashAHex + " hashB=" + claim.hashBHex + ")";
        return false;
    }
    uint32_t seeds[4] = {0, 0, 0, 0};
    const std::string hex[4] = {claim.meetingPointASeed1Hex, claim.meetingPointASeed2Hex, claim.meetingPointBSeed1Hex, claim.meetingPointBSeed2Hex};
    for (int i = 0; i < 4; ++i) {
        if (!hexToU32(hex[i], seeds[i])) {
            error = "malformed meeting point in claim response (" + hex[i] + ")";
            return false;
        }
    }
    req.meetingPointA = {seeds[0], seeds[1]};
    req.meetingPointB = {seeds[2], seeds[3]};
    if (!toMatchCheck(claim, req.matchCheck, error))
        return false;
    // The memory this range's table would take - never more than this client
    // said it would give (RegisterRequest::meetInTheMiddleTableBytes). The
    // server doesn't hand out more, so this only catches a server that got it
    // wrong: a table is the volunteer's memory, not the server's to spend.
    const uint64_t tableBytes = MeetInTheMiddleTable::bytesFor((int) req.alphabet.size(), req.tableLength);
    if (tableBytes > (uint64_t) args.meetInTheMiddleTableBytes) {
        error = "its table of " + std::to_string(req.tableLength) + " characters needs " + std::to_string(tableBytes / (1024 * 1024)) +
                " MB, more than the " + std::to_string((uint64_t) args.meetInTheMiddleTableBytes / (1024 * 1024)) +
                " MB meet_in_the_middle_table_memory allows";
        return false;
    }
    // Where the claim's tails are among the ones a search of this length
    // walks - see MeetInTheMiddleSpace.
    MeetInTheMiddleSpace space;
    if (!MeetInTheMiddleSpace::create(req.alphabet, req.minLength, req.maxLength, req.tableLength, space, error))
        return false;
    const int tailLength = space.maxTailLength();
    if ((uint64_t) claim.endTailIndex > space.countOf(tailLength)) {
        error = "the claim's range ends at tail " + std::to_string(claim.endTailIndex) + ", past the last one (" +
                std::to_string(space.countOf(tailLength) - 1) + ")";
        return false;
    }
    base = space.numberOf(tailLength, 0);
    req.startNumber = space.numberOf(tailLength, (uint64_t) claim.firstTailIndex);
    req.endNumber = space.numberOf(tailLength, (uint64_t) claim.endTailIndex);
    req.outputFilePath = matchesFilePath(args.matchesDir, claim.targetName);
    // The server keeps the progress (a heartbeat's next_tail_index), so
    // there's no progress file.
    req.progressFilePath.clear();
    return true;
}

// A meet-in-the-middle target's range: runMeetInTheMiddleSearch over the
// tails the claim names, its progress the tail it has got to. False if it
// can't be searched at all - the range is handed back, and the caller should
// wait a while before claiming again, rather than be handed the same one.
bool runMeetInTheMiddleRange(MeetInTheMiddleBackend& backends, const CoordinatorArgs& args, const std::string& token, const ClaimResponse& claim,
                      const std::atomic<bool>* pauseRequested, std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    std::string error;
    MeetInTheMiddleRequest req;
    uint64_t base = 0;
    SearchBackend* backend = nullptr;
    if (!toMeetInTheMiddleRequest(claim, args, req, base, error) || (backend = backends.get(error)) == nullptr) {
        fprintf(stderr, "[coordinator] range %lld: can't search it (%s) - handing it back\n", (long long) claim.rangeId, error.c_str());
        reportQuit(args, token, claim.rangeId, QuitRequest(), nullptr, nullptr);
        return false;
    }

    if (callbacks && callbacks->onRangeClaimed) {
        // What a caller can show of it: the directories it covers, with the
        // table's characters as '?' - the claim's own bounds.
        SearchRequest shown;
        shown.prefix = req.prefix;
        shown.suffix = req.suffix;
        shown.alphabet = req.alphabet;
        shown.lowerBound = claim.lowerBoundFilename;
        shown.upperBound = claim.upperBoundFilename;
        shown.outputFilePath = req.outputFilePath;
        callbacks->onRangeClaimed(claim, shown);
    }

    std::mutex progressMutex;
    std::optional<int64_t> reached;
    RangeSearch search;
    search.outputFilePath = req.outputFilePath;
    search.nextTailIndex = [&]() {
        std::lock_guard<std::mutex> lock(progressMutex);
        return reached;
    };
    search.run = [&](std::atomic<bool>& abort) {
        MeetInTheMiddleSearchHooks hooks;
        hooks.onProgress = [&](uint64_t next) {
            // The walk's node numbers are this length's tails, offset by
            // where they start (see toMeetInTheMiddleRequest).
            std::lock_guard<std::mutex> lock(progressMutex);
            reached = (int64_t) (next - base);
        };
        hooks.onFalseMatch = [&](const FalseMatch& falseMatch) { search.falseMatches.add(falseMatch.filename); };
        if (callbacks) {
            hooks.onCount = callbacks->onMeetInTheMiddleCount;
            hooks.onStatus = callbacks->onMeetInTheMiddleStatus;
            hooks.onPartialMatch = callbacks->onMeetInTheMiddleMatch;
        }
        MeetInTheMiddleResult found = runMeetInTheMiddleSearch(*backend, req, &abort, hooks, pauseRequested);
        return RangeResult{found.ok, found.error, found.found, found.filename, found.aborted};
    };
    runRange(args, token, claim, search, pauseRequested, quitRequested, callbacks);
    return true;
}

// Runs exactly one claimed range, of whichever kind of target. False if it
// couldn't be searched at all, so the caller should wait a while before
// claiming again (see runDictionaryRange).
bool runOneRange(SearchBackend& backend, DictionaryBackend& dictionaryBackend, MeetInTheMiddleBackend& meetInTheMiddleBackend,
                 const CoordinatorArgs& args, const std::string& token, const ClaimResponse& claim, const std::atomic<bool>* pauseRequested,
                 std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    printf("[coordinator] starting range %lld (target %s) [%s .. %s]\n",
           (long long) claim.rangeId, claim.targetName.c_str(), claim.lowerBoundFilename.c_str(), claim.upperBoundFilename.c_str());
    if (claim.dictionary)
        return runDictionaryRange(dictionaryBackend, args, token, claim, pauseRequested, quitRequested, callbacks);
    if (claim.meetInTheMiddle)
        return runMeetInTheMiddleRange(meetInTheMiddleBackend, args, token, claim, pauseRequested, quitRequested, callbacks);
    runAlphabetRange(backend, args, token, claim, pauseRequested, quitRequested, callbacks);
    return true;
}

} // namespace

bool buildCoordinatorArgs(const ConfigFile& config, CoordinatorArgs& out, std::string& error) {
    ConfigSectionReader r(config.coordinator);
    if (!r.getRequired("server_url", out.serverUrl, error))
        return false;
    // Left "" if absent - resolveMissingIdentity (called from runCoordinator)
    // auto-detects and interactively confirms/persists a value for either.
    out.username = r.getOptional("username", "");
    out.hostname = r.getOptional("hostname", "");
    std::string pollStr = r.getOptional("poll_interval_secs", "30");
    const std::string meetInTheMiddleMemoryStr = r.getOptional("meet_in_the_middle_table_memory", "0");

    std::string unknown = r.firstUnknownKey();
    if (!unknown.empty()) {
        error = "unknown key '" + unknown + "' in [coordinator] section";
        return false;
    }

    try {
        out.pollIntervalSecs = std::stoi(pollStr);
    } catch (const std::exception&) {
        error = "invalid poll_interval_secs: " + pollStr;
        return false;
    }
    uint64_t meetInTheMiddleBytes = 0;
    if (!parseConfigBytes(meetInTheMiddleMemoryStr, meetInTheMiddleBytes) || meetInTheMiddleBytes > (uint64_t) INT64_MAX) {
        error = "invalid meet_in_the_middle_table_memory: '" + meetInTheMiddleMemoryStr +
                "' (expected a number of bytes, or one with a K, M or G after it)";
        return false;
    }
    out.meetInTheMiddleTableBytes = (int64_t) meetInTheMiddleBytes;
    out.matchesDir = config.matchesDir;
    out.backend = config.backend;
    return true;
}

int runCoordinator(CoordinatorArgs args, std::atomic<bool>* pauseRequested, std::atomic<bool>* quitRequested,
                    const CoordinatorCallbacks* callbacks, std::atomic<bool>* finishRangeThenPause) {
    resolveMissingIdentity(args);

    // Not thread-safe to call lazily once the heartbeat thread may already
    // be making requests concurrently with the main thread - do it once,
    // up front, before any thread touches libcurl. Paired with
    // curl_global_cleanup() via this guard's destructor on every return path.
    CurlGlobalGuard curlGuard;

    // One backend for every range this client claims - runSearch() resets
    // its state at the start of each. Created before registering, which
    // tells the server which backend this client uses.
    std::string backendError;
    std::unique_ptr<SearchBackend> backend = createBackend(args.backend, backendError);
    if (!backend) {
        fprintf(stderr, "%s\n", backendError.c_str());
        if (callbacks && callbacks->onStatus)
            callbacks->onStatus(backendError);
        return 1;
    }

    CoordinatorClient client(args.serverUrl);
    RegisterResponse registration;
    std::string error;
    if (!client.registerClient(args.username, args.hostname, backend->name(), args.meetInTheMiddleTableBytes, registration, error)) {
        // Too old for the server (see CoordinatorClient::upgradeRequired):
        // its message says what to get.
        std::string msg = client.upgradeRequired() ? error : "failed to register with coordinator: " + error;
        fprintf(stderr, "%s\n", msg.c_str());
        if (callbacks && callbacks->onStatus)
            callbacks->onStatus(msg);
        return 1;
    }
    {
        // What this client will give a meet-in-the-middle target's table
        // decides whether it's offered that work at all, so it's said out
        // loud.
        std::string tableNote = "no meet-in-the-middle work (set meet_in_the_middle_table_memory in [coordinator] to take it on)";
        if (args.meetInTheMiddleTableBytes > 0)
            tableNote = std::to_string((uint64_t) args.meetInTheMiddleTableBytes / (1024 * 1024)) + " MB for a meet-in-the-middle table";
        std::string msg = "[coordinator] registered with coordinator as user " + std::to_string(registration.userId) + " (" + args.username +
                           "@" + args.hostname + ", " + backend->name() + " backend, " + tableNote + ") - namebreak " + namebreakVersion() +
                           ", client protocol v" + kProtocolVersion + ", server protocol v" + registration.serverProtocolVersion;
        printf("%s\n", msg.c_str());
        fflush(stdout);
        if (callbacks && callbacks->onStatus)
            callbacks->onStatus(msg);
    }

    // Dictionary and meet-in-the-middle ranges' backends, each chosen when
    // the first one of its kind comes.
    DictionaryBackend dictionaryBackend(*backend);
    MeetInTheMiddleBackend meetInTheMiddleBackend(*backend);

    auto pollInterval = std::chrono::seconds(args.pollIntervalSecs);
    RetryBackoff claimBackoff(pollInterval);

    while (true) {
        // Never claim new work while paused - "pause the client" means
        // exactly that, not just "pause whichever range is currently in
        // hand". This is also what makes staying paused actually stick after
        // a claim-expired-while-paused release (see runOneRange above): that
        // release doesn't touch pauseRequested itself, it just stops the
        // now-invalid range's search and heartbeat - it's this gate, right
        // here, that keeps the loop from immediately claiming a replacement.
        //
        // It's also where finishRangeThenPause turns into a real pause: this
        // is the one point between finishing a range and claiming the next.
        // Checked on every pass, not just once, so that turning it on while
        // paused here (which also resumes - see its callers) pauses again
        // straight away rather than claiming a whole new range.
        while (true) {
            if (finishRangeThenPause && pauseRequested && finishRangeThenPause->exchange(false, std::memory_order_relaxed)) {
                pauseRequested->store(true, std::memory_order_relaxed);
                // The key hint is for the console only - the GUI (onStatus)
                // resumes with its Resume button instead.
                const char* msg = "[coordinator] paused before claiming new work, as asked";
                printf("%s. Resume by pressing 'p'\n", msg);
                if (callbacks && callbacks->onStatus)
                    callbacks->onStatus(msg);
            }
            if (!(pauseRequested && pauseRequested->load(std::memory_order_relaxed)))
                break;
            if (quitRequested && quitRequested->load(std::memory_order_relaxed))
                return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (quitRequested && quitRequested->load(std::memory_order_relaxed))
            return 0;

        std::optional<ClaimResponse> claim;
        if (!client.claim(claim, error)) {
            if (client.upgradeRequired()) {
                // The server has stopped accepting this release since it
                // registered - retrying won't change that.
                fprintf(stderr, "%s\n", error.c_str());
                if (callbacks && callbacks->onStatus)
                    callbacks->onStatus(error);
                return 1;
            }
            auto cap = claimBackoff.cap();
            auto sleepFor = claimBackoff.nextSleep();
            std::string msg = "[coordinator] claim failed, retrying in " + std::to_string(sleepFor.count()) + "s (backoff cap " +
                               std::to_string(cap.count()) + "s): " + error;
            fprintf(stderr, "%s\n", msg.c_str());
            if (callbacks && callbacks->onStatus)
                callbacks->onStatus(msg);
            interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(sleepFor), quitRequested);
            continue;
        }
        claimBackoff.reset(); // once the server is reachable again

        if (!claim) {
            printf("[coordinator] no work available, sleeping\n");
            fflush(stdout);
            if (callbacks && callbacks->onStatus)
                callbacks->onStatus("[coordinator] no work available, sleeping");
            interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(pollInterval), quitRequested);
            continue;
        }

        if (!runOneRange(*backend, dictionaryBackend, meetInTheMiddleBackend, args, client.token(), *claim, pauseRequested, quitRequested,
                         callbacks)) {
            // Most likely the next claim would be the same range again.
            interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(pollInterval), quitRequested);
        }
        if (quitRequested && quitRequested->load(std::memory_order_relaxed))
            return 0;
    }
}
