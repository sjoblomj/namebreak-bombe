#include "net/coordinator_runner.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
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
#include "common/config.h"
#include "common/matches_file.h"
#include "net/coordinator_client.h"
#include "common/string_util.h"
#include "engine/candidate.h"
#include "common/platform.h"
#include "engine/search.h"

namespace {

// Fixed rather than derived from a range's lease, so progress checkpoints
// (and the liveness signal the server's reclaim sweep relies on) land at a
// steady, predictable cadence regardless of how big a range is or how fast a
// client is.
constexpr int kHeartbeatIntervalSeconds = 60;

// Ceiling for the exponential backoff below, so a prolonged outage doesn't
// leave a client waiting arbitrarily long once the server comes back.
constexpr std::chrono::seconds kMaxClaimBackoff{600};

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

SearchRequest toSearchRequest(const ClaimResponse& claim, const std::string& matchesDir, std::string& error) {
    SearchRequest req;
    req.alphabet = claim.alphabet;
    req.maxBackslashCount = (int) claim.maxBackslashCount;
    req.prefix = claim.prefix;
    req.suffix = claim.suffix;
    req.lowerBound = removePrefixAndSuffix(claim.lowerBoundFilename, req.prefix, req.suffix);
    req.upperBound = removePrefixAndSuffix(claim.upperBoundFilename, req.prefix, req.suffix);
    // A claimed range always starts exactly at its own lower bound (unlike
    // local "continuous" mode's arbitrary mid-space resume point).
    req.startCandidate = req.lowerBound;
    req.pruneSymbolRuns = claim.pruneSymbolRuns;
    req.pruneUnopenedBrackets = claim.pruneUnopenedBrackets;
    req.continuous = false; // a coordinator range is always run "bounded"
    req.outputFilePath = matchesFilePath(matchesDir, claim.targetName);
    if (!hexToU32(claim.hashAHex, req.targetHashA) || !hexToU32(claim.hashBHex, req.targetHashB)) {
        error = "malformed target hash in claim response (hashA=" + claim.hashAHex + " hashB=" + claim.hashBHex + ")";
    }
    return req;
}

// Runs exactly one claimed range: spawns the heartbeat thread, runs the
// search in-process on the calling thread, then reports completion.
// `quitRequested`/`callbacks`, if given, are as documented on runCoordinator
// (coordinator_runner.h) - both default null for the CLI's own call site
// below, so this function's added behavior is opt-in.
void runOneRange(SearchBackend& backend, const CoordinatorArgs& args, const std::string& token, const ClaimResponse& claim,
                 const std::atomic<bool>* pauseRequested, std::atomic<bool>* quitRequested, const CoordinatorCallbacks* callbacks) {
    printf("[coordinator] starting range %lld (target %s) [%s .. %s]\n",
           (long long) claim.rangeId, claim.targetName.c_str(), claim.lowerBoundFilename.c_str(), claim.upperBoundFilename.c_str());

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
    std::string lastHashAMatch;
    bool haveLastHashAMatch = false;
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
    // server-signaled release, so runSearch() needs no separate awareness of
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
        while (!stopCv.wait_for(lock, std::chrono::seconds(kHeartbeatIntervalSeconds), [&] { return stopRequested; })) {
            std::optional<std::string> latest;
            {
                std::lock_guard<std::mutex> mlock(lastMatchMutex);
                if (haveLastHashAMatch)
                    latest = lastHashAMatch;
            }
            lock.unlock();
            HeartbeatResponse resp;
            std::string err;
            switch (hbClient.heartbeat(claim.rangeId, latest, resp, err)) {
                case CoordinatorClient::HeartbeatOutcome::Ok:
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
                    fprintf(stderr, "[coordinator] range %lld: heartbeat failed: %s\n", (long long) claim.rangeId, err.c_str());
                    break;
            }
            lock.lock();
        }
    stopHeartbeating:;
    });

    auto started = std::chrono::steady_clock::now();
    SearchResult result = runSearch(backend, req, &abortRequested, [&](const std::string& filename) {
        std::lock_guard<std::mutex> mlock(lastMatchMutex);
        lastHashAMatch = filename;
        haveLastHashAMatch = true;
    }, pauseRequested);
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
        // reaches the same abortRequested flag runSearch() polls.
        if (externallyQuit) {
            printf("[coordinator] range %lld: aborted - quit requested\n", (long long) claim.rangeId);
        } else if (lostOwnership) {
            printf("[coordinator] range %lld: aborted - no longer assigned to us%s\n", (long long) claim.rangeId,
                   wasPausedWhenStopped ? " (still paused)" : "");
        } else if (wasPausedWhenStopped) {
            printf("[coordinator] range %lld: aborted - claim released while paused\n", (long long) claim.rangeId);
        } else {
            printf("[coordinator] range %lld: aborted - target was already solved by someone else\n", (long long) claim.rangeId);
        }
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
    CompleteRequest completeReq;
    completeReq.found = result.found;
    if (result.found)
        completeReq.filename = result.filename;
    completeReq.elapsedSeconds = elapsedSeconds;
    completeReq.candidatesProcessed = claim.candidateCount;

    std::string err;
    auto outcome = completeClient.complete(claim.rangeId, completeReq, err);
    switch (outcome) {
        case CoordinatorClient::CompleteOutcome::Ok:
            break;
        case CoordinatorClient::CompleteOutcome::Conflict:
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
        case CoordinatorClient::CompleteOutcome::Error:
            fprintf(stderr, "[coordinator] range %lld: failed to report completion: %s\n", (long long) claim.rangeId, err.c_str());
            break;
    }
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
    int64_t userId = 0;
    std::string serverProtocolVersion;
    std::string error;
    if (!client.registerClient(args.username, args.hostname, backend->name(), userId, serverProtocolVersion, error)) {
        std::string msg = "failed to register with coordinator: " + error;
        fprintf(stderr, "%s\n", msg.c_str());
        if (callbacks && callbacks->onStatus)
            callbacks->onStatus(msg);
        return 1;
    }
    {
        std::string msg = "[coordinator] registered with coordinator as user " + std::to_string(userId) + " (" + args.username + "@" +
                           args.hostname + ", " + backend->name() + " backend) - client protocol v" + kProtocolVersion + ", server protocol v" +
                           serverProtocolVersion;
        printf("%s\n", msg.c_str());
        if (callbacks && callbacks->onStatus)
            callbacks->onStatus(msg);
    }

    auto pollInterval = std::chrono::seconds(args.pollIntervalSecs);
    auto claimBackoff = pollInterval;
    std::mt19937 rng(std::random_device{}());

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
                const char* msg = "[coordinator] paused before claiming new work, as asked";
                printf("%s\n", msg);
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
            // Exponential backoff, capped at kMaxClaimBackoff, so a real
            // outage doesn't get hammered at the normal poll rate. Jittered
            // (sleep a random fraction of the current backoff rather than
            // its full length) so a fleet of clients that all started
            // failing at once - e.g. the coordinator itself going down -
            // doesn't retry in lockstep once it recovers.
            std::uniform_int_distribution<long long> jitter(0, claimBackoff.count());
            auto sleepFor = std::chrono::seconds(jitter(rng));
            std::string msg = "[coordinator] claim failed, retrying in " + std::to_string(sleepFor.count()) + "s (backoff cap " +
                               std::to_string(claimBackoff.count()) + "s): " + error;
            fprintf(stderr, "%s\n", msg.c_str());
            if (callbacks && callbacks->onStatus)
                callbacks->onStatus(msg);
            interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(sleepFor), quitRequested);
            claimBackoff = std::min(claimBackoff * 2, kMaxClaimBackoff);
            continue;
        }
        claimBackoff = pollInterval; // reset once the server is reachable again

        if (!claim) {
            printf("[coordinator] no work available, sleeping\n");
            if (callbacks && callbacks->onStatus)
                callbacks->onStatus("[coordinator] no work available, sleeping");
            interruptibleSleep(std::chrono::duration_cast<std::chrono::milliseconds>(pollInterval), quitRequested);
            continue;
        }

        runOneRange(*backend, args, client.token(), *claim, pauseRequested, quitRequested, callbacks);
        if (quitRequested && quitRequested->load(std::memory_order_relaxed))
            return 0;
    }
}
