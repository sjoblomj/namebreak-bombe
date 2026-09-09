#include "coordinator_runner.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>

#include "config.h"
#include "coordinator_client.h"
#include "cpu-utils.h"
#include "platform.h"
#include "search.h"

namespace {

// Fixed rather than derived from a range's lease, so progress checkpoints
// (and the liveness signal the server's reclaim sweep relies on) land at a
// steady, predictable cadence regardless of how big a range is or how fast a
// client is - matches the old Rust client's HEARTBEAT_INTERVAL.
constexpr int kHeartbeatIntervalSeconds = 60;

std::string trimLine(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Interactively confirms (or lets the user override) auto-discovered values
// for whichever of `args`'s username/hostname were left unset by
// config.conf, then persists the final values back into the file - so this
// only ever prompts once per machine (future runs find both keys already
// set there). Falls back to silently using the discovered defaults with no
// prompt/write when stdin isn't a terminal (e.g. a cron job or systemd
// service), so a missing config never hangs a non-interactive run.
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
                    f.key.c_str(), f.value->c_str(), kConfigPath);
        }
        return;
    }

    printf("No %s configured in %s. Detected:\n", missing.size() == 2 ? "username/hostname" : missing[0].key.c_str(), kConfigPath);
    for (const Field& f : missing) printf("  %s = %s\n", f.key.c_str(), f.value->c_str());
    printf("Use %s? [Y/n]: ", missing.size() == 2 ? "these" : "this");
    fflush(stdout);

    std::string line;
    std::getline(std::cin, line);
    line = trimLine(line);
    bool accepted = line.empty() || line == "y" || line == "Y" || line == "yes" || line == "Yes";

    for (Field& f : missing) {
        if (!accepted) {
            printf("Enter %s [%s]: ", f.key.c_str(), f.value->c_str());
            fflush(stdout);
            std::string entered;
            std::getline(std::cin, entered);
            entered = trimLine(entered);
            if (!entered.empty()) *f.value = entered;
        }
        if (appendKeyToConfigSection(kConfigPath, "coordinator", f.key, *f.value)) {
            printf("[coordinator] saved %s = %s to %s\n", f.key.c_str(), f.value->c_str(), kConfigPath);
        } else {
            fprintf(stderr, "[coordinator] warning: could not save %s to %s - you'll be asked again next time\n", f.key.c_str(), kConfigPath);
        }
    }
}

SearchRequest toSearchRequest(const ClaimResponse& claim, std::string& error) {
    SearchRequest req;
    req.alphabet = claim.alphabet;
    req.maxBackslashCount = (int) claim.maxBackslashCount;
    req.prefix = claim.prefix;
    req.suffix = claim.suffix;
    req.lowerBound = remove_prefix_and_suffix(claim.lowerBoundFilename, req.prefix, req.suffix);
    req.upperBound = remove_prefix_and_suffix(claim.upperBoundFilename, req.prefix, req.suffix);
    // A claimed range always starts exactly at its own lower bound (unlike
    // local "continuous" mode's arbitrary mid-space resume point).
    req.startCandidate = req.lowerBound;
    req.pruneSymbolRuns = claim.pruneSymbolRuns;
    req.continuous = false; // a coordinator range is always run "bounded"
    if (!hexToU32(claim.hashAHex, req.targetHashA) || !hexToU32(claim.hashBHex, req.targetHashB)) {
        error = "malformed target hash in claim response (hashA=" + claim.hashAHex + " hashB=" + claim.hashBHex + ")";
    }
    return req;
}

// Runs exactly one claimed range: spawns the heartbeat thread, runs the
// search in-process on the calling thread, then reports completion. Mirrors
// coordinator/client/src/main.rs's run_one.
void runOneRange(const std::string& serverUrl, const std::string& token, const ClaimResponse& claim) {
    printf("[coordinator] starting range %lld (target %s) [%s .. %s]\n",
           (long long) claim.rangeId, claim.targetName.c_str(), claim.lowerBoundFilename.c_str(), claim.upperBoundFilename.c_str());

    std::string reqError;
    SearchRequest req = toSearchRequest(claim, reqError);
    if (!reqError.empty()) {
        fprintf(stderr, "[coordinator] range %lld: %s - skipping, letting the lease expire so it gets reassigned\n",
                (long long) claim.rangeId, reqError.c_str());
        return;
    }

    std::mutex lastMatchMutex;
    std::string lastHashAMatch;
    bool haveLastHashAMatch = false;
    std::atomic<bool> abortRequested{false};

    std::mutex stopMutex;
    std::condition_variable stopCv;
    bool stopRequested = false;

    std::thread heartbeatThread([&]() {
        CoordinatorClient hbClient(serverUrl);
        hbClient.setToken(token);
        std::unique_lock<std::mutex> lock(stopMutex);
        while (!stopCv.wait_for(lock, std::chrono::seconds(kHeartbeatIntervalSeconds), [&] { return stopRequested; })) {
            std::optional<std::string> latest;
            {
                std::lock_guard<std::mutex> mlock(lastMatchMutex);
                if (haveLastHashAMatch) latest = lastHashAMatch;
            }
            lock.unlock();
            HeartbeatResponse resp;
            std::string err;
            if (hbClient.heartbeat(claim.rangeId, latest, resp, err)) {
                if (resp.targetSolved) {
                    printf("[coordinator] range %lld: target already solved elsewhere - signaling abort\n", (long long) claim.rangeId);
                    abortRequested.store(true, std::memory_order_relaxed);
                    lock.lock();
                    break;
                }
            } else {
                fprintf(stderr, "[coordinator] range %lld: heartbeat failed: %s\n", (long long) claim.rangeId, err.c_str());
            }
            lock.lock();
        }
    });

    auto started = std::chrono::steady_clock::now();
    SearchResult result = runSearch(req, &abortRequested, [&](const std::string& filename) {
        std::lock_guard<std::mutex> mlock(lastMatchMutex);
        lastHashAMatch = filename;
        haveLastHashAMatch = true;
    });
    double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    {
        std::lock_guard<std::mutex> lock(stopMutex);
        stopRequested = true;
    }
    stopCv.notify_all();
    heartbeatThread.join();

    if (!result.ok) {
        fprintf(stderr, "[coordinator] range %lld: search failed (%s) - letting the lease expire so it gets reassigned\n",
                (long long) claim.rangeId, result.error.c_str());
        return;
    }

    if (result.aborted) {
        printf("[coordinator] range %lld: aborted - target was already solved by someone else\n", (long long) claim.rangeId);
        return;
    }

    if (result.found) {
        printf("[coordinator] range %lld: MATCH FOUND: %s\n", (long long) claim.rangeId, result.filename.c_str());
    } else {
        printf("[coordinator] range %lld: exhausted, no match\n", (long long) claim.rangeId);
    }

    CoordinatorClient completeClient(serverUrl);
    completeClient.setToken(token);
    CompleteRequest completeReq;
    completeReq.found = result.found;
    if (result.found) completeReq.filename = result.filename;
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
                        "the match is still recorded locally in matches.txt, but the server was never told about it; "
                        "check matches.txt manually\n",
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

bool buildCoordinatorArgs(const std::map<std::string, std::string>& section, CoordinatorArgs& out, std::string& error) {
    ConfigSectionReader r(section);
    if (!r.getRequired("server_url", out.serverUrl, error)) return false;
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
    return true;
}

int runCoordinator(CoordinatorArgs args) {
    resolveMissingIdentity(args);

    // Not thread-safe to call lazily once the heartbeat thread may already
    // be making requests concurrently with the main thread - do it once,
    // up front, before any thread touches libcurl.
    curl_global_init(CURL_GLOBAL_DEFAULT);

    CoordinatorClient client(args.serverUrl);
    int64_t userId = 0;
    std::string error;
    if (!client.registerClient(args.username, args.hostname, userId, error)) {
        fprintf(stderr, "failed to register with coordinator: %s\n", error.c_str());
        return 1;
    }
    printf("[coordinator] registered with coordinator as user %lld (%s@%s)\n", (long long) userId, args.username.c_str(), args.hostname.c_str());

    auto pollInterval = std::chrono::seconds(args.pollIntervalSecs);
    while (true) {
        std::optional<ClaimResponse> claim;
        if (!client.claim(claim, error)) {
            fprintf(stderr, "[coordinator] claim failed, retrying after backoff: %s\n", error.c_str());
            std::this_thread::sleep_for(pollInterval);
            continue;
        }
        if (!claim) {
            printf("[coordinator] no work available, sleeping\n");
            std::this_thread::sleep_for(pollInterval);
            continue;
        }

        runOneRange(args.serverUrl, client.token(), *claim);
    }
}
