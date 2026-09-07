#include "coordinator_runner.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>

#include <curl/curl.h>

#include "coordinator_client.h"
#include "cpu-utils.h"
#include "search.h"

namespace {

// Fixed rather than derived from a range's lease, so progress checkpoints
// (and the liveness signal the server's reclaim sweep relies on) land at a
// steady, predictable cadence regardless of how big a range is or how fast a
// client is - matches the old Rust client's HEARTBEAT_INTERVAL.
constexpr int kHeartbeatIntervalSeconds = 60;

struct CoordinatorArgs {
    std::string serverUrl;
    std::string username;
    std::string hostname;
    int pollIntervalSecs = 30;
};

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : fallback;
}

std::string resolveHostname(const std::string& given) {
    if (!given.empty()) return given;
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) {
        buf[sizeof(buf) - 1] = '\0';
        if (buf[0] != '\0') return std::string(buf);
    }
    return "unknown-host";
}

bool parseArgs(int argc, char** argv, CoordinatorArgs& out, std::string& error) {
    out.serverUrl = envOr("NAMEBREAK_SERVER_URL", "");
    out.username = envOr("NAMEBREAK_USERNAME", "");
    out.hostname = envOr("NAMEBREAK_HOSTNAME", "");

    // argv[0]=binary, argv[1]="coordinator", flags start at argv[2].
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextValue = [&](const char* flagName) -> std::string {
            if (i + 1 >= argc) {
                error = std::string(flagName) + " requires a value";
                return "";
            }
            return argv[++i];
        };
        if (arg == "--server-url") {
            out.serverUrl = nextValue("--server-url");
        } else if (arg == "--username") {
            out.username = nextValue("--username");
        } else if (arg == "--hostname") {
            out.hostname = nextValue("--hostname");
        } else if (arg == "--poll-interval-secs") {
            std::string v = nextValue("--poll-interval-secs");
            if (!error.empty()) return false;
            try {
                out.pollIntervalSecs = std::stoi(v);
            } catch (const std::exception&) {
                error = "invalid --poll-interval-secs: " + v;
                return false;
            }
        } else {
            error = "unknown argument: " + arg;
            return false;
        }
        if (!error.empty()) return false;
    }

    if (out.serverUrl.empty()) { error = "--server-url (or NAMEBREAK_SERVER_URL) is required"; return false; }
    if (out.username.empty()) { error = "--username (or NAMEBREAK_USERNAME) is required"; return false; }
    out.hostname = resolveHostname(out.hostname);
    return true;
}

// hexToU32 mirrors main()'s CLI hash parsing (std::stoul(..., 16), which
// tolerates an optional "0x"/"0X" prefix) - the server sends hashes as
// "0x{:08X}" strings (see ClaimResponse::hashAHex/hashBHex).
bool hexToU32(const std::string& s, uint32_t& out) {
    try {
        out = (uint32_t) std::stoul(s, nullptr, 16);
        return true;
    } catch (const std::exception&) {
        return false;
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

int runCoordinator(int argc, char** argv) {
    CoordinatorArgs args;
    std::string error;
    if (!parseArgs(argc, argv, args, error)) {
        fprintf(stderr, "%s\n", error.c_str());
        fprintf(stderr,
                "Usage: %s coordinator --server-url <url> --username <name> [--hostname <name>] [--poll-interval-secs <n>]\n"
                "(--server-url/--username may also come from NAMEBREAK_SERVER_URL/NAMEBREAK_USERNAME)\n",
                argv[0]);
        return 1;
    }

    // Not thread-safe to call lazily once the heartbeat thread may already
    // be making requests concurrently with the main thread - do it once,
    // up front, before any thread touches libcurl.
    curl_global_init(CURL_GLOBAL_DEFAULT);

    CoordinatorClient client(args.serverUrl);
    int64_t userId = 0;
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
