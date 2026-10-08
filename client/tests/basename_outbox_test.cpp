// BasenameOutbox (net/basename_outbox.h): what a dictionary range's reports
// to the coordinator say - the basenames found that the server hasn't got,
// and how far the search has got, never past a basename a report leaves
// waiting. Including while the search adds on one thread and the heartbeats
// send from another. No network access, no files.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "net/basename_outbox.h"

static int g_failures = 0;
static int g_checks = 0;

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) {
        printf("  ok: %s\n", what.c_str());
    } else {
        fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

using Names = std::vector<std::string>;

static void testReports() {
    printf("--- reports ---\n");
    BasenameOutbox outbox(1000);
    BasenameOutbox::Report r = outbox.peek(2);
    check(r.basenames.empty() && r.next == 1000, "nothing yet: no basenames, and nothing searched past the start");

    // A backend call [1000, 1500) finds A and B; then [1500, 2000) finds C
    // and D; then [2000, 2600) nothing.
    outbox.add("A.WAV");
    outbox.add("B.WAV");
    outbox.setProgress(1500);
    outbox.add("C.WAV");
    outbox.add("D.WAV");
    outbox.setProgress(2000);
    outbox.setProgress(2600);
    r = outbox.peek(10);
    check(r.basenames == Names{"A.WAV", "B.WAV", "C.WAV", "D.WAV"} && r.next == 2600, "all of them fit: all the progress");
    r = outbox.peek(3);
    check(r.basenames == Names{"A.WAV", "B.WAV", "C.WAV"} && r.next == 1500,
          "D left out: no further than where it was found - the start of its call");
    r = outbox.peek(1);
    check(r.basenames == Names{"A.WAV"} && r.next == 1000, "B left out, found in the first call: nothing searched at all");

    // A report of the first three got through.
    outbox.remove(3);
    r = outbox.peek(1);
    check(r.basenames == Names{"D.WAV"} && r.next == 2600 && outbox.size() == 1, "the rest: D, and all the progress");
    outbox.add("E.WAV");
    r = outbox.peek(1);
    check(r.basenames == Names{"D.WAV"} && r.next == 2600, "E found in the call now running, from 2600: D's report can say 2600");
    outbox.remove(1);
    outbox.remove(5);
    check(outbox.size() == 0 && outbox.peek(1).next == 2600, "removing more than there are leaves none");

    outbox.setProgress(2100);
    check(outbox.peek(1).next == 2600, "progress never goes back");
    outbox.add("SAME.WAV");
    outbox.add("SAME.WAV");
    check(outbox.size() == 2, "the search hands on each basename once - the outbox keeps what it's given");
}

// The search's thread finds basenames and moves on, a call at a time; the
// sender's peeks, "sends" and removes. Every report has to leave nothing
// found below its number undelivered, and every basename has to be
// delivered once, in order.
static void testThreads() {
    printf("--- a search adding while reports send ---\n");
    const uint64_t start = 5000;
    BasenameOutbox outbox(start);
    // Every basename found, in order, with its call's start as the search
    // knows it.
    std::vector<std::pair<std::string, uint64_t>> found;
    std::mutex foundMutex;
    std::atomic<bool> done{false};
    const int calls = 3000;
    std::thread search([&]() {
        uint64_t next = start;
        for (int call = 0; call < calls; ++call) {
            for (int i = 0; i < call % 4; ++i) {
                const std::string name = "C" + std::to_string(call) + "_" + std::to_string(i) + ".WAV";
                {
                    std::lock_guard<std::mutex> lock(foundMutex);
                    found.emplace_back(name, next);
                }
                outbox.add(name);
            }
            next += 1000;
            outbox.setProgress(next);
        }
        done = true;
    });
    std::vector<std::string> delivered;
    bool everReportedPast = false, backwards = false;
    uint64_t lastNext = start;
    while (!done || outbox.size() > 0) {
        const size_t max = 1 + delivered.size() % 7;
        const BasenameOutbox::Report report = outbox.peek(max);
        // The first basename found that neither this report carries nor one
        // before it did - if there's one - has to be at or past the report's
        // number (the ones after it were found no earlier).
        {
            std::lock_guard<std::mutex> lock(foundMutex);
            const size_t firstUnsent = delivered.size() + report.basenames.size();
            if (firstUnsent < found.size())
                everReportedPast = everReportedPast || found[firstUnsent].second < report.next;
        }
        backwards = backwards || report.next < lastNext;
        lastNext = report.next;
        delivered.insert(delivered.end(), report.basenames.begin(), report.basenames.end());
        outbox.remove(report.basenames.size());
    }
    search.join();
    bool inOrder = delivered.size() == found.size();
    for (size_t i = 0; inOrder && i < delivered.size(); ++i)
        inOrder = delivered[i] == found[i].first;
    check(inOrder, std::to_string(delivered.size()) + " basenames delivered, each once, in the order found");
    check(!everReportedPast, "no report ever said the search got past a basename it didn't carry and wasn't delivered");
    check(!backwards, "nor went back");
    check(outbox.peek(1).next == start + 1000ull * calls, "and at the end, all the progress");
}

int main() {
    testReports();
    testThreads();
    if (g_failures) {
        fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("ALL %d CHECKS PASSED\n", g_checks);
    return 0;
}
