#ifndef NAMEBREAK_NET_BASENAME_OUTBOX_H
#define NAMEBREAK_NET_BASENAME_OUTBOX_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// What a dictionary range's reports to the coordinator say (see
// ClaimResponse::sendBasenames): how far its search has got, and the
// basenames it has found that the server hasn't got yet - those found since
// the last report that got an answer (heartbeat, quit or completion).
//
// A report never says the search has got further than a basename it doesn't
// carry: the server takes everything below a report's number as searched,
// and would never hand it out again - so a basename found there that never
// reached it (the client crashed, say) would be missed for good. Each
// basename is kept with how far the search had got when it was found (the
// first number of the backend call that found it), and a report that can't
// carry them all - at most kMaxBasenamesPerReport - says the search has got
// no further than that of the first one it leaves out. The rest of the
// range is searched again then, if it comes to that, and the basenames found
// again: they're kept in memory only, never in a file.
//
// The search adds to it on one thread while the heartbeats send from it on
// another; one report at a time sends (`peek`, then `remove` once it got an
// answer), and the search only ever adds after what was peeked.
class BasenameOutbox {
public:
    // A range whose search starts at number `start`.
    explicit BasenameOutbox(uint64_t start) : progress_(start) {}

    // The search found `basename`, in the backend call it's in now - which
    // starts where `setProgress` last said.
    void add(const std::string& basename);

    // Every candidate numbered below `next` has been searched, and every
    // basename found among them added.
    void setProgress(uint64_t next);

    struct Report {
        std::vector<std::string> basenames;
        // How far a report carrying `basenames` may say the search has got:
        // the progress, or where the first basename left waiting was found.
        uint64_t next = 0;
    };
    // The oldest `max` (or fewer) basenames waiting, to send - and the
    // progress a report carrying them may claim.
    Report peek(size_t max) const;

    // Drops the oldest `count` - those a report just got to the server.
    void remove(size_t count);

    size_t size() const;

private:
    struct Waiting {
        std::string basename;
        uint64_t foundAt;
    };

    mutable std::mutex mutex_;
    uint64_t progress_;
    std::vector<Waiting> waiting_;
};

#endif // NAMEBREAK_NET_BASENAME_OUTBOX_H
