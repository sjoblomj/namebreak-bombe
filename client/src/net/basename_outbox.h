#ifndef NAMEBREAK_NET_BASENAME_OUTBOX_H
#define NAMEBREAK_NET_BASENAME_OUTBOX_H

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

// The basenames a coordinator client has found for a target (see
// ClaimResponse::sendBasenames) that the server hasn't got yet: the ones
// found since the last report - heartbeat, quit or completion - that got an
// answer. Kept in a file per target, one per line, oldest first, which is
// cleared as reports get through - so a basename isn't lost when one
// doesn't, nor when the client quits or stops before sending it: the next
// report of a range of the same target sends what's left.
//
// The search adds to it on one thread while the heartbeats send from it on
// another; one report at a time sends (`peek`, then `remove` once it got an
// answer), and the search only ever adds after what was peeked.
class BasenameOutbox {
public:
    // The file for `targetName`'s in `matchesDir`:
    // unsent-basenames-<target name>.txt (see namedFilePath, matches_file.h).
    static std::string pathFor(const std::string& matchesDir, const std::string& targetName);

    // Reads what an earlier search left in `path`, if anything - a missing
    // file is an empty outbox. False, with `error` set, if it can't be read.
    bool open(const std::string& path, std::string& error);

    // Adds `basename` at the end, unless it's waiting already, and appends
    // it to the file - which is created then. A warning if it can't be
    // written; it's still sent.
    void add(const std::string& basename);

    // The oldest `max` (or fewer) waiting, to send.
    std::vector<std::string> peek(size_t max) const;

    // Drops the oldest `count` - those a report just got to the server -
    // and rewrites the file with the rest, or removes it if there are none.
    void remove(size_t count);

    size_t size() const;
    const std::string& path() const { return path_; }

private:
    void rewrite();

    mutable std::mutex mutex_;
    std::string path_;
    std::vector<std::string> waiting_;
};

#endif // NAMEBREAK_NET_BASENAME_OUTBOX_H
