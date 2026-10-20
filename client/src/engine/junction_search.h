#ifndef NAMEBREAK_ENGINE_JUNCTION_SEARCH_H
#define NAMEBREAK_ENGINE_JUNCTION_SEARCH_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine/backend.h"
#include "engine/junction.h"

// A junction search: a file whose name is a known prefix, an unknown
// directory and a known suffix, where the hash state after the directory -
// the junction - is already known, for hashA and for hashB. See junction.h
// for what that buys (the search comes at the directory from both ends and
// meets in the middle) and README.md's "Junction mode" for how to run one.
//
// The search goes through the directories in the order of its backward
// walk's nodes (JunctionSpace): shortest tail first, and within a tail
// length by the tail's index. A node's number is where a search resumes
// from, and what a range of a search is given in.

struct JunctionRequest {
    // The characters the unknown directory is made of.
    std::string alphabet;
    // The filename around it. The prefix may be empty; the suffix is what
    // follows the directory - for a file in that directory, the '\' and the
    // basename.
    std::string prefix;
    std::string suffix;
    // The hash state after prefix + the directory, for hashA and for hashB
    // (tools/mitm/junction.c recovers these). Hashing `suffix` onto them
    // must give the file's two hashes, or the search refuses to start.
    HashState junctionA{0, 0};
    HashState junctionB{0, 0};
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
    // How long the directory may be, in characters.
    int minLength = 1;
    int maxLength = 1;
    // How many of a directory's first characters the table holds - the rest
    // are what the backward walk searches, so each character here is one
    // the search doesn't brute-force. Negative (the default): as many as
    // `tableBytes` has room for (junctionTableLengthForBytes). 0: no table
    // at all, and the whole directory is rewound from the junction - a
    // brute force in the other direction, which only a test wants.
    int tableLength = -1;
    // The memory the table may take, when its length is left to be worked
    // out. A character more needs about as many times more memory as the
    // alphabet has characters, so this decides how far the search reaches.
    uint64_t tableBytes = (uint64_t) 1 << 30;
    // The cheap rules that skip implausible directories - as in a bounded
    // search, but applied to every character of the directory rather than
    // its leading ones (see dirPasses, junction.h).
    PruneRules rules;
    // The walk's nodes to search: [startNumber, endNumber). Every node
    // before startNumber counts as searched (see the progress file).
    uint64_t startNumber = 0;
    uint64_t endNumber = UINT64_MAX;
    // The most recent hit, or the match of both hashes - see
    // SearchRequest::outputFilePath (search.h).
    std::string outputFilePath = "matches.txt";
    // The progress file - see writeJunctionProgress. Written this often,
    // whenever the search pauses, and when it ends. Empty for none.
    std::string progressFilePath = "junction.txt";
    std::chrono::milliseconds progressInterval{30000};
    // How often a line of progress is printed.
    std::chrono::milliseconds statusInterval{10000};
};

struct JunctionResult {
    // False only for a problem with the request or its files - see `error`.
    bool ok = true;
    std::string error;
    // True once a directory matched both of the file's hashes - `filename`.
    bool found = false;
    std::string filename;
    // True if abortRequested stopped the search.
    bool aborted = false;
    // Every node numbered below this has been searched - endNumber (or the
    // number of nodes there are) once all of them have.
    uint64_t nextNumber = 0;
    // How many tails were looked up in the table (those the rules skipped
    // aren't), and how many directories matched the junction.
    uint64_t tailsProbed = 0;
    uint64_t hits = 0;
    // The table the search used: how many of a directory's characters it
    // held, how many directories that was, and the memory it took.
    int tableLength = 0;
    uint64_t tableEntries = 0;
    uint64_t tableBytes = 0;
};

// What a junction search tells its caller as it goes - each optional, and
// each called on the thread running the search.
struct JunctionSearchHooks {
    // With the filename of every directory that matched the junction.
    std::function<void(const std::string& filename)> onPartialMatch;
    // After every call to the backend, and at the end, with the search's
    // JunctionResult::nextNumber so far.
    std::function<void(uint64_t nextNumber)> onProgress;
    // Alongside onProgress: how many of the walk's nodes have been searched,
    // of how many there are to search - what the progress line's percentage
    // is of.
    std::function<void(uint64_t searched, uint64_t toSearch)> onCount;
    // With every line of progress (see JunctionRequest::statusInterval), the
    // last of them at the end: the tail the walk is about to reach, with the
    // table's characters written as '?' - "" once there's none left.
    std::function<void(const std::string& tail)> onStatus;
};

// Runs a junction search on `backend`, which must supportsJunction(). The
// rest as runSearch (search.h): abortRequested stops it, pauseRequested
// holds it between calls to the backend, and onPartialMatch is called with
// every directory that matches the junction.
JunctionResult runJunctionSearch(SearchBackend& backend, const JunctionRequest& req, std::atomic<bool>* abortRequested = nullptr,
                                 std::function<void(const std::string&)> onPartialMatch = nullptr,
                                 const std::atomic<bool>* pauseRequested = nullptr);
// The same, with every hook.
JunctionResult runJunctionSearch(SearchBackend& backend, const JunctionRequest& req, std::atomic<bool>* abortRequested,
                                 const JunctionSearchHooks& hooks, const std::atomic<bool>* pauseRequested = nullptr);

// Whether the junction states and the file's hashes agree: hashing the
// suffix onto the junction must give the hash, for hashA and for hashB. They
// come from different places (the junction from a recovery like
// tools/mitm/junction.c, the hashes from the archive), and a junction that
// doesn't fit the hashes - the state after the wrong position in the name,
// say, or the state of another file's directory - would have the search walk
// the whole space and find nothing, so it's checked before it starts.
// False, with `error` saying which of the two doesn't fit.
bool junctionFitsHashes(const JunctionRequest& req, const uint32_t* cryptTable, std::string& error);

// How much of a junction search a table of `tableLength` characters leaves
// to brute-force: how many tails the walk has to rewind, all lengths
// together. UINT64_MAX if that doesn't fit in 64 bits.
uint64_t junctionTailsToWalk(int alphabetSize, int maxLength, int tableLength);

// Identifies everything that decides which directories a search covers and
// what it looks for - every field of `req` but the start and end, the file
// paths and the intervals. 16 hex digits.
std::string junctionFingerprint(const JunctionRequest& req);

// The progress file: which search wrote it, and the node number every node
// below which has been searched - plus, for a human, how many nodes there
// are and the tail at that number. Lines of `key = value`:
//
//   # namebreak junction search progress
//   fingerprint = 0123456789abcdef
//   next = 123456
//   total = 147008443
//   tail = ?????AT
//
// Replaced as a whole each time (replaceFileContents, match_writer.h), so
// it's never half written.
struct JunctionProgress {
    std::string fingerprint;
    uint64_t next = 0;
};
bool writeJunctionProgress(const std::string& path, const JunctionProgress& progress, uint64_t total, const std::string& tail,
                           std::string& error);
// False, with `error` set, if the file can't be read or isn't a junction
// search's progress file. `exists` is false (and the result true) if there's
// no file at all.
bool readJunctionProgress(const std::string& path, JunctionProgress& out, bool& exists, std::string& error);

#endif // NAMEBREAK_ENGINE_JUNCTION_SEARCH_H
