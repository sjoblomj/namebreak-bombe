#ifndef NAMEBREAK_ENGINE_DICTIONARY_SEARCH_H
#define NAMEBREAK_ENGINE_DICTIONARY_SEARCH_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "engine/backend.h"
#include "engine/dictionary.h"

// A dictionary search: every candidate made of words from a list, with
// separators between them (see dictionary.h for which candidates, and their
// numbers), checked against a target's hashA and hashB - and, given the
// file's encryption key, its basename against that key, and then only the
// candidates whose basename matches against the hashes (see
// dictionaryHashes, engine/backend.h).
//
// It walks the candidates in number order. The bounds are whole filenames:
// a candidate outside them is skipped, and so is everything after a leading
// part that already decides it's outside (FilenameBounds::classifyStart) -
// only the leading parts the bounds cut through have their candidates
// compared one by one.

// The MPQ hash's state: (seed1, seed2).
using HashState = std::pair<uint32_t, uint32_t>;
constexpr HashState kInitialHashState{0x7FED7FED, 0xEEEEEEEE};

// The crypt-table offsets of the hashes a search uses: hashA (Storm's hash
// type 1), hashB (2), and the one an encryption key is made with (3).
constexpr int kHashAOffset = 0x100;
constexpr int kHashBOffset = 0x200;
constexpr int kFileKeyOffset = 0x300;

// `state` continued over `text` with the crypt table at `offset`.
HashState continueHash(HashState state, const std::string& text, int offset, const uint32_t* cryptTable);
// The basename hash's state continued over `text`: hash type 3 of what
// follows the last '\' - every '\' starts it over from kInitialHashState.
// From kInitialHashState over a whole filename, its first value is the
// file's encryption key, before any adjustment for its position in the
// archive (mpqcli's encryption-key-raw).
HashState continueBasenameHash(HashState state, const std::string& text, const uint32_t* cryptTable);
// What follows the last '\' of `filename` (all of it, if there's none).
std::string basenameOf(const std::string& filename);

struct DictionaryRequest {
    // The words (sorted, without duplicates), separators and word counts -
    // see DictionaryPattern.
    DictionaryPattern pattern;
    // Normalized (normalizeMpqName, wordlist.h), as is everything below
    // that's text.
    std::string prefix;
    std::string suffix;
    FilenameBounds bounds;
    uint32_t targetHashA = 0;
    uint32_t targetHashB = 0;
    // The file's encryption key is known: compare each candidate's basename
    // to basenameKey (see continueBasenameHash). Unless the suffix has a
    // '\', only the candidates whose basename matches are then compared to
    // the targets - the file's name has that basename - unless
    // recordHashAMatches (see dictionaryHashes, engine/backend.h).
    bool checkBasename = false;
    uint32_t basenameKey = 0;
    // With checkBasename, write every basename that matches to
    // basenamesFilePath (or, without one, only to onBasenameMatch).
    bool recordBasenames = true;
    // With checkBasename, compare every candidate to the targets all the
    // same, reporting every hashA hit, as if the key weren't known.
    bool recordHashAMatches = false;
    // The number of the first candidate to search - every one before it is
    // taken as searched (see the progress file below) - and of the first not
    // to: a coordinator range is the candidates [startNumber, endNumber).
    uint64_t startNumber = 0;
    uint64_t endNumber = UINT64_MAX;
    // Where the words came from, for the summary at the start - e.g.
    // "english-1 + words.txt".
    std::string wordSource;
    // The most recent hashA match, or the match of both hashes - see
    // SearchRequest::outputFilePath.
    std::string outputFilePath = "matches.txt";
    // With recordBasenames, every basename that matched basenameKey, one per
    // line, each once - appended to, never replaced. Empty for none: a
    // coordinator range's go to onBasenameMatch only (see
    // runDictionarySearch).
    std::string basenamesFilePath = "basenames.txt";
    // The progress file - see writeDictionaryProgress. Written this often,
    // whenever the search pauses, and when it ends. Empty for none.
    std::string progressFilePath = "wordnumber.txt";
    std::chrono::milliseconds progressInterval{30000};
    // How often a line of progress is printed.
    std::chrono::milliseconds statusInterval{10000};
};

struct DictionaryResult {
    // False only for a problem with the request or its files - see `error`.
    bool ok = true;
    std::string error;
    // True once a candidate matched both hashes - `filename`.
    bool found = false;
    std::string filename;
    // True if abortRequested stopped the search.
    bool aborted = false;
    // Every candidate numbered below this has been searched - endNumber (or
    // the number of candidates there are) once all of them have.
    uint64_t nextNumber = 0;
    // How many candidates were hashed (those the bounds skipped aren't).
    uint64_t candidatesSearched = 0;
    // How many basename matches were found (each basename counted once).
    uint64_t basenameHits = 0;
};

// What a dictionary search tells its caller as it goes - each optional, and
// each called on the thread running the search.
struct DictionarySearchHooks {
    // With every hashA match's filename.
    std::function<void(const std::string& filename)> onPartialMatch;
    // With every basename that matched the key, the first time it does in
    // this search - and the filename it matched in.
    std::function<void(const std::string& basename, const std::string& filename)> onBasenameMatch;
    // After every call to the backend, and at the end, with the search's
    // DictionaryResult::nextNumber so far: every candidate numbered below it
    // has been searched.
    std::function<void(uint64_t nextNumber)> onProgress;
    // Alongside onProgress: how many candidates have been hashed so far
    // (DictionaryResult::candidatesSearched), of how many there are to hash
    // in all - what the progress line's percentage is of.
    std::function<void(uint64_t searched, uint64_t toSearch)> onCount;
};

// Runs a dictionary search on `backend`, which must supportsDictionary(). The
// rest as runSearch (search.h): abortRequested stops it, pauseRequested
// holds it between calls to the backend, and onPartialMatch is called with
// every hashA match.
DictionaryResult runDictionarySearch(SearchBackend& backend, const DictionaryRequest& req, std::atomic<bool>* abortRequested = nullptr,
                                     std::function<void(const std::string&)> onPartialMatch = nullptr,
                                     const std::atomic<bool>* pauseRequested = nullptr);
// The same, with every hook.
DictionaryResult runDictionarySearch(SearchBackend& backend, const DictionaryRequest& req, std::atomic<bool>* abortRequested,
                                     const DictionarySearchHooks& hooks, const std::atomic<bool>* pauseRequested = nullptr);

// How many of `space`'s candidates numbered from `startNumber` up to (not
// including) `endNumber` have a filename (prefix + candidate + suffix) within
// `bounds` - what a search of them hashes.
uint64_t countDictionaryCandidates(const DictionarySpace& space, const std::string& prefix, const std::string& suffix,
                                   const FilenameBounds& bounds, uint64_t startNumber, uint64_t endNumber = UINT64_MAX);

// `seconds` for a human: "6s", "4m05s", "1h02m03s", "2d03h04m" - or "?" if
// it's negative or absurdly long.
std::string formatDuration(double seconds);

// Identifies everything that decides which candidates a search checks and
// what it looks for - every field of `req` but the start and end, the file
// paths and the intervals. 16 hex digits.
std::string dictionaryFingerprint(const DictionaryRequest& req);

// The fingerprints of the searches a search of `req` may be resumed from:
// its own, and those of the searches that did at least as much with the
// same words, bounds and hashes - that compared every candidate to the
// hashes where it compares only those whose basename matches the key (see
// dictionaryHashes), or recorded the basenames where it doesn't. Of those,
// only the ones with its key, or with none.
std::vector<std::string> dictionaryResumableFingerprints(const DictionaryRequest& req);

// The progress file: which search wrote it, and the number every candidate
// below which has been searched - plus, for a human, how many candidates
// there are, and the one at that number. Lines of `key = value`:
//
//   # namebreak dictionary search progress
//   fingerprint = 0123456789abcdef
//   next = 123456
//   total = 4080040000
//   candidate = CRDT_LST
//
// Replaced as a whole each time (replaceFileContents, match_writer.h), so
// it's never half written.
struct DictionaryProgress {
    std::string fingerprint;
    uint64_t next = 0;
};
bool writeDictionaryProgress(const std::string& path, const DictionaryProgress& progress, uint64_t total,
                             const std::string& candidate, std::string& error);
// False, with `error` set, if the file can't be read or isn't a progress
// file. `exists` is false (and the result true) if there's no file at all.
bool readDictionaryProgress(const std::string& path, DictionaryProgress& out, bool& exists, std::string& error);

#endif // NAMEBREAK_ENGINE_DICTIONARY_SEARCH_H
