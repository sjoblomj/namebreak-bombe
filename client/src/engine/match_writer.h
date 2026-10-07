#ifndef NAMEBREAK_ENGINE_MATCH_WRITER_H
#define NAMEBREAK_ENGINE_MATCH_WRITER_H

#include <chrono>
#include <string>

// Keeps a search's matches file (SearchRequest::outputFilePath) at its most
// recent Hash-A match: one line, replaced rather than appended to. Appending
// every hit - hundreds a second on a fast GPU - made a target's file
// hundreds of MB, and all anything reads of it is the last line (the GUI's
// list and progress bar, resume_from_last_candidate).
//
// A match of both hashes is always the line it ends with: it's written at
// once, and nothing after it replaces it. It's also appended to found.txt
// next to it (foundFilePath), which nothing ever replaces - a later search
// writing the same matches file (a coordinator target whose find was never
// reported, say, claimed again after a restart) would otherwise lose it.
class MatchWriter {
public:
    // How often a Hash-A-only match is written, at most - replacing a file
    // costs more than appending a line did, and nothing reads it more often
    // than this (the GUI, once a second).
    static constexpr std::chrono::milliseconds kDefaultInterval{1000};

    explicit MatchWriter(std::chrono::milliseconds interval = kDefaultInterval) : interval_(interval) {}

    // Creates `path`'s missing directories, and `path` itself if it's
    // missing - leaving what it holds until the first match replaces it.
    // Returns false with `error` set if it can't be written.
    bool open(const std::string& path, std::string& error);

    // A Hash-A-only match - the file's line once writeIfDue() or flush()
    // writes it, unless another comes first.
    void hit(const std::string& filename);
    // Writes the latest hit not written yet, if the interval has passed
    // since the last write.
    void writeIfDue();
    // Writes the latest hit not written yet - at the end of a search.
    void flush();
    // A match of both hashes: written at once, to the matches file and to
    // found.txt. Every hit after it is ignored. Returns false (having
    // printed why) if either couldn't be written.
    bool found(const std::string& filename);

private:
    void write();

    std::string path_;
    std::chrono::milliseconds interval_;
    std::chrono::steady_clock::time_point lastWrite_;
    bool wroteAny_ = false;
    std::string pending_;
    bool hasPending_ = false;
    bool foundWritten_ = false;
    // A failed write is reported once, until one succeeds again.
    bool warned_ = false;
};

// Replaces what `path` holds with `contents`: written to a file next to it,
// which is then renamed over it - so a reader (the GUI, once a second) gets
// what it held before or `contents`, never part of either. False, with
// `error` set, if it can't be written.
bool replaceFileContents(const std::string& path, const std::string& contents, std::string& error);

// The file a search whose matches file is `matchesPath` appends its
// both-hashes matches to: found.txt, in the same directory.
std::string foundFilePath(const std::string& matchesPath);

#endif // NAMEBREAK_ENGINE_MATCH_WRITER_H
