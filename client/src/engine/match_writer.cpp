#include "engine/match_writer.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

namespace {

// Opens `path` in `mode` ("w" or "a") and writes `line` and a line break.
bool writeLine(const std::string& path, const char* mode, const std::string& line, std::string& error) {
    FILE* f = fopen(path.c_str(), mode);
    if (!f) {
        error = "fopen " + path + ": " + strerror(errno);
        return false;
    }
    bool ok = fprintf(f, "%s\n", line.c_str()) >= 0;
    ok = fclose(f) == 0 && ok;
    if (!ok)
        error = "cannot write " + path + ": " + strerror(errno);
    return ok;
}

// Writes `contents` to `path`, replacing what it held - in text mode, as
// writeLine does, so a line break is "\r\n" on Windows.
bool writeContents(const std::string& path, const std::string& contents, std::string& error) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        error = "fopen " + path + ": " + strerror(errno);
        return false;
    }
    bool ok = fwrite(contents.data(), 1, contents.size(), f) == contents.size();
    ok = fclose(f) == 0 && ok;
    if (!ok)
        error = "cannot write " + path + ": " + strerror(errno);
    return ok;
}

} // namespace

bool replaceFileContents(const std::string& path, const std::string& contents, std::string& error) {
    const std::string tmp = path + ".tmp";
    if (!writeContents(tmp, contents, error))
        return false;
    std::error_code ec;
    // Windows won't rename over a file another process has open - as the GUI
    // does, for a moment, once a second - so it's tried a few times. Then
    // the file is written in place instead, where a reader could catch it
    // half-written.
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::filesystem::rename(tmp, path, ec);
        if (!ec)
            return true;
    }
    std::filesystem::remove(tmp, ec);
    return writeContents(path, contents, error);
}

namespace {

// Replaces what `path` holds with `line` - see replaceFileContents.
bool replaceWithLine(const std::string& path, const std::string& line, std::string& error) {
    return replaceFileContents(path, line + "\n", error);
}

} // namespace

std::string foundFilePath(const std::string& matchesPath) {
    return (std::filesystem::path(matchesPath).parent_path() / "found.txt").string();
}

bool MatchWriter::open(const std::string& path, std::string& error) {
    path_ = path;
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            error = "cannot create " + dir.string() + ": " + ec.message();
            return false;
        }
    }
    FILE* f = fopen(path.c_str(), "a");
    if (!f) {
        error = "fopen " + path + ": " + strerror(errno);
        return false;
    }
    fclose(f);
    return true;
}

void MatchWriter::hit(const std::string& filename) {
    if (foundWritten_)
        return;
    pending_ = filename;
    hasPending_ = true;
}

void MatchWriter::writeIfDue() {
    if (hasPending_ && (!wroteAny_ || std::chrono::steady_clock::now() - lastWrite_ >= interval_))
        write();
}

void MatchWriter::flush() {
    if (hasPending_)
        write();
}

void MatchWriter::write() {
    hasPending_ = false;
    wroteAny_ = true;
    lastWrite_ = std::chrono::steady_clock::now();
    std::string error;
    if (replaceWithLine(path_, pending_, error)) {
        warned_ = false;
    } else if (!warned_) {
        fprintf(stderr, "warning: cannot record the latest match in %s (%s) - trying again with the next one\n", path_.c_str(),
                error.c_str());
        warned_ = true;
    }
}

bool MatchWriter::found(const std::string& filename) {
    hasPending_ = false;
    foundWritten_ = true;
    bool ok = true;
    std::string error;
    const std::string foundPath = foundFilePath(path_);
    if (!writeLine(foundPath, "a", filename, error)) {
        fprintf(stderr, "ERROR: cannot record the match of both hashes (%s) in %s: %s\n", filename.c_str(), foundPath.c_str(), error.c_str());
        ok = false;
    }
    if (!replaceWithLine(path_, filename, error)) {
        fprintf(stderr, "ERROR: cannot record the match of both hashes (%s) in %s: %s\n", filename.c_str(), path_.c_str(), error.c_str());
        ok = false;
    }
    return ok;
}
