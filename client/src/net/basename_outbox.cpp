#include "net/basename_outbox.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "common/matches_file.h"
#include "engine/match_writer.h"

std::string BasenameOutbox::pathFor(const std::string& matchesDir, const std::string& targetName) {
    return namedFilePath(matchesDir, "unsent-basenames", targetName);
}

bool BasenameOutbox::open(const std::string& path, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = path;
    waiting_.clear();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
        return true;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty() && std::find(waiting_.begin(), waiting_.end(), line) == waiting_.end())
            waiting_.push_back(line);
    }
    if (in.bad()) {
        error = "cannot read " + path;
        return false;
    }
    return true;
}

void BasenameOutbox::add(const std::string& basename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(waiting_.begin(), waiting_.end(), basename) != waiting_.end())
        return;
    waiting_.push_back(basename);
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(path_).parent_path();
    if (!dir.empty())
        std::filesystem::create_directories(dir, ec);
    FILE* file = fopen(path_.c_str(), "ab");
    bool written = file != nullptr;
    if (file) {
        written = fprintf(file, "%s\n", basename.c_str()) >= 0;
        written = fclose(file) == 0 && written;
    }
    if (!written) {
        fprintf(stderr, "[coordinator] warning: cannot keep the basename %s in %s (%s) - it's still sent, but lost if this client stops first\n",
                basename.c_str(), path_.c_str(), strerror(errno));
    }
}

std::vector<std::string> BasenameOutbox::peek(size_t max) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<std::string>(waiting_.begin(), waiting_.begin() + (ptrdiff_t) std::min(max, waiting_.size()));
}

void BasenameOutbox::remove(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count == 0)
        return;
    waiting_.erase(waiting_.begin(), waiting_.begin() + (ptrdiff_t) std::min(count, waiting_.size()));
    rewrite();
}

size_t BasenameOutbox::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_.size();
}

void BasenameOutbox::rewrite() {
    std::error_code ec;
    if (waiting_.empty()) {
        std::filesystem::remove(path_, ec);
        if (ec)
            fprintf(stderr, "[coordinator] warning: cannot clear %s (%s) - what it has was sent already\n", path_.c_str(), ec.message().c_str());
        return;
    }
    std::string text;
    for (const std::string& basename : waiting_)
        text += basename + "\n";
    std::string error;
    if (!replaceFileContents(path_, text, error))
        fprintf(stderr, "[coordinator] warning: cannot rewrite %s (%s)\n", path_.c_str(), error.c_str());
}
