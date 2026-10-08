#include "net/basename_outbox.h"

#include <algorithm>

void BasenameOutbox::add(const std::string& basename) {
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_.push_back({basename, progress_});
}

void BasenameOutbox::setProgress(uint64_t next) {
    std::lock_guard<std::mutex> lock(mutex_);
    progress_ = std::max(progress_, next);
}

BasenameOutbox::Report BasenameOutbox::peek(size_t max) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Report report;
    const size_t count = std::min(max, waiting_.size());
    for (size_t i = 0; i < count; ++i)
        report.basenames.push_back(waiting_[i].basename);
    report.next = count < waiting_.size() ? std::min(progress_, waiting_[count].foundAt) : progress_;
    return report;
}

void BasenameOutbox::remove(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_.erase(waiting_.begin(), waiting_.begin() + (ptrdiff_t) std::min(count, waiting_.size()));
}

size_t BasenameOutbox::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_.size();
}
