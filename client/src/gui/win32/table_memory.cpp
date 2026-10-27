// The meet-in-the-middle memory slider's sizes - see table_memory.h.

#include "gui/win32/table_memory.h"

#include <algorithm>
#include <cstdio>

#include "engine/limits.h"
#include "engine/meet_in_the_middle.h"
#include "gui/win32/win32.h"

namespace gui {

namespace {

// The sizes the slider stops at - none, then powers of two from 256 MB up.
constexpr uint64_t kTableMemoryStops[] = {0,            256ull << 20, 512ull << 20, 1ull << 30,  2ull << 30,
                                           4ull << 30,  8ull << 30,   16ull << 30,  32ull << 30, 64ull << 30};

} // namespace

std::vector<uint64_t> tableMemoryStopsFor(uint64_t totalBytes) {
    std::vector<uint64_t> stops;
    for (uint64_t stop : kTableMemoryStops)
        if (stop <= totalBytes || stops.size() < 2)
            stops.push_back(stop);
    return stops;
}

uint64_t proposedTableMemory(const std::vector<uint64_t>& stops, uint64_t totalBytes) {
    const uint64_t cap = (std::min) (totalBytes / 8, 4ull << 30);
    uint64_t proposed = 0;
    for (uint64_t stop : stops)
        if (stop <= cap)
            proposed = stop;
    return proposed;
}

int stopIndexFor(const std::vector<uint64_t>& stops, uint64_t bytes) {
    int index = 0;
    for (int i = 0; i < (int) stops.size(); ++i)
        if (stops[i] <= bytes)
            index = i;
    return index;
}

std::string bytesForConfig(uint64_t bytes) {
    if (bytes == 0)
        return "0";
    if (bytes % (1ull << 30) == 0)
        return std::to_string(bytes >> 30) + "G";
    return std::to_string(bytes >> 20) + "M";
}

std::string stopText(uint64_t bytes) {
    if (bytes == 0)
        return "None";
    if (bytes < (1ull << 30))
        return std::to_string(bytes >> 20) + " MB";
    return std::to_string(bytes >> 30) + " GB";
}

std::string memoryText(uint64_t bytes) {
    char buf[32];
    snprintf(buf, sizeof buf, "%.1f GB", (double) bytes / (double) (1ull << 30));
    return buf;
}

std::string tableMemoryLabel(uint64_t bytes, uint64_t availableBytes) {
    if (bytes == 0)
        return "None: the server hands this client no meet-in-the-middle work.";
    std::string text = stopText(bytes) + " - a table of " +
                       std::to_string(meetInTheMiddleTableLengthForBytes(43, MAX_CANDIDATE_LEN, bytes)) +
                       " characters of a 43-character alphabet, " +
                       std::to_string(meetInTheMiddleTableLengthForBytes(26, MAX_CANDIDATE_LEN, bytes)) + " of a 26-character one.";
    if (bytes > availableBytes)
        text += "\r\nThat is more than is free on this machine right now.";
    return text;
}

MachineMemory machineMemory() {
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof memory;
    if (!GlobalMemoryStatusEx(&memory))
        return {4ull << 30, 1ull << 30}; // modest, if Windows won't say
    return {memory.ullTotalPhys, memory.ullAvailPhys};
}

} // namespace gui
