#ifndef NAMEBREAK_GUI_WIN32_TABLE_MEMORY_H
#define NAMEBREAK_GUI_WIN32_TABLE_MEMORY_H

#include <cstdint>
#include <string>
#include <vector>

// How much memory this client will give a meet-in-the-middle target's table
// ([hardware]'s meet_in_the_middle_table_memory) - the Hardware dialog's
// slider (hardware_dialog.h), and what first-run setup proposes. It is a
// coarse choice: a character more of the table needs about as many times
// more memory as the alphabet has characters, so the useful values are
// powers of two far apart, and a slider over them says more than a number
// box would (what you are giving, out of what this machine has).

namespace gui {

// The sizes the slider stops at - none, then powers of two from 256 MB up -
// that a machine with `totalBytes` of memory could hold at all: always at
// least None and the smallest size, so there is something to drag
// even on a very small machine.
std::vector<uint64_t> tableMemoryStopsFor(uint64_t totalBytes);

// Which of `stops` to propose on a machine this size: an eighth of its memory at most,
// and never more than 4 GB unasked - enough for five characters of a
// 43-character alphabet, which is what the directories looked for in
// practice need.
uint64_t proposedTableMemory(const std::vector<uint64_t>& stops, uint64_t totalBytes);

// The stop at or below `bytes` - which stop a value from config.conf sits
// at, rounded down so that the slider never offers more than the file asked
// for.
int stopIndexFor(const std::vector<uint64_t>& stops, uint64_t bytes);

// A size as config.conf writes it: 0, 256M, 4G.
std::string bytesForConfig(uint64_t bytes);

// A size as the slider says it: None, 256 MB, 4 GB (every stop is a whole
// number of its unit).
std::string stopText(uint64_t bytes);

// A machine's memory, as the line above the slider says it: 31.8 GB.
std::string memoryText(uint64_t bytes);

// What a stop means, for the line under the slider: the size, and how long a
// table of it holds - which is the whole point of giving the memory. Two
// alphabet sizes, because what one buys depends on the target's own alphabet
// and the steps are wide: 2 GB and 32 GB are the same five characters of a
// 43-character alphabet, but six and seven of a 26-character one.
std::string tableMemoryLabel(uint64_t bytes, uint64_t availableBytes);

// This machine's memory, and how much of it is free now.
struct MachineMemory {
    uint64_t totalBytes;
    uint64_t availableBytes;
};
MachineMemory machineMemory();

} // namespace gui

#endif // NAMEBREAK_GUI_WIN32_TABLE_MEMORY_H
