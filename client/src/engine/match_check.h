#ifndef NAMEBREAK_ENGINE_MATCH_CHECK_H
#define NAMEBREAK_ENGINE_MATCH_CHECK_H

#include <cstdint>
#include <string>

// What else is known about the file than its two hashes, and so what a name
// matching both has to have as well to be its name. Two 32-bit hashes match
// by chance about once in 2^64 names - and a long search hashes that many:
// REZ\SFZSSP050WSRL.WAV matched a file's hashA and hashB, and was nothing
// like its name. Checked only once both hashes match, which almost never
// happens, so it costs nothing.
//
// A name that matches both hashes but not this is a false match: never the
// end of a search, which goes on past it as if it were only a hashA match -
// it's reported as one too (see FalseMatch).
struct MatchCheck {
    // The file's encryption key, before the adjustment for its position and
    // size in the archive (mpqcli's encryption-key-raw): hash type 3 of the
    // name's basename - what follows its last '\' - must be it.
    bool checkKey = false;
    uint32_t key = 0;
    // Where the file's entry is in the archive's hash table: Storm looks a
    // name up from slot hash type 0 % hashTableSize onwards, slot by slot,
    // and gives up at the first one never used - so the name's slot has to
    // be one from which the lookup reaches the entry, firstStartSlot to
    // lastStartSlot (wrapping past the table's end if firstStartSlot is
    // the greater). hashTableSize is a power of two, as Storm's is.
    bool checkSlot = false;
    uint32_t hashTableSize = 0;
    uint32_t firstStartSlot = 0;
    uint32_t lastStartSlot = 0;

    bool any() const { return checkKey || checkSlot; }
};

// A name that matched both hashes but not the MatchCheck, and why not.
struct FalseMatch {
    std::string filename;
    std::string reason;

    bool operator==(const FalseMatch& other) const { return filename == other.filename && reason == other.reason; }
};

// Why `filename`, which matches both of the file's hashes, isn't its name
// after all - "" if it passes `check`. `cryptTable` has 0x500 entries (see
// prepareCryptTable).
std::string falseMatchReason(const MatchCheck& check, const std::string& filename, const uint32_t* cryptTable);

// Prints what `check` checks, if anything - with a search's other settings,
// as it starts.
void printMatchCheck(const MatchCheck& check);

// The slot Storm's lookup of `filename` starts at, in a hash table of
// `hashTableSize` slots (a power of two).
uint32_t hashTableStartSlot(const std::string& filename, uint32_t hashTableSize, const uint32_t* cryptTable);

#endif // NAMEBREAK_ENGINE_MATCH_CHECK_H
