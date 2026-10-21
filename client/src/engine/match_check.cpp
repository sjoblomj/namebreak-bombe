#include "engine/match_check.h"

#include <cstdio>

#include "engine/mpq_hash.h"

namespace {

// Hash type 0, the one the hash table is indexed by.
constexpr int kTableIndexOffset = 0x000;

std::string hex32(uint32_t value) {
    char text[16];
    snprintf(text, sizeof text, "0x%08X", value);
    return text;
}

} // namespace

uint32_t hashTableStartSlot(const std::string& filename, uint32_t hashTableSize, const uint32_t* cryptTable) {
    return continueHash(kInitialHashState, filename, kTableIndexOffset, cryptTable).first & (hashTableSize - 1);
}

std::string falseMatchReason(const MatchCheck& check, const std::string& filename, const uint32_t* cryptTable) {
    std::string reason;
    if (check.checkKey) {
        const size_t slash = filename.rfind('\\');
        const std::string basename = slash == std::string::npos ? filename : filename.substr(slash + 1);
        const uint32_t key = continueHash(kInitialHashState, basename, kFileKeyOffset, cryptTable).first;
        if (key != check.key)
            reason = "its basename's key is " + hex32(key) + ", not the file's " + hex32(check.key);
    }
    if (check.checkSlot && check.hashTableSize > 0) {
        const uint32_t slot = hashTableStartSlot(filename, check.hashTableSize, cryptTable);
        const bool reaches = check.firstStartSlot <= check.lastStartSlot ? slot >= check.firstStartSlot && slot <= check.lastStartSlot
                                                                         : slot >= check.firstStartSlot || slot <= check.lastStartSlot;
        if (!reaches) {
            if (!reason.empty())
                reason += ", and ";
            reason += "its lookup starts at hash table slot " + std::to_string(slot) + ", not at one from " + std::to_string(check.firstStartSlot) +
                      " to " + std::to_string(check.lastStartSlot) + " of " + std::to_string(check.hashTableSize);
        }
    }
    return reason;
}

void printMatchCheck(const MatchCheck& check) {
    if (check.checkKey)
        printf("encryption key: %s (a match of both hashes must have a basename with it)\n", hex32(check.key).c_str());
    if (check.checkSlot)
        printf("hash table: lookups starting at slots %u to %u of %u reach the file (a match of both hashes must start at one)\n",
               check.firstStartSlot, check.lastStartSlot, check.hashTableSize);
}
