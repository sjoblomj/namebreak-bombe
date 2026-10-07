#ifndef NAMEBREAK_ENGINE_WORDLIST_H
#define NAMEBREAK_ENGINE_WORDLIST_H

#include <cstdint>
#include <string>
#include <vector>

// The word lists a dictionary search (dictionary_search.h) builds its
// candidates from: the dictionary compiled into the program (english-1) and
// any number of the user's own, one word per line. Every list is read the
// same way, so that the same files give the same words - and so the same
// candidate numbers - on every OS.

// The name of the dictionary compiled into the program - data/english-1.txt.
// It never changes; a different list would be english-2.
inline constexpr const char* kEnglish1 = "english-1";

// `s` as Storm hashes it: ASCII lowercase letters made uppercase and '/'
// made '\' (every other byte as it is). Words, separators, prefix, suffix
// and bounds all go through this, so that a search compares and hashes
// exactly what Storm would.
std::string normalizeMpqName(std::string s);

// Reads a word list's `text`: one word per line, LF or CRLF. Spaces and tabs
// around a word are dropped, and blank lines and lines starting with '#' are
// skipped. A word must be printable ASCII (' ' to '~') - a line with any
// other byte is skipped, with a warning in `warnings` naming `sourceName`
// and the line number. Each word is normalized (normalizeMpqName) and
// appended to `words`, in file order, duplicates and all.
void parseWordList(const std::string& text, const std::string& sourceName, std::vector<std::string>& words,
                   std::vector<std::string>& warnings);

// parseWordList on the file at `path`. False, with `error` set, if it can't
// be read.
bool loadWordListFile(const std::string& path, std::vector<std::string>& words, std::vector<std::string>& warnings,
                      std::string& error);

// Whether `name` is a dictionary compiled into the program.
bool isBuiltinWordList(const std::string& name);
// The words of the compiled-in dictionary `name`, read with parseWordList -
// empty if there's no such dictionary.
std::vector<std::string> builtinWordList(const std::string& name);

// `words` sorted (byte order, the order a search walks them in) and without
// duplicates.
std::vector<std::string> sortedUniqueWords(std::vector<std::string> words);

// A 64-bit FNV-1a checksum of `words`, in order, each followed by '\n' - to
// tell two lists apart, not a cryptographic hash.
uint64_t wordListChecksum(const std::vector<std::string>& words);

// FNV-1a 64 over `bytes`, continuing from `hash` - what wordListChecksum is
// made of, for checksums of other things.
uint64_t fnv1a64(const std::string& bytes, uint64_t hash = 0xCBF29CE484222325ull);

// `value` as 16 lowercase hex digits.
std::string hex64(uint64_t value);

#endif // NAMEBREAK_ENGINE_WORDLIST_H
