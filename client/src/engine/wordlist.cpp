#include "engine/wordlist.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "engine/english_1_words.h"

std::string normalizeMpqName(std::string s) {
    for (char& c : s) {
        if (c >= 'a' && c <= 'z')
            c = (char) (c - 'a' + 'A');
        else if (c == '/')
            c = '\\';
    }
    return s;
}

void parseWordList(const std::string& text, const std::string& sourceName, std::vector<std::string>& words,
                   std::vector<std::string>& warnings) {
    size_t lineStart = 0;
    int lineNo = 0;
    while (lineStart < text.size()) {
        size_t lineEnd = text.find('\n', lineStart);
        if (lineEnd == std::string::npos)
            lineEnd = text.size();
        ++lineNo;
        std::string line = text.substr(lineStart, lineEnd - lineStart);
        lineStart = lineEnd + 1;

        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos)
            continue;
        const size_t last = line.find_last_not_of(" \t");
        line = line.substr(first, last - first + 1);
        if (line[0] == '#')
            continue;
        bool printable = true;
        for (char c : line)
            printable = printable && c >= ' ' && c <= '~';
        if (!printable) {
            warnings.push_back(sourceName + ":" + std::to_string(lineNo) +
                               ": skipped - a word can only have printable ASCII characters (' ' to '~')");
            continue;
        }
        words.push_back(normalizeMpqName(line));
    }
}

bool loadWordListFile(const std::string& path, std::vector<std::string>& words, std::vector<std::string>& warnings,
                      std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open word list " + path;
        return false;
    }
    std::ostringstream text;
    text << in.rdbuf();
    if (in.bad()) {
        error = "cannot read word list " + path;
        return false;
    }
    parseWordList(text.str(), path, words, warnings);
    return true;
}

bool isBuiltinWordList(const std::string& name) {
    return name == kEnglish1;
}

std::vector<std::string> builtinWordList(const std::string& name) {
    std::vector<std::string> words;
    if (name != kEnglish1)
        return words;
    // Joined back into the file it was made from, and read like any other.
    std::string text;
    for (const char* word : kEnglish1Words) {
        text += word;
        text += '\n';
    }
    std::vector<std::string> warnings;
    parseWordList(text, kEnglish1, words, warnings);
    return words;
}

std::vector<std::string> sortedUniqueWords(std::vector<std::string> words) {
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    return words;
}

uint64_t fnv1a64(const std::string& bytes, uint64_t hash) {
    for (unsigned char c : bytes) {
        hash ^= c;
        hash *= 0x100000001B3ull;
    }
    return hash;
}

uint64_t wordListChecksum(const std::vector<std::string>& words) {
    uint64_t hash = fnv1a64("");
    for (const std::string& word : words)
        hash = fnv1a64(word + "\n", hash);
    return hash;
}

std::string hex64(uint64_t value) {
    char buf[17];
    snprintf(buf, sizeof buf, "%016llx", (unsigned long long) value);
    return buf;
}
