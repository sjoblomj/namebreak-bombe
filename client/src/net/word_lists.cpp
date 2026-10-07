#include "net/word_lists.h"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "engine/match_writer.h"
#include "engine/wordlist.h"

namespace {

// `text`'s words, read as a word list, sorted and without duplicates - and
// their checksum, as the server gives it.
std::string checksumOfList(const std::string& text, const std::string& name, std::vector<std::string>& words) {
    std::vector<std::string> warnings;
    parseWordList(text, name, words, warnings);
    return hex64(wordListChecksum(sortedUniqueWords(words)));
}

bool readFile(const std::string& path, std::string& text) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    std::ostringstream contents;
    contents << in.rdbuf();
    text = contents.str();
    return !in.bad();
}

} // namespace

bool isValidWordListName(const std::string& name) {
    if (name.empty() || name.size() > 64 || !std::isalnum((unsigned char) name[0]))
        return false;
    for (unsigned char c : name) {
        if (!std::isalnum(c) && c != '.' && c != '-' && c != '_')
            return false;
    }
    return true;
}

bool resolveClaimWords(const ClaimResponse& claim, const std::string& cacheDir, const WordListDownloader& download,
                       std::vector<std::string>& words, std::vector<std::string>& downloaded, std::string& error) {
    words.clear();
    downloaded.clear();
    if (claim.wordLists.empty() || claim.wordLists.size() != claim.wordListChecksums.size()) {
        error = "the claim names no word lists, or not a checksum for each";
        return false;
    }
    for (size_t i = 0; i < claim.wordLists.size(); ++i) {
        const std::string& name = claim.wordLists[i];
        const std::string& expected = claim.wordListChecksums[i];
        std::vector<std::string> listWords;
        if (isBuiltinWordList(name)) {
            listWords = builtinWordList(name);
            const std::string checksum = hex64(wordListChecksum(sortedUniqueWords(listWords)));
            if (checksum != expected) {
                error = "the server's word list " + name + " (checksum " + expected + ") isn't the one compiled into this client (" + checksum + ")";
                return false;
            }
        } else {
            if (!isValidWordListName(name)) {
                error = "the server named a word list '" + name + "', which a word list can't be named";
                return false;
            }
            const std::string path = (std::filesystem::path(cacheDir) / (name + ".txt")).string();
            std::string text;
            if (!readFile(path, text) || checksumOfList(text, path, listWords) != expected) {
                listWords.clear();
                if (!download(name, text, error))
                    return false;
                const std::string checksum = checksumOfList(text, name, listWords);
                if (checksum != expected) {
                    error = "the word list " + name + " the server sent has checksum " + checksum + ", not the " + expected + " it said it has";
                    return false;
                }
                std::error_code ec;
                std::filesystem::create_directories(cacheDir, ec);
                std::string writeError;
                if (ec || !replaceFileContents(path, text, writeError))
                    fprintf(stderr, "[coordinator] warning: can't keep word list %s in %s (%s) - it'll be downloaded again next time\n", name.c_str(),
                            path.c_str(), ec ? ec.message().c_str() : writeError.c_str());
                downloaded.push_back(name);
            }
        }
        words.insert(words.end(), listWords.begin(), listWords.end());
    }
    words = sortedUniqueWords(std::move(words));
    const std::string checksum = hex64(wordListChecksum(words));
    if (checksum != claim.wordsChecksum) {
        error = "the word lists' words together have checksum " + checksum + ", not the server's " + claim.wordsChecksum;
        return false;
    }
    return true;
}
