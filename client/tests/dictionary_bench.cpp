// Times a real dictionary search (runDictionarySearch, engine/
// dictionary_search.h) on one backend: two words of english-1 - one word
// and two, four separators, about 16.3 billion candidates - after MUSIC\BG,
// with .WAV after them, an unreachable target and an encryption key, so
// that it runs to the end and hashes every candidate's basename - rather
// than its hashA - as a search for a file's name usually does.
// PERFORMANCE.md's "Dictionary searches" has what it measured.
//
//   dictionary_bench [--backend <name>] [--words <n>] [--max-words <n>] [--no-key | --record-hasha-matches]
//                    [--tails <element>,...]
//
// --words takes the first <n> words of english-1 only (to time a slower
// backend in a reasonable time - the CPU backend takes minutes for all of
// them); --max-words searches up to <n> words a candidate (default 2);
// --no-key searches without the encryption key, hashing every candidate's
// hashA instead; --record-hasha-matches hashes both (record_hasha_matches);
// --tails gives every word tails, e.g. --tails digits:1-2,letters:0-1 (see
// expandDictionaryTails).
// Creating the backend (for CUDA, the GPU context, and the self-tests) and
// a first, small search happen before the clock starts. Writes its files
// under ./dictionary_bench/.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include "backends/backends.h"
#include "engine/dictionary_search.h"
#include "engine/mpq_hash.h"
#include "engine/wordlist.h"

int main(int argc, char** argv) {
    std::string backendName;
    size_t wordLimit = 0;
    int maxWords = 2;
    bool key = true, everyHashA = false;
    std::vector<std::string> tailElements;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backendName = argv[++i];
        } else if (arg == "--words" && i + 1 < argc) {
            wordLimit = std::stoul(argv[++i]);
        } else if (arg == "--max-words" && i + 1 < argc) {
            maxWords = std::stoi(argv[++i]);
        } else if (arg == "--no-key") {
            key = false;
        } else if (arg == "--record-hasha-matches") {
            everyHashA = true;
        } else if (arg == "--tails" && i + 1 < argc) {
            const std::string list = argv[++i];
            for (size_t start = 0; start <= list.size();) {
                const size_t comma = std::min(list.find(',', start), list.size());
                tailElements.push_back(list.substr(start, comma - start));
                start = comma + 1;
            }
        } else {
            fprintf(stderr, "usage: %s [--backend <name>] [--words <n>] [--max-words <n>] [--no-key | --record-hasha-matches] [--tails <element>,...]\n",
                    argv[0]);
            return 1;
        }
    }
    std::string error;
    std::unique_ptr<SearchBackend> backend = createDictionaryBackend(backendName, error);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::vector<std::string> words = sortedUniqueWords(builtinWordList("english-1"));
    if (wordLimit && wordLimit < words.size())
        words.resize(wordLimit);

    const std::string dir = "dictionary_bench";
    std::filesystem::remove_all(dir);
    DictionaryRequest req;
    req.pattern.words = words;
    req.pattern.separators = {"", "_", "-", " "};
    req.pattern.minWords = 1;
    req.pattern.maxWords = maxWords;
    req.prefix = "MUSIC\\BG";
    req.suffix = ".WAV";
    req.targetHashA = 0x216A81D3;
    req.targetHashB = 0;
    req.checkBasename = key;
    req.basenameKey = key ? 0x1D5AD26C : 0;
    req.recordHashAMatches = everyHashA;
    if (!expandDictionaryTails(tailElements, req.pattern.tails, error)) {
        fprintf(stderr, "--tails: %s\n", error.c_str());
        return 1;
    }
    req.wordSource = "english-1";
    req.outputFilePath = dir + "/matches.txt";
    req.basenamesFilePath = dir + "/basenames.txt";
    req.progressFilePath = dir + "/wordnumber.txt";
    req.statusInterval = std::chrono::hours(1);

    // A warm-up: the first thousand words, one a candidate.
    DictionaryRequest warmUp = req;
    warmUp.pattern.words.resize(std::min<size_t>(1000, words.size()));
    warmUp.pattern.maxWords = 1;
    const DictionaryResult warm = runDictionarySearch(*backend, warmUp);
    if (!warm.ok) {
        fprintf(stderr, "%s\n", warm.error.c_str());
        return 1;
    }
    std::filesystem::remove_all(dir);

    const auto start = std::chrono::steady_clock::now();
    const DictionaryResult result = runDictionarySearch(*backend, req);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!result.ok) {
        fprintf(stderr, "%s\n", result.error.c_str());
        return 1;
    }
    std::filesystem::remove_all(dir);
    std::string tails = req.pattern.tails.size() > 1 ? ", " + std::to_string(req.pattern.tails.size()) + " tails" : "";
    printf("\n%s: %llu candidates in %.2f s - %.2f G candidates/s (%zu words, up to %d a candidate%s, %s)\n", backend->name(),
           (unsigned long long) result.candidatesSearched, seconds, (double) result.candidatesSearched / seconds / 1e9, words.size(), maxWords,
           tails.c_str(), !key ? "without a key" : everyHashA ? "with the key, every hashA hit" : "with the key");
    return 0;
}
