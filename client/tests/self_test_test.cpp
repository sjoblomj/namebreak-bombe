// Checks that the backend self-tests (src/backends/self_test.h) - which
// createBackend and createDictionaryBackend run on every backend before
// handing it out - actually catch a backend that searches wrongly. Wraps a
// real, working backend (`reference`, and `cpu`) in ones that each break it
// in a way a buggy GPU kernel, driver or compiler could: dropping hits whose
// last character is past the 32nd (a 32-bit mask), dropping hits in a
// range's cut-short first row, reporting a candidate outside the range, or
// never reporting a both-hashes match. Each must fail the self-test; the
// unbroken backends, and a wrapper that passes everything through, must pass
// it. And the same for a dictionary search: dropping a batch's last word,
// the words a GPU kernel's first thread block of a batch doesn't reach, words
// with characters past ASCII, every hit with a long suffix, the basename
// matches past the first 4,096 or all of them, reporting the word just past
// a batch, the one just before it as a basename match or basename matches
// it wasn't asked for, or never reporting a both-hashes match. Pure CPU.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/backends.h"
#include "backends/self_test.h"

namespace {

enum class Breakage { None, DropHighCharacters, DropFirstPartialRow, ExtraHitOutsideRange, NeverFound };
enum class DictionaryBreakage {
    None, DropLastWord, DropPastFirstBlock, DropNonAscii, DropLongSuffix, KeepFirstBasenames, ExtraWordPastBatch, NeverFound,
    NoBasenames, UnaskedBasenames, ExtraBasename
};

class BrokenBackend : public SearchBackend {
public:
    BrokenBackend(std::unique_ptr<SearchBackend> inner, Breakage breakage, DictionaryBreakage dictionaryBreakage = DictionaryBreakage::None)
        : inner_(std::move(inner)), breakage_(breakage), dictionaryBreakage_(dictionaryBreakage) {}

    const char* name() const override { return inner_->name(); }
    int windowChars() const override { return inner_->windowChars(); }
    int maxTrailingLen() const override { return inner_->maxTrailingLen(); }
    uint64_t batchSize(int alphabetSize) const override { return inner_->batchSize(alphabetSize); }
    void beginSearch(const SearchConstants& constants) override {
        constants_ = constants;
        inner_->beginSearch(constants);
    }
    void endSearch() override { inner_->endSearch(); }

    BatchOutcome runBatch(int trailingLen, uint64_t start, uint64_t count, const BatchParams& params) override {
        const uint64_t as = constants_.alphabet.size();
        if (breakage_ == Breakage::DropFirstPartialRow && start % as != 0) {
            // Skip ahead to the next whole row, as a kernel that mishandles a
            // cut-short first row might.
            const uint64_t skip = std::min(count, as - start % as);
            if (skip == count)
                return BatchOutcome{};
            start += skip;
            count -= skip;
        }
        BatchOutcome outcome = inner_->runBatch(trailingLen, start, count, params);
        if (breakage_ == Breakage::DropHighCharacters) {
            const size_t lastCharAt = (size_t) params.prefixSize + trailingLen - 1;
            auto high = [&](const std::string& hit) { return constants_.alphabet.find(hit[lastCharAt]) >= 32; };
            outcome.hits.erase(std::remove_if(outcome.hits.begin(), outcome.hits.end(), high), outcome.hits.end());
            if (outcome.found && high(outcome.foundFilename)) {
                outcome.found = false;
                outcome.foundFilename.clear();
            }
        } else if (breakage_ == Breakage::ExtraHitOutsideRange) {
            // Also report the candidate just past the range's end - as a
            // kernel that runs one past its last row's end might, if it
            // happens to match.
            const uint64_t past = start + count;
            std::string trailing(trailingLen, constants_.alphabet[0]);
            uint64_t index = past;
            for (int i = trailingLen - 1; i >= 0; --i) {
                trailing[i] = constants_.alphabet[index % as];
                index /= as;
            }
            outcome.hits.push_back(std::string(params.prefix, params.prefixSize) + trailing + constants_.suffix);
            ++outcome.hitCount;
        } else if (breakage_ == Breakage::NeverFound) {
            outcome.found = false;
            outcome.foundFilename.clear();
        }
        return outcome;
    }

    bool supportsDictionary() const override { return inner_->supportsDictionary(); }
    void beginDictionarySearch(const DictionaryConstants& constants) override {
        dictionary_ = constants;
        inner_->beginDictionarySearch(constants);
    }
    void endDictionarySearch() override { inner_->endDictionarySearch(); }

    DictionaryOutcome runDictionaryBatches(const std::vector<DictionaryBatch>& batches) override {
        DictionaryOutcome outcome = inner_->runDictionaryBatches(batches);
        const std::string& suffix = dictionary_.suffix;
        // The batch and word a reported filename is - (-1, 0) if none.
        auto locate = [&](const std::string& filename) -> std::pair<int, uint32_t> {
            for (size_t b = 0; b < batches.size(); ++b) {
                const std::string& leading = batches[b].leading;
                if (filename.size() < leading.size() + suffix.size() || filename.compare(0, leading.size(), leading) != 0)
                    continue;
                const std::string word = filename.substr(leading.size(), filename.size() - leading.size() - suffix.size());
                const auto it = std::lower_bound(dictionary_.words.begin(), dictionary_.words.end(), word);
                const uint32_t w = (uint32_t) (it - dictionary_.words.begin());
                if (it != dictionary_.words.end() && *it == word && w - batches[b].firstWord < batches[b].wordCount)
                    return {(int) b, w};
            }
            return {-1, 0};
        };
        auto dropped = [&](const std::string& filename) {
            const std::pair<int, uint32_t> at = locate(filename);
            if (at.first < 0)
                return false;
            const DictionaryBatch& batch = batches[at.first];
            const std::string& word = dictionary_.words[at.second];
            switch (dictionaryBreakage_) {
                case DictionaryBreakage::DropLastWord: return at.second == batch.firstWord + batch.wordCount - 1;
                // As a GPU kernel that searched only a batch's first thread
                // block's words (see dictionaryKernel in cuda_backend.cu).
                case DictionaryBreakage::DropPastFirstBlock: return at.second - batch.firstWord >= 8192;
                case DictionaryBreakage::DropNonAscii:
                    return std::any_of(word.begin(), word.end(), [](char c) { return (unsigned char) c >= 0x80; });
                case DictionaryBreakage::DropLongSuffix: return suffix.size() > 12;
                default: return false;
            }
        };
        auto drop = [&](std::vector<std::string>& hits) { hits.erase(std::remove_if(hits.begin(), hits.end(), dropped), hits.end()); };
        drop(outcome.hits);
        drop(outcome.basenameHits);
        if (outcome.found && dropped(outcome.foundFilename)) {
            outcome.found = false;
            outcome.foundFilename.clear();
        }
        if (dictionaryBreakage_ == DictionaryBreakage::KeepFirstBasenames && outcome.basenameHits.size() > 4096) {
            // As a GPU backend that didn't search a launch again when it had
            // more hits than room for them.
            outcome.basenameHits.resize(4096);
        } else if (dictionaryBreakage_ == DictionaryBreakage::ExtraWordPastBatch) {
            // The word just past each batch, as a hashA hit - as a kernel that
            // ran one past a batch's end might report, if it happened to match.
            for (const DictionaryBatch& batch : batches) {
                if (batch.firstWord + batch.wordCount < dictionary_.words.size())
                    outcome.hits.push_back(batch.leading + dictionary_.words[batch.firstWord + batch.wordCount] + suffix);
            }
        } else if (dictionaryBreakage_ == DictionaryBreakage::NeverFound) {
            outcome.found = false;
            outcome.foundFilename.clear();
        } else if (dictionaryBreakage_ == DictionaryBreakage::NoBasenames) {
            // As a kernel that never compared the basename hash.
            outcome.basenameHits.clear();
        } else if (dictionaryBreakage_ == DictionaryBreakage::UnaskedBasenames && !dictionary_.checkBasename && !batches.empty()) {
            // As a kernel that compared it whether asked to or not.
            outcome.basenameHits.push_back(batches[0].leading + dictionary_.words[batches[0].firstWord] + suffix);
        } else if (dictionaryBreakage_ == DictionaryBreakage::ExtraBasename && !batches.empty() && batches[0].firstWord > 0) {
            // The word just before a batch, as a basename hit - as a kernel
            // that started a batch one word early might report.
            outcome.basenameHits.push_back(batches[0].leading + dictionary_.words[batches[0].firstWord - 1] + suffix);
        }
        return outcome;
    }

private:
    std::unique_ptr<SearchBackend> inner_;
    Breakage breakage_;
    DictionaryBreakage dictionaryBreakage_;
    SearchConstants constants_;
    DictionaryConstants dictionary_;
};

} // namespace

int main() {
    int failures = 0;
    for (const char* innerName : {"reference", "cpu"}) {
        struct Variant { Breakage breakage; const char* what; };
        const Variant variants[] = {
            {Breakage::None, "passes everything through"},
            {Breakage::DropHighCharacters, "drops hits whose last character is past the 32nd"},
            {Breakage::DropFirstPartialRow, "drops a range's cut-short first row"},
            {Breakage::ExtraHitOutsideRange, "reports a candidate just past the range"},
            {Breakage::NeverFound, "never reports a both-hashes match"},
        };
        for (const Variant& v : variants) {
            std::string error;
            std::unique_ptr<SearchBackend> inner = createBackend(innerName, error);
            if (!inner) {
                fprintf(stderr, "FAILED: the %s backend itself: %s\n", innerName, error.c_str());
                ++failures;
                continue;
            }
            BrokenBackend backend(std::move(inner), v.breakage);
            std::string why;
            const bool passed = selfTestBackend(backend, why);
            const bool shouldPass = v.breakage == Breakage::None;
            printf("%-9s that %-50s -> %s%s%s\n", innerName, v.what, passed ? "passes" : "fails: ", passed ? "" : why.c_str(),
                   passed == shouldPass ? "" : "   <-- WRONG");
            if (passed != shouldPass)
                ++failures;
        }
    }
    for (const char* innerName : {"reference", "cpu"}) {
        struct Variant { DictionaryBreakage breakage; const char* what; };
        const Variant variants[] = {
            {DictionaryBreakage::None, "passes everything through"},
            {DictionaryBreakage::DropLastWord, "drops hits of a batch's last word"},
            {DictionaryBreakage::DropPastFirstBlock, "drops hits past a batch's 8,192nd word"},
            {DictionaryBreakage::DropNonAscii, "drops hits of words with characters past ASCII"},
            {DictionaryBreakage::DropLongSuffix, "drops every hit with a suffix longer than 12"},
            {DictionaryBreakage::KeepFirstBasenames, "reports 4,096 basename matches at most"},
            {DictionaryBreakage::ExtraWordPastBatch, "reports the word just past a batch"},
            {DictionaryBreakage::NeverFound, "never reports a both-hashes match"},
            {DictionaryBreakage::NoBasenames, "never reports a basename match"},
            {DictionaryBreakage::UnaskedBasenames, "reports basename matches it wasn't asked for"},
            {DictionaryBreakage::ExtraBasename, "reports the word just before a batch as a basename match"},
        };
        for (const Variant& v : variants) {
            std::string error;
            std::unique_ptr<SearchBackend> inner = createDictionaryBackend(innerName, error);
            if (!inner) {
                fprintf(stderr, "FAILED: the %s backend itself: %s\n", innerName, error.c_str());
                ++failures;
                continue;
            }
            BrokenBackend backend(std::move(inner), Breakage::None, v.breakage);
            std::string why;
            const bool passed = selfTestDictionaryBackend(backend, why);
            const bool shouldPass = v.breakage == DictionaryBreakage::None;
            printf("%-9s dictionary search that %-48s -> %s%s%s\n", innerName, v.what, passed ? "passes" : "fails: ", passed ? "" : why.c_str(),
                   passed == shouldPass ? "" : "   <-- WRONG");
            if (passed != shouldPass)
                ++failures;
        }
    }
    {
        // A backend that can't search dictionaries fails it, saying so.
        class RowsOnly : public SearchBackend {
        public:
            const char* name() const override { return "rows-only"; }
            int windowChars() const override { return 1; }
            int maxTrailingLen() const override { return 1; }
            uint64_t batchSize(int) const override { return 1; }
            void beginSearch(const SearchConstants&) override {}
            BatchOutcome runBatch(int, uint64_t, uint64_t, const BatchParams&) override { return {}; }
            void endSearch() override {}
        } rowsOnly;
        std::string why;
        const bool passed = selfTestDictionaryBackend(rowsOnly, why);
        const bool ok = !passed && why.find("can't search dictionaries") != std::string::npos;
        printf("a backend that can't search dictionaries -> %s%s\n", passed ? "passes" : why.c_str(), ok ? "" : "   <-- WRONG");
        if (!ok)
            ++failures;
    }
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
