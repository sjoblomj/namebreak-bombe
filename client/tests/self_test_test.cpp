// Checks that the backend self-test (src/backends/self_test.h) - which
// createBackend runs on every backend before handing it out - actually
// catches a backend that searches wrongly. Wraps a real, working backend
// (`reference`, and `cpu`) in ones that each break it in a way a buggy GPU
// kernel, driver or compiler could: dropping hits whose last character is
// past the 32nd (a 32-bit mask), dropping hits in a range's cut-short first
// row, reporting a candidate outside the range, or never reporting a
// both-hashes match. Each must fail the self-test; the unbroken backends,
// and a wrapper that passes everything through, must pass it. Pure CPU.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

#include "backends/backends.h"
#include "backends/self_test.h"

namespace {

enum class Breakage { None, DropHighCharacters, DropFirstPartialRow, ExtraHitOutsideRange, NeverFound };

class BrokenBackend : public SearchBackend {
public:
    BrokenBackend(std::unique_ptr<SearchBackend> inner, Breakage breakage) : inner_(std::move(inner)), breakage_(breakage) {}

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

private:
    std::unique_ptr<SearchBackend> inner_;
    Breakage breakage_;
    SearchConstants constants_;
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
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
