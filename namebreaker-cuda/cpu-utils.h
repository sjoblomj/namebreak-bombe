#ifndef NAMEBREAK_CUDA_CPU_UTILS_H
#define NAMEBREAK_CUDA_CPU_UTILS_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

std::pair<uint32_t, uint32_t > mpqHashWithPrefixCache_CPU(const char* str, const uint32_t* cryptTable);

// CPU-side pruning, applied to the *leading* characters of a candidate
// (runSearch's leadingIdx loop, namebreak.cu) - the GPU-brute-forced trailing
// characters are never checked at all. See the leadingIdx loop's doc comment
// for why: in short, these are cheap heuristics for skipping implausible
// candidates, not correctness rules, and checking them (or even just a
// narrow, warp-friendly slice of them) on the GPU never paid for itself once
// gpuWindowChars made the trailing window small (measured, more than once:
// no throughput benefit, consistently a net loss, from SIMT lockstep and
// fixed per-thread overhead - a `return` only saves time if it takes a whole
// 32-lane warp with it, and even then the check itself is paid by every
// thread regardless). Checking the leading characters on the CPU instead is
// different: skipping an entire leading value here skips its whole trailing
// batch (up to batchSize candidates) before ever asking the GPU to do
// anything, with no SIMT caveat, since this is one thread deciding whether
// to launch a kernel at all - not GPU lanes deciding whether to keep
// computing mid-kernel.
//
bool hasForbiddenSymbolRun_CPU(const std::string& s);
int countBackslashes_CPU(const std::string& s);

// Incrementally hashes a fixed base state (typically the hash of a fixed
// filename prefix) extended by a variable-length "leading" string of
// `leadingLen` alphabet characters, walked in the same odometer order as
// indexToString/stringToIndex (last character fastest-changing, like an
// odometer's rightmost digit).
//
// advance() moves to the next leading value the same way runSearch's
// leadingIdx loop does (always exactly +1), but instead of re-hashing the
// whole leading string from scratch each time (what calling
// mpqHashWithPrefixCache_CPU(prefix + indexToString(leadingIdx, ...))  every
// iteration amounts to - leadingLen+ hash steps every single time), it keeps
// a stack of the hash state after each prefix length (0 characters of
// leading, 1 character, 2, ...) and only re-hashes from the lowest position
// that actually changed - the same amortized-O(1)-per-step argument as
// incrementing a mixed-radix counter (position p only changes every
// alphabet.size()^p increments, so summed over N increments the total work
// is O(N), not O(N * leadingLen)). Only reset() ever pays the full
// O(leadingLen) cost, once, to seed a specific starting value.
class IncrementalPrefixHasher {
public:
    // `baseState` is the hash state of the fixed part alone (e.g.
    // mpqHashWithPrefixCache_CPU(prefix, cryptTable)) - what advance()/reset()
    // extend by the leading characters. `alphabet` and `cryptTable` must
    // outlive this object; both are copied/referenced as given elsewhere in
    // this codebase (small, process-lifetime data).
    IncrementalPrefixHasher(std::pair<uint32_t, uint32_t> baseState, int leadingLen,
                             std::string alphabet, const uint32_t* cryptTable);

    // One-time full hash to (re)start at a specific leading value - the only
    // O(leadingLen) operation this class performs. Must be called once
    // before the first advance().
    void reset(uint64_t leadingIdx);

    // Moves to the next leading value (leadingIdx + 1) - O(1) amortized.
    // Undefined which leading value results if called before reset(), or
    // past the last representable value (alphabet.size()^leadingLen - 1);
    // callers own bounding the loop, exactly as the leadingIdx loop already
    // does via startLeadingIdx/endLeadingIdx.
    void advance();

    // The current leading characters (leadingLen of them).
    const std::string& leading() const { return leading_; }

    // Hash state after the fixed part followed by the current leading value -
    // what mpqHashWithPrefixCache_CPU(prefix + leading()) would return.
    std::pair<uint32_t, uint32_t> state() const { return stack_.back(); }

private:
    std::string alphabet_;
    const uint32_t* cryptTable_;
    int leadingLen_;
    std::string leading_;
    std::vector<int> digitIndex_;
    // stack_[k] = hash state after the fixed part + leading_[0..k-1];
    // stack_[0] = baseState, stack_[leadingLen_] = state().
    std::vector<std::pair<uint32_t, uint32_t>> stack_;
};

// Below: every function that can encounter a character outside `alphabet`
// reports that as a plain false/error-string return instead of exit()ing the
// process - this code runs from a long-lived coordinator daemon (once per
// claimed range), where a single malformed range must not take the whole
// process down with it. Callers that only ever run once at startup (config
// parsing) still get the same fail-fast behavior; they just do it via their
// own existing `return false` error path instead.

// Returns false (with `error` set) if `str` contains a character not in
// `alphabet`; otherwise fills `out` with str's numeric value in that
// alphabet's base and returns true.
bool stringToIndex(const std::string& str, const std::string& alphabet, uint64_t& out, std::string& error);

std::string indexToString(uint64_t index, int len, const std::string& alphabet);

// Returns false (with `error` set) if `a` or `b` contains a character not in
// `alphabet`; otherwise fills `outIsBefore` with whether `a` sorts before `b`
// and returns true.
bool isBeforeInAlphabet(const std::string& a, const std::string& b, const std::string& alphabet, bool& outIsBefore, std::string& error);

void prepareCryptTable(uint32_t* table);

// Returns false (with `error` set) if `path` doesn't start with `prefix` or
// isn't long enough to also hold `suffix`; otherwise fills `out` with the
// candidate portion and returns true.
bool getStartCandidate(const std::string& path, const std::string& prefix, const std::string& suffix, std::string& out, std::string& error);

std::string make_bound_string(std::string input, int candidateLen);
std::string remove_prefix_and_suffix(std::string base, std::string prefix, std::string suffix);

// Returns false (with `error` set) if `input` contains a character not in
// `alphabet`; otherwise fills `out` per getLowerBound/getUpperBound's own
// doc comments (in cpu-utils.cpp) and returns true.
bool getLowerBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error);
bool getUpperBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error);

// Parses a hex string into a uint32_t - tolerates (but doesn't require) a
// leading "0x"/"0X", matching how target hashes are written everywhere in
// this project (CLI args, config.conf, the coordinator's JSON). Returns
// false (rather than throwing) on anything that doesn't parse.
bool hexToU32(const std::string& s, uint32_t& out);

#endif //NAMEBREAK_CUDA_CPU_UTILS_H
