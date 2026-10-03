#ifndef NAMEBREAK_ENGINE_CANDIDATE_H
#define NAMEBREAK_ENGINE_CANDIDATE_H

#include <cstdint>
#include <string>
#include <string_view>

// Candidates as strings over an alphabet: converting between a candidate and
// its index in enumeration order, the bounds a search runs between, and the
// cheap rules that skip implausible candidates.
//
// Terminology:
// * Candidate = The part of the name that we are brute-forcing
// * Filename  = The Prefix + Candidate + Suffix

// CPU-side pruning, applied to the *leading* characters of a candidate
// (runSearch's leadingIdx loop, cuda_backend.cu) - the GPU-brute-forced trailing
// characters are never checked at all. See README.md's "Design decisions"
// section for why.
//
// Take string_view rather than a string - a caller with just a substring to
// check (e.g. a candidate's leading portion) shouldn't have to allocate one
// to ask.
bool hasForbiddenSymbolRun_CPU(std::string_view s);
int countBackslashes_CPU(std::string_view s);

// Brackets: round ( ) and square [ ] are counted separately - a ')' only
// ever closes a '(', and a ']' only ever closes a '['. How the two kinds nest
// within each other isn't checked, so "[(])" passes.
struct OpenBrackets {
    int round = 0;
    int square = 0;
};
// How many brackets of each kind `s` leaves open. A closer with nothing of
// its kind open is ignored rather than driving that count negative - meant
// for a search's fixed prefix, whose own stray closers aren't any
// candidate's fault.
OpenBrackets openBracketsAfter_CPU(std::string_view s);
// True if `s` closes a bracket it never opened: at some point, more ')' than
// '(' (or more ']' than '[') have been seen, counting `openBefore` brackets
// already open when `s` starts (see openBracketsAfter_CPU). Checking a
// candidate's leading characters this way is sound regardless of what
// follows them - once a count has dropped below zero, no later character can
// undo that.
bool hasUnopenedBracket_CPU(std::string_view s, OpenBrackets openBefore);

// The rules as a state machine, one character at a time - how the engine
// checks a candidate's leading characters, and how a backend prunes its
// trailing ones when a search prunes the whole candidate
// (SearchRequest::pruneWholeCandidate). Stepping over a string gives the
// same verdict as the functions above: the symbol run starts at the
// candidate's first character, backslashes are counted from it, and
// brackets start from those the prefix leaves open. Two backslashes next to
// each other include one the prefix ends with.
struct PruneRules {
    bool symbolRuns = false;          // hasForbiddenSymbolRun_CPU
    bool unopenedBrackets = false;    // hasUnopenedBracket_CPU
    int maxBackslashCount = 0;        // countBackslashes_CPU; 0 means unlimited
    int minBackslashCount = 0;        // canReachMinBackslashes_CPU; 0 means none needed
    bool adjacentBackslashes = false; // no "\\" anywhere
    bool any() const { return symbolRuns || unopenedBrackets || maxBackslashCount != 0 || minBackslashCount != 0 || adjacentBackslashes; }
};
// What the rules need to know about the characters so far.
struct PruneState {
    int symbolRun = 0; // how many non-alphanumeric, non-space characters in a row, up to 2
    OpenBrackets open;
    int backslashes = 0;
    bool lastWasBackslash = false;
};
// Steps `state` over `c`; false if `c` breaks one of `rules` (`state` is
// then no longer meaningful).
bool pruneStep_CPU(const PruneRules& rules, PruneState& state, char c);
// Whether a candidate in `state` can still have rules.minBackslashCount
// backslashes, with `remaining` characters still to come - as many of them
// backslashes as can be (every other one, with rules.adjacentBackslashes).
// Unlike pruneStep_CPU's rules, this one is about characters not checked:
// it's asked once, after the last character that is. Once false, it stays
// false for every character added, so asking it after fewer characters
// prunes no more than asking it after more.
bool canReachMinBackslashes_CPU(const PruneRules& rules, const PruneState& state, int remaining);
// Whether a character counts towards a symbol run - neither alphanumeric
// (A-Z, 0-9) nor a space.
bool isRunSymbol_CPU(char c);

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

// Returns false (with `error` set) if `path` doesn't start with `prefix` and
// end with `suffix`; otherwise fills `out` with the candidate portion between
// them (empty if `path` is just prefix + suffix) and returns true.
bool getStartCandidate(const std::string& path, const std::string& prefix, const std::string& suffix, std::string& out, std::string& error);

// `input` extended to `candidateLen` characters by repeating its last
// character (or truncated to it, if longer).
std::string makeBoundString(std::string input, int candidateLen);
// `base` with a leading `prefix` and a trailing `suffix` removed, where present.
std::string removePrefixAndSuffix(std::string base, std::string prefix, std::string suffix);

// Returns false (with `error` set) if `input` contains a character not in
// `alphabet`; otherwise fills `out` per getLowerBound/getUpperBound's own
// doc comments (in candidate.cpp) and returns true.
bool getLowerBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error);
bool getUpperBound(const std::string& input, const std::string& alphabet, std::string& out, std::string& error);

#endif // NAMEBREAK_ENGINE_CANDIDATE_H
