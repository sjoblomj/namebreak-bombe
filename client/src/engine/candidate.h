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
