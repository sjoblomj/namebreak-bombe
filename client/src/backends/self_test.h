#ifndef NAMEBREAK_BACKENDS_SELF_TEST_H
#define NAMEBREAK_BACKENDS_SELF_TEST_H

#include <string>

#include "engine/backend.h"

// A known-answer test of `backend` on the machine it's about to search on:
// a handful of small batches, each with a candidate planted as the target
// (both hashes), that must be found - at the first and last last-character
// positions, on both sides of bit 32 of a row's mask, in rows cut short at
// both ends, as the very first and very last candidate of a range, in the
// first and last row of a row group, with long, empty and non-ASCII suffixes
// and prefixes - plus targets planted just
// outside a batch's range, which must not be. Every hit reported must also
// genuinely match.
//
// The test suite only ever runs on the machines it's run on; this runs
// wherever namebreak does, so a GPU, driver or compiler that gets the search
// wrong in a way the tests never saw is caught before it can silently miss
// a match. createBackend runs it on every backend it hands out. Takes a few
// tens of milliseconds. False, with `error` saying what went wrong, if the
// backend fails it.
bool selfTestBackend(SearchBackend& backend, std::string& error);

// The same for a dictionary search (engine/dictionary_search.h), on a
// backend that supportsDictionary(): calls with a candidate planted as the
// target (both hashes, and its basename as the encryption key) that must be
// found - as the first, last and only word of a batch, on both sides of
// where a GPU backend splits a batch's words between thread blocks, in the
// first, a middle and the last of many batches searched at once, in a batch
// of every word and in batches of some of them, with words of every length,
// words with a '\' and with characters past ASCII, and with no, short,
// long and very long suffixes, and one with a '\' - plus words planted just
// outside a batch, which must not be, and more basename matches than a GPU
// backend has room for at first, every one of which must be. Every hit
// reported must also genuinely match.
//
// Separate from selfTestBackend, as a GPU backend that compiles its kernels
// at runtime (OpenCL, Metal) compiles several here, which a search that
// isn't a dictionary search shouldn't wait for: createDictionaryBackend
// (backends.h) runs both. False, with `error` saying what went wrong, if the
// backend fails it.
bool selfTestDictionaryBackend(SearchBackend& backend, std::string& error);

// The same for a meet-in-the-middle search
// (engine/meet_in_the_middle_search.h), on a backend that
// supportsMeetInTheMiddle(): small searches with a directory planted as the
// one the meeting point belongs to, which must be found - as the first, last
// and only tail of a batch, with tails of every length from none (the whole
// directory in the table) to the whole directory, with a '\' in the directory
// and in the alphabet, with and without a prefix - plus tails planted just
// outside a batch, and tails the pruning rules reject, which must not be.
// Every batch's hits and the number of tails it looked up are compared
// against a plain loop over the same tails, so a backend that rewinds the
// hash wrongly, skips the wrong tails or mixes up a tail's index is caught
// here.
//
// Separate from selfTestBackend, as selfTestDictionaryBackend is:
// createMeetInTheMiddleBackend (backends.h) runs both. False, with `error`
// saying what went wrong, if the backend fails it.
bool selfTestMeetInTheMiddleBackend(SearchBackend& backend, std::string& error);

#endif // NAMEBREAK_BACKENDS_SELF_TEST_H
