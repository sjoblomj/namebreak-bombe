#ifndef NAMEBREAK_BACKENDS_SELF_TEST_H
#define NAMEBREAK_BACKENDS_SELF_TEST_H

#include <string>

#include "engine/backend.h"

// A known-answer test of `backend` on the machine it's about to search on:
// a handful of small batches, each with a candidate planted as the target
// (both hashes), that must be found - at the first and last last-character
// positions, on both sides of bit 32 of a row's mask, in rows cut short at
// both ends, as the very first and very last candidate of a range, with
// long, empty and non-ASCII suffixes and prefixes - plus targets planted just
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

#endif // NAMEBREAK_BACKENDS_SELF_TEST_H
