#ifndef NAMEBREAK_ENGINE_LIMITS_H
#define NAMEBREAK_ENGINE_LIMITS_H

// Limits on what a search can be asked to do - the same for every backend,
// which size their buffers from these. Each backend's own tuning values
// (e.g. backends/cuda/tuning.h) live with that backend.

// Upper bound on the alphabet's character count. Must be >= the largest size
// any backend supports.
// 63: a row's candidates, one bit per last character, and one past the last
// of them must fit a 64-bit mask (see the backends' row masks and the lookup
// filter's table).
#define MAX_ALPHABET_SIZE 63
#define MAX_CANDIDATE_LEN 16
#define MAX_FILENAME_LEN 128
// How many hashA hits one batch can record. More than this in a single batch
// is handled by re-searching that batch's range in halves (see searchChunk in
// search.cpp), so it only bounds a buffer, not correctness. Overridable at
// compile time (-DMAX_MATCHES=N) so tests/search_overflow_test.cpp can
// exercise that path with just a couple of colliding candidates instead of
// needing >1024 of them. Must be >= 1.
#ifndef MAX_MATCHES
#define MAX_MATCHES 1024
#endif

// Capacity of BatchParams::prefix (backend.h) - the prefix extended by the
// candidate's leading characters - including the terminating NUL.
constexpr int kMaxPrefixSize = 64;
// Capacity of a backend's copy of the suffix, including the terminating NUL.
constexpr int kMaxSuffixSize = 64;

static_assert(MAX_MATCHES >= 1, "MAX_MATCHES must be at least 1 (a one-candidate range can have one hit)");

#endif // NAMEBREAK_ENGINE_LIMITS_H
