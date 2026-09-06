-- Targets are now bounded by actual candidate strings (e.g. "BLACKSMITH" to
-- "CATAPULT"), not raw length numbers: the server always starts at the very
-- beginning implied by lower_bound and always searches as long as the
-- alphabet allows (see max_supported_len), tightening the search to exactly
-- the given bounds at every candidate length in between - not just at the
-- bounds' own literal lengths (see alphabet::bound_indices_at_len).
--
-- Any target rows that already exist get backfilled from their old
-- min_len/max_len (+ their own alphabet's first/last character) *before*
-- those columns are dropped, so an already-deployed target keeps searching
-- exactly the same space it did before this migration - "full space from
-- min_len to max_len" is exactly what
-- (alphabet_min_char * min_len) .. (alphabet_max_char * max_len) means.
-- The 'hex(zeroblob(n))' / 'replace' pair is a standard SQLite idiom for
-- building a string of n repeated characters (SQLite has no REPEAT()).
ALTER TABLE targets ADD COLUMN lower_bound TEXT NOT NULL DEFAULT '';
ALTER TABLE targets ADD COLUMN upper_bound TEXT NOT NULL DEFAULT '';

UPDATE targets SET
  lower_bound = substr(replace(hex(zeroblob(min_len)), '00', substr(alphabet, 1, 1)), 1, min_len),
  upper_bound = substr(replace(hex(zeroblob(max_len)), '00', substr(alphabet, -1, 1)), 1, max_len);

ALTER TABLE targets DROP COLUMN min_len;
ALTER TABLE targets DROP COLUMN max_len;
