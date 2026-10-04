-- Automatic priority ranges - see AdminCreateTargetRequest::auto_priority
-- and ranges::create_next_auto_priority_range. A target with auto_priority
-- set gets a priority range made for it whenever it has no other priority
-- work left: the next most likely prefix, at the length its main sweep is on.
ALTER TABLE targets ADD COLUMN auto_priority INTEGER NOT NULL DEFAULT 0;
-- The length at which every likely prefix has been made into a priority
-- range (or had nothing left to search), so claims stop looking. NULL: not
-- run out at any length yet. Cleared when auto_priority or the alphabet is
-- patched.
ALTER TABLE targets ADD COLUMN auto_priority_exhausted_len INTEGER;
-- Where an automatic priority range's prefix came from: 'listfile' or
-- 'dictionary' (see likely_prefixes.rs). NULL for one an operator made.
ALTER TABLE priority_ranges ADD COLUMN auto_source TEXT;
