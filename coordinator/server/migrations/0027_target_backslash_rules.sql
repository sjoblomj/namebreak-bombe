-- Whether claims for this target tell clients to skip candidates with fewer
-- backslashes than this, or with two next to each other - see
-- AdminCreateTargetRequest::min_backslash_count and
-- AdminCreateTargetRequest::prune_adjacent_backslashes.
ALTER TABLE targets ADD COLUMN min_backslash_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE targets ADD COLUMN prune_adjacent_backslashes INTEGER NOT NULL DEFAULT 0;
