-- Whether claims for this target tell clients to apply the pruning rules to
-- every character of a candidate but the last, rather than only the leading
-- ones - see AdminCreateTargetRequest::prune_whole_candidate.
ALTER TABLE targets ADD COLUMN prune_whole_candidate INTEGER NOT NULL DEFAULT 0;
