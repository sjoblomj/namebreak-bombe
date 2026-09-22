-- Whether claims for this target tell clients to skip candidates that close
-- a bracket never opened - see AdminCreateTargetRequest::prune_unopened_brackets.
ALTER TABLE targets ADD COLUMN prune_unopened_brackets INTEGER NOT NULL DEFAULT 0;
