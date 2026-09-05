-- Highest candidate index within this range that's confirmed searched, from the
-- client's periodic heartbeats. NULL means no progress has been checkpointed yet.
ALTER TABLE ranges ADD COLUMN progress_index INTEGER;
