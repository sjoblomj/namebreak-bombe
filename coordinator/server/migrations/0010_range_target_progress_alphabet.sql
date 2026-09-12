-- Lets an admin PATCH a target's alphabet (see handlers::admin_patch_target)
-- without corrupting already-carved candidate<->index mappings: every range,
-- and the carving cursor itself (target_progress), now remembers which
-- alphabet it was carved/computed under, the same way targets already did
-- (see 0003_target_alphabet.sql). A NULL here (only ever true for rows that
-- existed before this migration) is backfilled below to the owning target's
-- alphabet at the time - the only alphabet that could possibly apply, since
-- alphabet patching didn't exist yet. From this point on the application
-- always writes a non-NULL value on every insert/update of these columns
-- (see ranges::insert_skip_range, ranges::persist_progress, and the range
-- INSERT in ranges::claim_range/reclaim_expired); the columns stay nullable
-- at the schema level only because SQLite can't add a NOT NULL column
-- without a fixed literal default, not because NULL is ever written again.
ALTER TABLE ranges ADD COLUMN alphabet_name TEXT;
ALTER TABLE ranges ADD COLUMN alphabet TEXT;
ALTER TABLE target_progress ADD COLUMN alphabet_name TEXT;
ALTER TABLE target_progress ADD COLUMN alphabet TEXT;

UPDATE ranges
SET alphabet_name = (SELECT alphabet_name FROM targets WHERE targets.id = ranges.target_id),
    alphabet = (SELECT alphabet FROM targets WHERE targets.id = ranges.target_id)
WHERE alphabet_name IS NULL;

UPDATE target_progress
SET alphabet_name = (SELECT alphabet_name FROM targets WHERE targets.id = target_progress.target_id),
    alphabet = (SELECT alphabet FROM targets WHERE targets.id = target_progress.target_id)
WHERE alphabet_name IS NULL;
