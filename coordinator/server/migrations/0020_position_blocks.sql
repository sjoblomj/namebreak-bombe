-- Candidates longer than alphabet::index_width (11 for every predefined
-- alphabet) have more positions than an INTEGER can hold, so every stored
-- position becomes a (block, index) pair: the candidate's leading prefix
-- (numbered) and its index among that prefix's candidates - see
-- alphabet::split_pos. Block is always 0 up to that length, which is all
-- that could exist before, so existing rows read back unchanged.
ALTER TABLE ranges ADD COLUMN start_block INTEGER NOT NULL DEFAULT 0;
ALTER TABLE ranges ADD COLUMN end_block INTEGER NOT NULL DEFAULT 0;
ALTER TABLE ranges ADD COLUMN progress_block INTEGER NOT NULL DEFAULT 0;
ALTER TABLE target_progress ADD COLUMN next_block INTEGER NOT NULL DEFAULT 0;
ALTER TABLE priority_ranges ADD COLUMN start_block INTEGER NOT NULL DEFAULT 0;
ALTER TABLE priority_ranges ADD COLUMN end_block INTEGER NOT NULL DEFAULT 0;
ALTER TABLE priority_ranges ADD COLUMN next_block INTEGER NOT NULL DEFAULT 0;
