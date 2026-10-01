-- One-off: removes range 897, the main sweep's latest range on target 5
-- (rez-wav-gamemenu-1), " _         " to "((E6MFP4)6S" at length 11 in size42,
-- and moves the sweep back to its start, so its candidates are carved again
-- as if it had never been handed out.
--
-- Only while the sweep is still exactly at the range's end - nothing carved
-- after it - and the range isn't completed. Otherwise nothing happens:
-- removing it would then leave candidates no range covers that the sweep
-- never comes back to, or throw away searched work. Length 11 is within
-- size42's index_width, so its positions are all in block 0 and the start
-- reads back as the same cursor (see alphabet::split_end).
UPDATE target_progress
SET next_block = r.start_block,
    next_index = r.start_index
FROM ranges AS r
WHERE r.id = 897
  AND r.target_id = 5
  AND r.candidate_len = 11
  AND r.alphabet_name = 'size42'
  AND r.status != 'completed'
  AND r.priority_range_id IS NULL
  AND r.skip_range_id IS NULL
  AND r.start_block = 0
  AND EXISTS (SELECT 1 FROM targets WHERE id = 5 AND name = 'rez-wav-gamemenu-1')
  AND target_progress.target_id = r.target_id
  AND target_progress.candidate_len = r.candidate_len
  AND target_progress.alphabet_name = r.alphabet_name
  AND target_progress.next_block = r.end_block
  AND target_progress.next_index = r.end_index;

-- Only if the sweep was moved back above.
DELETE FROM ranges
WHERE id = 897
  AND target_id = 5
  AND EXISTS (
      SELECT 1 FROM target_progress AS tp
      WHERE tp.target_id = ranges.target_id
        AND tp.candidate_len = ranges.candidate_len
        AND tp.alphabet_name = ranges.alphabet_name
        AND tp.next_block = ranges.start_block
        AND tp.next_index = ranges.start_index
  );
