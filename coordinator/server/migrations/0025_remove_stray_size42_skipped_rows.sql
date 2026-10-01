-- One-off cleanup. Switching target 4 (rez-wav1) from size49 to size42 and
-- back made ranges::migrate_skip_ranges_to_new_alphabet write a size42 copy
-- of every one of its length-11 skip ranges' `skipped` rows, although the
-- sweep had long since passed length 11: the size49 rows were already
-- there, and size49's completed ranges and skipped rows cover every
-- candidate of length 11. The copies say nothing the size49 rows don't, and
-- showed on the dashboard with gaps between them. It no longer writes rows
-- where the sweep has been.
--
-- Matched by id as well as by everything that made them strays, so this
-- does nothing on any other database.
DELETE FROM ranges
WHERE id BETWEEN 836 AND 869
  AND target_id = 4
  AND status = 'skipped'
  AND alphabet_name = 'size42'
  AND candidate_len = 11
  AND skip_range_id IN (SELECT id FROM skip_ranges WHERE target_id = 4 AND alphabet_name = 'size49')
  AND EXISTS (SELECT 1 FROM targets WHERE id = 4 AND name = 'rez-wav1' AND alphabet_name = 'size49');
