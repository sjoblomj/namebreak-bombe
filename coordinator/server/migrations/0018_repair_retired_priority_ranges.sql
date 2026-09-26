-- Deleting a priority range that had already handed out work used to
-- "retire" it: next_index was forced to end_index while its full span was
-- kept. The main sweep jumps over a priority range's whole span, so the part
-- that hadn't been handed out yet was never searched by anyone.
--
-- This repairs those rows the way ranges::remove_priority_range now removes
-- one: next_index goes back to how far the priority range really got (the
-- furthest end of the ranges carved from it), and the span is cut back to
-- just what the main sweep has already gone past - see
-- ranges::main_sweep_passed_to. So the unsearched remainder goes back to
-- the main sweep where it still can, and is handed out as priority work
-- again where the main sweep is already past it.
--
-- Rows whose carved ranges are in another alphabet (an alphabet patch
-- translated the priority range afterwards) can't be compared and are left
-- alone. A priority range that ran out naturally has carved all the way to
-- end_index, so it doesn't match.
WITH retired AS (
    SELECT pr.id,
           (SELECT MAX(r.end_index) FROM ranges r WHERE r.priority_range_id = pr.id) AS carved_to
    FROM priority_ranges pr
    WHERE pr.next_index = pr.end_index
      AND NOT EXISTS (SELECT 1 FROM ranges r WHERE r.priority_range_id = pr.id AND r.alphabet_name != pr.alphabet_name)
)
UPDATE priority_ranges
SET next_index = retired.carved_to,
    end_index = CASE
        WHEN tp.alphabet_name != priority_ranges.alphabet_name OR tp.candidate_len > priority_ranges.candidate_len THEN priority_ranges.end_index
        WHEN tp.candidate_len < priority_ranges.candidate_len THEN retired.carved_to
        ELSE MIN(MAX(tp.next_index, retired.carved_to), priority_ranges.end_index)
    END
FROM retired, target_progress AS tp
WHERE retired.id = priority_ranges.id
  AND tp.target_id = priority_ranges.target_id
  AND retired.carved_to IS NOT NULL
  AND retired.carved_to < priority_ranges.end_index;
