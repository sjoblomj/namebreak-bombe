-- One-off: target 1151 (rez-wav-2) was solved by REZ\SFZSSP050WSRL.WAV, a
-- false match (see 0035) - a client older than protocol 1.7 took it for the
-- file. This makes it unsolved again, keeps the name as a false match, and
-- hands back what was left of the five ranges in progress when it was found.
--
-- The file's encryption key is 0x4A8FA566 (worked out from
-- bw-ptbr-install.exe, see client/tests/match_check_test.cpp), and
-- REZ\SFZSSP050WSRL.WAV's basename's is 0x38D3F662. Its entry is slot 858
-- of the 1024-slot hash table of StarCraft's INSTALL.EXE, where a lookup can
-- start at slots 848 to 858 - and the name's starts at 762. With those
-- known, the server and 1.7 clients both tell the name apart from the file
-- from now on.
--
-- Every statement checks that the target is still solved with that name, so
-- nothing happens on any other database, or if it was changed by hand first.
-- The target itself is changed last, as the others go by it.

-- The name, as a false match reported by its finder in range 5120, with the
-- reason false_matches::false_match_reason gives.
INSERT OR IGNORE INTO false_matches (target_id, filename, reason, user_id, range_id, reported_at)
SELECT id, found_filename,
       'its basename''s key is 0x38D3F662, not the file''s 0x4A8FA566, and its lookup starts at hash table slot 762, not at one from 848 to 858 of 1024',
       found_by_user_id, 5120, found_at
FROM targets
WHERE id = 1151
  AND name = 'rez-wav-2'
  AND status = 'solved'
  AND found_filename = 'REZ\SFZSSP050WSRL.WAV';

-- The five ranges closed as completed when it was found: 5120, whose client
-- found it, and 5116, 5117, 5119 and 5122, closed at their next heartbeat.
-- Each goes back to pending as if released (see ranges::release_range),
-- keeping its last heartbeat's checkpoint and who searched up to it - so the
-- next claim of it splits off the searched part as completed, credited to
-- them, and hands out only the rest (ranges::split_off_searched_portion).
UPDATE ranges
SET status = 'pending',
    assigned_user_id = NULL,
    lease_expires_at = NULL,
    last_progress_at = NULL,
    completed_at = NULL
WHERE id IN (5116, 5117, 5119, 5120, 5122)
  AND target_id = 1151
  AND status = 'completed'
  AND progress_index IS NOT NULL
  AND last_assigned_user_id IS NOT NULL
  AND completed_at >= (
      SELECT found_at FROM targets
      WHERE id = 1151
        AND name = 'rez-wav-2'
        AND status = 'solved'
        AND found_filename = 'REZ\SFZSSP050WSRL.WAV'
  );

-- 5120 was searched up to and including the false match itself, further
-- than its last heartbeat (FZS0--U FIJ7): a client reports a find only once
-- the batch it's in is done, as complete_range's hand_back_after_false_match
-- takes it. FZSSP050WSRL in size40 at length 12 is block 18 ('F'), index
-- 406793106237170824 (ZSSP050WSRL) - inside the range's own block-18 span,
-- and past its heartbeat's 406605176116386490.
UPDATE ranges
SET progress_block = 18,
    progress_index = 406793106237170824
WHERE id = 5120
  AND target_id = 1151
  AND status = 'pending'
  AND candidate_len = 12
  AND alphabet_name = 'size40'
  AND start_block = 18
  AND end_block = 18
  AND start_index <= 406793106237170824
  AND end_index > 406793106237170824
  AND progress_block = 18
  AND progress_index < 406793106237170824;

UPDATE targets
SET status = 'active',
    found_filename = NULL,
    found_by_user_id = NULL,
    found_at = NULL,
    encryption_key = 1250927974, -- 0x4A8FA566
    hash_table_size = 1024,
    hash_table_first_start_slot = 848,
    hash_table_last_start_slot = 858
WHERE id = 1151
  AND name = 'rez-wav-2'
  AND status = 'solved'
  AND found_filename = 'REZ\SFZSSP050WSRL.WAV';
