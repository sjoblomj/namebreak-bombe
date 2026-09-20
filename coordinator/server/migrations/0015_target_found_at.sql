-- Unix time the target was solved - set alongside found_filename/
-- found_by_user_id when a range reports the match. NULL for targets that
-- aren't solved, and for any solved before this column existed (the moment
-- wasn't recorded then, and nothing else pins it down exactly).
ALTER TABLE targets ADD COLUMN found_at INTEGER;
