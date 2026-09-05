-- Max '\' occurrences namebreak will allow in a candidate before discarding it
-- unhashed. 0 means unlimited (matches prior behavior, and is the natural
-- default: an operator who wants zero backslashes ever should pick an alphabet
-- that doesn't contain '\' instead).
ALTER TABLE targets ADD COLUMN max_backslash_count INTEGER NOT NULL DEFAULT 0;
