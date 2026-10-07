-- Dictionary targets: candidates made of words from word lists the server
-- stores, with separators between them, numbered as the client numbers them
-- (see dictionary.rs) - rather than every string of an alphabet. A
-- dictionary target's ranges are ranges of those numbers: candidate_len is
-- how many words its candidates have, start/end/progress_index are candidate
-- numbers (always block 0), and alphabet_name/alphabet are 'dictionary'/''.
ALTER TABLE targets ADD COLUMN kind TEXT NOT NULL DEFAULT 'alphabet';
-- JSON arrays of strings: the word lists' names, in the order given, and the
-- separators. NULL for an alphabet target.
ALTER TABLE targets ADD COLUMN word_lists TEXT;
ALTER TABLE targets ADD COLUMN separators TEXT;
ALTER TABLE targets ADD COLUMN min_words INTEGER;
ALTER TABLE targets ADD COLUMN max_words INTEGER;
-- 1 to have clients send every basename matching the encryption key - see
-- AdminCreateTargetRequest::send_basenames.
ALTER TABLE targets ADD COLUMN send_basenames INTEGER NOT NULL DEFAULT 0;
-- How many rows of basenames below are the target's, kept as they're added:
-- three words of english-1 make about a million, which the dashboard
-- shouldn't count every time it's shown.
ALTER TABLE targets ADD COLUMN basename_count INTEGER NOT NULL DEFAULT 0;

-- A dictionary search's rate is a few hundred times lower than an alphabet
-- search's on the same GPU, so it's kept apart - see ranges::claim_range.
ALTER TABLE users ADD COLUMN ema_dictionary_rate_per_sec REAL;

-- Word lists, as uploaded. Never changed once stored: a different list gets
-- a different name, so a client's cached copy of a name stays good.
CREATE TABLE word_lists (
    name TEXT PRIMARY KEY,
    -- The file exactly as uploaded - what clients download.
    content BLOB NOT NULL,
    -- How many different words it has, read as a client reads it.
    word_count INTEGER NOT NULL,
    -- dictionary::checksum of those words, sorted: 16 lowercase hex digits.
    checksum TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

-- Basenames clients found matching a target's encryption key - see
-- AdminCreateTargetRequest::send_basenames. Each once per target: the first
-- report of it, by whom and in which range.
CREATE TABLE basenames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets (id),
    basename TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users (id),
    range_id INTEGER NOT NULL,
    reported_at INTEGER NOT NULL,
    UNIQUE (target_id, basename)
);
-- A target's basenames in the order they came (by rowid), for the latest
-- of them and for listing them a page at a time.
CREATE INDEX idx_basenames_target ON basenames (target_id);
