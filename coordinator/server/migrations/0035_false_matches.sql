-- Two 32-bit hashes match by chance about once in 2^64 names, and a long
-- search hashes that many: REZ\SFZSSP050WSRL.WAV matched a file's Hash A and
-- Hash B and was nothing like its name. So a match of both hashes is checked
-- against what else is known about the file - its encryption key (the
-- column 0030 added) and where its entry is in the archive's hash table -
-- and one that fails is a false match, not a find (see false_matches.rs).
--
-- Where the entry is: the hash table's size (a power of two), and the slots
-- a lookup can start at and still reach it, first to last (wrapping past the
-- table's end if first is the greater) - see
-- AdminCreateTargetRequest::hash_table_position. All three NULL: not known.
ALTER TABLE targets ADD COLUMN hash_table_size INTEGER;
ALTER TABLE targets ADD COLUMN hash_table_first_start_slot INTEGER;
ALTER TABLE targets ADD COLUMN hash_table_last_start_slot INTEGER;

-- Every false match reported for a target, once: by a client that checked it
-- and searched on (HeartbeatRequest::false_matches), or found by the server
-- in what a client older than protocol 1.7 took for a find. `reason` says
-- what it failed, as the server worked it out when it was reported.
CREATE TABLE false_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets(id),
    filename TEXT NOT NULL,
    reason TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    range_id INTEGER,
    reported_at INTEGER NOT NULL,
    UNIQUE (target_id, filename)
);

-- 1 for a range a client older than protocol 1.7 stopped in, at a false
-- match it took for the file, before the rest of it was searched: what's
-- left of it is handed out again, but only to clients that know a false
-- match when they see one - an older one would stop at the same name again,
-- over and over. Carried over to whatever the range is split into.
ALTER TABLE ranges ADD COLUMN false_match_inside INTEGER NOT NULL DEFAULT 0;
