-- Basenames are kept compressed (see basenames.rs): a report's new ones as
-- one gzip member (basename_reports.names), with an 8-byte hash of each
-- (basename_hashes) to keep each once per target - 33 MB for a million
-- three-word basenames, where a row each (0031's basenames table) took 100.
-- No client released before this sends basenames, so 0031's table has none
-- to keep.
DROP INDEX idx_basenames_target;
DROP TABLE basenames;
UPDATE targets SET basename_count = 0;

-- What one heartbeat, quit or completion brought that the target hadn't
-- had: who sent it, when, with which range, and the basenames themselves,
-- in the order sent, one per line, gzipped.
CREATE TABLE basename_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets (id),
    user_id INTEGER NOT NULL REFERENCES users (id),
    range_id INTEGER NOT NULL,
    reported_at INTEGER NOT NULL,
    count INTEGER NOT NULL,
    names BLOB NOT NULL
);
-- A target's reports in the order they came (by rowid).
CREATE INDEX idx_basename_reports_target ON basename_reports (target_id);

-- Every basename a target has, as basenames::hash of it.
CREATE TABLE basename_hashes (
    target_id INTEGER NOT NULL,
    hash INTEGER NOT NULL,
    PRIMARY KEY (target_id, hash)
) WITHOUT ROWID;
