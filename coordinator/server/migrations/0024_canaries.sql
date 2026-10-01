-- Canaries: virtual targets with a planted answer, now and then handed to a
-- client instead of real work, to check that it finds what it should - see
-- canary.rs. A virtual target takes part in nothing else (claims, the
-- dashboard's targets, /status, volunteers' totals).
ALTER TABLE targets ADD COLUMN is_virtual INTEGER NOT NULL DEFAULT 0;

CREATE TABLE canaries (
    target_id INTEGER PRIMARY KEY REFERENCES targets(id) ON DELETE CASCADE,
    range_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    -- The filename planted in the range: prefix + candidate + suffix.
    filename TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    -- 'pending' until the client completes the range: 'found' if it reported
    -- exactly `filename`, 'missed' if not.
    result TEXT NOT NULL DEFAULT 'pending',
    resolved_at INTEGER
);
CREATE INDEX canaries_user ON canaries(user_id);
