CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    hostname TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    ema_rate_per_sec REAL,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    UNIQUE (username, hostname)
);

CREATE TABLE targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    suffix TEXT NOT NULL,
    hash_a INTEGER NOT NULL,
    hash_b INTEGER NOT NULL,
    min_len INTEGER NOT NULL,
    max_len INTEGER NOT NULL,
    prune_symbol_runs INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    found_filename TEXT,
    found_by_user_id INTEGER REFERENCES users (id),
    created_at INTEGER NOT NULL
);

-- One row per target: tracks how far its search space has been carved into
-- ranges so far. candidate_len only ever advances (never resets down) once
-- next_index reaches that length's full space.
CREATE TABLE target_progress (
    target_id INTEGER PRIMARY KEY REFERENCES targets (id),
    candidate_len INTEGER NOT NULL,
    next_index INTEGER NOT NULL
);

CREATE TABLE ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets (id),
    candidate_len INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    end_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    assigned_user_id INTEGER REFERENCES users (id),
    assigned_at INTEGER,
    lease_seconds INTEGER,
    lease_expires_at INTEGER,
    completed_at INTEGER,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_ranges_target_status ON ranges (target_id, status);
CREATE INDEX idx_ranges_reclaim ON ranges (status, lease_expires_at);
