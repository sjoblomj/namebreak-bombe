-- Lets an admin fast-track a specific, bounded slice of a target's search
-- space ahead of its normal sequential sweep (see handlers::admin_create_priority_range,
-- ranges::claim_range) - e.g. "try everything starting with [ _-]S, at exactly
-- 9 characters, before anything else". Unlike skip_regex, this never removes
-- anything from the search; it only reorders when it happens. A priority
-- range is scoped to exactly one candidate_len (see alphabet::expand_priority_pattern's
-- doc comment on why) and, once created, permanently excludes its own
-- [start_index, end_index) span from the target's own main-sweep cursor at
-- that length - not persisted as its own 'skipped' row, since it isn't
-- uninteresting, just claimed by this mechanism instead (the dashboard shows
-- the gap purely by comparing this table against target_progress, with
-- nothing stored for it).
CREATE TABLE priority_ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets (id),
    -- Higher claims first, same convention as targets.priority; ties broken
    -- by created_at. Multiple priority ranges may share a priority value.
    priority INTEGER NOT NULL DEFAULT 0,
    -- The operator-supplied pattern this row's prefix was expanded from (see
    -- alphabet::expand_priority_pattern) - kept for display only; the actual
    -- carving math only ever uses start_index/end_index below.
    pattern TEXT NOT NULL,
    candidate_len INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    end_index INTEGER NOT NULL,
    -- This priority range's own cursor within [start_index, end_index) - once
    -- it reaches end_index, this row is permanently exhausted (there's no
    -- next length to bump to, unlike target_progress).
    next_index INTEGER NOT NULL,
    -- Frozen at creation, same reasoning as targets.alphabet - see
    -- 0003_target_alphabet.sql. Independent of the target's own alphabet
    -- transition handling (alphabet::transition_alphabet_cursor): a priority
    -- range simply keeps running under whatever alphabet it was created
    -- with, unaffected by a later admin_patch_target alphabet change.
    alphabet_name TEXT NOT NULL,
    alphabet TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_priority_ranges_target ON priority_ranges (target_id, priority DESC, created_at ASC);

-- Which priority range (if any) produced a given range row, purely for
-- dashboard labeling - claim/heartbeat/complete/reclaim logic never needs to
-- know a range's origin, they all operate on ranges generically.
ALTER TABLE ranges ADD COLUMN priority_range_id INTEGER REFERENCES priority_ranges (id);
