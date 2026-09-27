-- Replaces targets.skip_regex (0009_target_skip_regex.sql) with skip ranges,
-- and stores both skip and priority ranges as one row per operator request,
-- however many separate stretches of candidates its pattern covers - see
-- alphabet::pattern_spans. 'skipped' rows the regex already produced stay.
ALTER TABLE targets DROP COLUMN skip_regex;

-- Candidates at one length an operator never wants handed out, matched by
-- the same kind of pattern as a priority range. Written out as 'skipped'
-- ranges rows (see ranges.skip_range_id below) only when carving reaches
-- them - see ranges::claim_range.
CREATE TABLE skip_ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets (id),
    pattern TEXT NOT NULL,
    -- Plain text, shown on the dashboard - why these candidates are skipped.
    reason TEXT NOT NULL,
    candidate_len INTEGER NOT NULL,
    -- The alphabet the segments below are denominated in. An alphabet patch
    -- re-expands the pattern under the new one - see
    -- ranges::migrate_skip_ranges_to_new_alphabet.
    alphabet_name TEXT NOT NULL,
    alphabet TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    -- Set when an operator removed it after it had already skipped
    -- something: its segments are gone, but the row stays so the 'skipped'
    -- rows it produced keep their reason - see ranges::remove_skip_range.
    removed_at INTEGER
);

CREATE INDEX idx_skip_ranges_target ON skip_ranges (target_id, candidate_len, alphabet_name);

CREATE TABLE skip_range_segments (
    skip_range_id INTEGER NOT NULL REFERENCES skip_ranges (id),
    start_block INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    end_block INTEGER NOT NULL,
    end_index INTEGER NOT NULL
);

CREATE INDEX idx_skip_range_segments ON skip_range_segments (skip_range_id);

-- Which skip range (if any) a 'skipped' row came from, for the dashboard.
ALTER TABLE ranges ADD COLUMN skip_range_id INTEGER REFERENCES skip_ranges (id);

-- A priority range owns the parts of its segments inside its own
-- [start, end) - start/end/next keep their old meaning, so shrinking end on
-- removal and clamping next on creation work as before.
CREATE TABLE priority_range_segments (
    priority_range_id INTEGER NOT NULL REFERENCES priority_ranges (id),
    start_block INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    end_block INTEGER NOT NULL,
    end_index INTEGER NOT NULL
);

CREATE INDEX idx_priority_range_segments ON priority_range_segments (priority_range_id);

-- Every existing priority range covers one literal prefix: one segment.
INSERT INTO priority_range_segments (priority_range_id, start_block, start_index, end_block, end_index)
SELECT id, start_block, start_index, end_block, end_index FROM priority_ranges;

-- From here on, prefix is only set on rows from before this migration (one
-- row per concrete prefix of their pattern), and an alphabet patch
-- translates those by their prefix. New rows leave it NULL and are
-- translated by their pattern. Rows from before 0012_priority_range_prefix.sql
-- never had a prefix; '' keeps them from being mistaken for new rows, and
-- they stay untranslatable, as before.
UPDATE priority_ranges SET prefix = '' WHERE prefix IS NULL;
