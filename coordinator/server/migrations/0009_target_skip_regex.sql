-- Lets an operator exclude parts of a target's search space from ever being
-- handed out: a regex tested against a candidate's *leading character only*
-- (see alphabet::skip_char_mask/find_skip_run) - e.g. "[M-Q]" skips every
-- candidate starting with M through Q, at every candidate length. Checked
-- only while carving fresh ranges off a target's cursor (ranges::claim_range),
-- never retroactively against ranges already carved. NULL/empty means no
-- skipping, same as every existing target.
ALTER TABLE targets ADD COLUMN skip_regex TEXT;
