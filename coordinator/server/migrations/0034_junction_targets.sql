-- Junction targets: a file whose name is a known prefix, an unknown
-- directory and a known suffix, where the hash state right after the
-- directory - the junction - is known for hashA and hashB. The client
-- searches the directory from both ends at once, meeting in the middle (see
-- junction.rs and the client README's "Junction mode"), so what a range
-- hands out is a stretch of one length's *tails*: candidate_len is the
-- directory's length, start/end/progress_index are tail indices (always
-- block 0, as a dictionary target's candidate numbers are), and
-- alphabet_name/alphabet are 'junction'/''. The target's alphabet, prefix,
-- suffix, hashes and pruning rules are the columns every target has;
-- start_len is the shortest directory it searches.
--
-- The junction itself: hashA's two seeds after the directory, then hashB's,
-- stored like hash_a (a u32 in an i64). NULL for a target that isn't one.
ALTER TABLE targets ADD COLUMN junction_a_seed1 INTEGER;
ALTER TABLE targets ADD COLUMN junction_a_seed2 INTEGER;
ALTER TABLE targets ADD COLUMN junction_b_seed1 INTEGER;
ALTER TABLE targets ADD COLUMN junction_b_seed2 INTEGER;
-- The longest directory it searches (the shortest is start_len), and how
-- many of a directory's first characters the client's table holds - which
-- decides both what a tail is and what the search costs in memory, so it's
-- the same for every client of this target.
ALTER TABLE targets ADD COLUMN junction_max_length INTEGER;
ALTER TABLE targets ADD COLUMN junction_table_length INTEGER;

-- A junction search's rate is in tails a second, a thousand times lower than
-- an alphabet search's candidates on the same GPU, so it's kept apart - as a
-- dictionary search's is (see ranges::claim_range).
ALTER TABLE users ADD COLUMN ema_junction_rate_per_sec REAL;
-- How much memory this client said it will give a junction search's table,
-- in bytes, at /register - see junction.rs. NULL for a client that doesn't
-- say (one older than protocol 1.6), which gets no junction work.
ALTER TABLE users ADD COLUMN junction_table_bytes INTEGER;
