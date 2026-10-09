-- A dictionary target's tails: what follows each candidate's last word - see
-- dictionary::expand_tails. A JSON array of the tail elements, as given;
-- NULL (as for every target made before) is none.
ALTER TABLE targets ADD COLUMN tails TEXT;
