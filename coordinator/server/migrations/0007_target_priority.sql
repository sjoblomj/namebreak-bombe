-- Lets an operator make one target's work get claimed ahead of others: a
-- worker's /claim always drains the highest-priority active target's
-- claimable work first (see ranges::claim_range), only falling through to
-- lower-priority targets once nothing claimable remains above them. Default
-- 0 keeps every existing target's behavior unchanged - plain creation-order
-- FIFO among targets that share a priority.
ALTER TABLE targets ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;
