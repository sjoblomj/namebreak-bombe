-- The shortest candidate length the main sweep carves for this target - see
-- AdminCreateTargetRequest::start_len.
ALTER TABLE targets ADD COLUMN start_len INTEGER NOT NULL DEFAULT 1;
