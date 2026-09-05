-- Preserves who most recently worked a range even after it's reclaimed:
-- assigned_user_id gets nulled out on reclaim (needed for ownership checks to
-- correctly reject a stale client's heartbeat/complete calls), but this column
-- never does, so the dashboard can show "who is/was working on it" regardless
-- of the range's current status.
ALTER TABLE ranges ADD COLUMN last_assigned_user_id INTEGER REFERENCES users(id);
