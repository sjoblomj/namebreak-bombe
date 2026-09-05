-- alphabet_name is kept for display/audit; alphabet stores the resolved
-- characters, frozen at target-creation time, so a target's already-carved
-- candidate<->index mapping can never shift if a predefined profile's
-- definition is ever edited in a later code change.
ALTER TABLE targets ADD COLUMN alphabet_name TEXT NOT NULL DEFAULT 'size49';
ALTER TABLE targets ADD COLUMN alphabet TEXT NOT NULL DEFAULT ' !&''()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_';
