-- Remembers which backend each client searches with (e.g. "cuda", "cpu"),
-- as declared at registration - see handlers::register. Informational only.
-- Empty for users that registered before clients sent it (or never re-register).
ALTER TABLE users ADD COLUMN backend TEXT NOT NULL DEFAULT '';
