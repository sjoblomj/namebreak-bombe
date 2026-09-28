-- The client release each user last registered with ("vYYYY-MM-DD", "dev",
-- ...) - NULL for a client from before protocol 1.1.0, which doesn't say.
-- See client_release.rs.
ALTER TABLE users ADD COLUMN client_release TEXT;

-- Small operator settings, changed at runtime through the admin API rather
-- than environment variables, so changing one needs no restart. Currently
-- only the minimum client release (see client_release::load_client_releases).
CREATE TABLE server_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
