-- Optional operator-supplied note shown on the target's dashboard card.
-- Rendered as raw HTML there (not escaped) so an operator can use tags like
-- <b> for emphasis - this is only ever set via the ADMIN_TOKEN-protected
-- admin API, never from untrusted input.
ALTER TABLE targets ADD COLUMN description TEXT;
