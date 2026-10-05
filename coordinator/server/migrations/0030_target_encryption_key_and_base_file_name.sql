-- What's known about an encrypted target file beyond its Hash A and Hash B -
-- see AdminCreateTargetRequest::encryption_key_hex and base_file_name. Only
-- stored and shown on the dashboard; the search doesn't use them. NULL: not
-- known (or not given).
ALTER TABLE targets ADD COLUMN encryption_key INTEGER;
ALTER TABLE targets ADD COLUMN base_file_name TEXT;
