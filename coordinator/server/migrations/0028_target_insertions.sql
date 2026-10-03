-- Text inserted into every candidate of the target long enough for it, and
-- where - see AdminCreateTargetRequest::insert_from_start and
-- insert_from_end. NULL text: nothing inserted.
ALTER TABLE targets ADD COLUMN insert_from_start_text TEXT;
ALTER TABLE targets ADD COLUMN insert_from_start_position INTEGER NOT NULL DEFAULT 0;
ALTER TABLE targets ADD COLUMN insert_from_end_text TEXT;
ALTER TABLE targets ADD COLUMN insert_from_end_position INTEGER NOT NULL DEFAULT 0;
