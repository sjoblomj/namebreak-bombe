-- What ran each range. users holds only what a client last registered with, so
-- once it upgrades, that says nothing about the ranges it searched before. A
-- range keeps its own copy, made when it's claimed, of what's known about the
-- client claiming it - who that is being last_assigned_user_id, whose username
-- and hostname never change. So all the ranges a bad release searched on one
-- kind of machine can be found, and handed out again. NULL for a range claimed
-- before this, or for what a client doesn't say.
ALTER TABLE ranges ADD COLUMN client_backend TEXT;
ALTER TABLE ranges ADD COLUMN client_release TEXT;
ALTER TABLE ranges ADD COLUMN client_protocol_version TEXT;
ALTER TABLE ranges ADD COLUMN client_platform TEXT;
ALTER TABLE ranges ADD COLUMN client_device TEXT;

-- The operating system and CPU architecture a client was built for, and the
-- device it searches on - see RegisterRequest::platform and device.
ALTER TABLE users ADD COLUMN platform TEXT;
ALTER TABLE users ADD COLUMN device TEXT;
