# namebreak coordinator

[Client for volunteers](https://github.com/sjoblomj/namebreak-bombe/releases) |
[Dashboard for community progress](https://namebreak-coordinator.fly.dev/)

Distributes `namebreak` (see `../client`) across multiple volunteers'
GPUs. A central **server** tracks a set of *targets* (a prefix/suffix + hash
pair to search for), carves each target's candidate space into time-boxed
*ranges*, and hands ranges out to clients over HTTP. The client side lives
directly in `namebreak` itself (`../client`) as its `coordinator`
mode: it registers, claims a range, searches it in-process (no subprocess),
heartbeats progress, and reports back when it's done - see
`../client/src/net/coordinator_runner.h` for that loop and
`../client/src/net/protocol.h` for the wire types, kept in sync by hand with
this directory's `protocol/` crate below (same JSON shapes, same server).

```
coordinator/
  protocol/   shared HTTP API types (used by the server; namebreak's own
              protocol.h mirrors these same shapes for the client side)
  server/     axum + sqlx(SQLite) coordinator - owns all range bookkeeping
```

Visit the server's base URL in a browser (`GET /`) for a live dashboard -
every target with its bound filenames, prefix, suffix, pruning rules and
insertions, its ranges, each range's status and who worked on it (from
`last_assigned_user_id`, which - unlike `assigned_user_id` - is never
cleared on reclaim), and a solved target's find called out with a prominent
banner. If a range was reassigned partway through (see progress
checkpointing below), each finished portion shows up as its own row credited
to whoever actually searched it, rather than the whole thing appearing under
just the most recent claimer. A meet-in-the-middle target shows its meeting
point, how much of the directory its table holds (and what that costs a
client in memory), and how many tails that leaves to rewind, with its ranges
as directories whose table characters are '?'. That's the **Targets** tab.
The **Matches** tab lists every name found, newest first - who found it and
when, its Hash A and Hash B, and the targets it solved (targets sharing a
Hash A/Hash B pair share a row). The **Volunteers** tab ranks every username
that has claimed work or found a name by candidates searched (the sizes of
its completed ranges, across all its hostnames - a little generous, since a
range closed early by a find still counts in full), alongside its completed
ranges, hostnames and names found (targets sharing a Hash A/Hash B pair are
one find), and its canaries found out of those handed to it (see below). The
**Introduction** tab explains what namebreaking is. The open tab is kept in
the URL's `#fragment`, so `/#volunteers` links straight to it. Plain
HTML/CSS/JS (`server/static/dashboard.html`, embedded into the binary at
compile time), which loads `GET /api/v1/dashboard` once each time it's
opened (or reloaded) - no build step, no framework. With desktop
notifications on, it checks the much smaller `GET /api/v1/status` every
minute for finds, and loads the dashboard's data again when there's one.
`GET /api/v1/dashboard` and `GET /` are both public (no auth), matching `GET
/api/v1/status`.

## Running locally

```sh
cd coordinator
ADMIN_TOKEN=devsecret DATABASE_URL=sqlite://namebreak.db cargo run -p namebreak-server
```

It listens on port 8080 (`BIND_ADDR`); see [Configuring and deploying the
server](docs/deploying.md) for the rest of its settings, and [Managing
targets](docs/targets.md) for adding something to search for.

## Running a client

Build `namebreak` as usual first (see [Building namebreak](../client/docs/building.md) - the
default build includes coordinator support; `-DNAMEBREAK_NETWORK=OFF` omits
it). Then, in the directory you want its `matches/` directory in, create a
`config.conf` (see [Configuring namebreak](../client/docs/configuration.md))
with a
`[coordinator]` section:

```sh
cd run   # or any working directory of your choice
cat > config.conf <<EOF
mode = coordinator

[coordinator]
server_url = http://localhost:8080
username = yourname
EOF
../client/build/namebreak
```

`hostname` (optional) defaults to the machine's actual hostname - on a
machine with several GPUs, each searches as a client of its own, GPU 0 as
`hostname` and GPU N as `<hostname>/gpuN`, so that each one's ranges are
sized by its own speed (see the client's [Choosing and tuning the
hardware](../client/docs/hardware.md)), and `poll_interval_secs` (optional) defaults to 30. In the
`[hardware]` section, `meet_in_the_middle_table_memory` (optional, default
none) is how much memory this client will give a meet-in-the-middle
target's table - `1G`, `512M`, or a number of bytes (v2026-10-22 had it in
`[coordinator]`, where it is still read). Without it, the server hands out no
meet-in-the-middle target's work: that table is memory a volunteer has to
agree to spare, and on a GPU it has to fit in the device's own too (see
[Meet-in-the-middle targets](docs/meet-in-the-middle-targets.md)). namebreak re-registers (idempotently)
on every start, claims a range, searches it in-process against exactly that
range, reports the result, and loops. If the search fails to even start (bad
claim data, matches file not writable), it skips reporting completion and
lets the range's lease expire so the server reassigns it - it won't report
success or silently drop bad work.

## Documentation

- [How the coordinator works](docs/how-it-works.md) - auth, protocol
  versions, how ranges are carved, leased, checkpointed and split, priority
  and skip ranges, alphabets, the pruning rules and inserted text, and
  canaries.
- [Managing targets](docs/targets.md) - the admin API: adding, changing,
  pausing and deleting targets, skip and priority ranges, and false matches.
- [Dictionary targets](docs/dictionary-targets.md) - word lists, and targets
  searched by words, with the basenames clients send.
- [Meet-in-the-middle targets](docs/meet-in-the-middle-targets.md) - an
  unknown directory, searched from both ends.
- [Configuring and deploying the server](docs/deploying.md) - environment
  variables, and deploying to fly.io.
