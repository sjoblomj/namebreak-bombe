# namebreak coordinator

Distributes `namebreak` (see `../namebreaker-cuda`) across multiple volunteers'
GPUs. A central **server** tracks a set of *targets* (a prefix/suffix + hash
pair to search for), carves each target's candidate space into time-boxed
*ranges*, and hands ranges out to **clients** over HTTP. Clients run the
existing `namebreak` CUDA binary as a subprocess against exactly the range
they were given and report back when it's done.

```
coordinator/
  protocol/   shared HTTP API types (used by both server and client)
  server/     axum + sqlx(SQLite) coordinator - owns all range bookkeeping
  client/     wraps a local `namebreak` binary: claims ranges, runs them, reports back
```

Visit the server's base URL in a browser (`GET /`) for a live dashboard - every
target with its bound filenames, its ranges, each range's status and who worked on it (from
`last_assigned_user_id`, which - unlike `assigned_user_id` - is never cleared on
reclaim), and a solved target's find called out with a prominent banner. If a
range was reassigned partway through (see progress checkpointing below), each
finished portion shows up as its own row credited to whoever actually searched
it, rather than the whole thing appearing under just the most recent claimer.
Plain HTML/CSS/JS (`server/static/dashboard.html`, embedded into the binary at
compile time), polling `GET /api/v1/dashboard` every 8s - no build step, no
framework. `GET /api/v1/dashboard` and `GET /` are both public (no auth),
matching `GET /api/v1/status`.

See the top-level plan/design notes for the full rationale; the short version:

- **Auth**: `/register {username, hostname}` (no password) hands back an opaque
  bearer token. Every other endpoint requires it - this is the only thing
  standing between a real client and a generic bot scraping the API, so it's
  intentionally simple rather than absent.
- **Ranges**: a target's candidate space is carved into contiguous chunks sized
  from each user's observed candidates/sec, so a chunk takes roughly
  `TARGET_CHUNK_SECONDS` regardless of GPU speed. A range that isn't completed
  or heartbeated before its lease expires is automatically reassigned to
  someone else.
- **Alphabets**: each target picks one of a small set of predefined alphabets
  (`server/src/alphabet.rs`'s `PREDEFINED_ALPHABETS`, also listable via
  `GET /api/v1/alphabets`) - variations on the default 49-character set, with or
  without brackets/backslash and with a reduced punctuation set, currently
  `size50`, `size49`, `size48`, `size47`, `size43` and `size42`. The set of
  distinct *sizes* (42/43/47/48/49/50) is compiled into `namebreak.cu` as
  separate template instantiations (the same zero-cost trick already used for
  `--prune-symbol-runs`), so picking a different alphabet costs no performance -
  but it does mean a genuinely new *size* (not just a new named profile at an
  existing size) requires editing `namebreak.cu`'s dispatch and
  recompiling/redistributing the binary to volunteers.
- **Backslash limiting**: each target also has a `max_backslash_count` (default
  `0` = unlimited). `namebreak` discards any candidate with more `\` occurrences
  than this before spending a hash chain on it - the same style of cheap
  pre-filter as `--prune-symbol-runs`'s "no 3 consecutive symbols" rule, just
  targeting one specific character instead. To forbid `\` entirely, use an
  alphabet that doesn't contain it rather than `max_backslash_count: 0` - `0` is
  the "no limit" sentinel, not "zero allowed".
- **Progress checkpointing**: every 60s the client heartbeats the most recent
  partial (Hash A only) match `namebreak` has printed for its current range, if
  any. `namebreak` only logs a match after the CUDA batch containing it has
  finished, so everything up to that candidate is known to be searched - the
  server records it as the range's `progress_index`. If the range is later
  reassigned (lease expired, client disconnected) with real progress recorded,
  the original row is shrunk down to exactly the searched portion and marked
  `completed` (still credited to whoever searched it), and a *new* row is
  carved for the remainder and handed to the next claimer - so the range is
  literally split in two, each part correctly attributed, rather than the
  whole thing ending up credited to whichever client finishes it. If progress
  had already reached the end, the range is simply marked complete instead of
  split/reassigned. Note this only helps when a Hash A collision happens to
  occur (roughly one in 2^32 candidates), so for smaller ranges a disconnect is
  often not checkpointed at all and the range gets fully redone (as a single
  reassigned row, same as before) - not a correctness problem, just a missed
  optimization in that case.
- **Stopping on a find**: the same heartbeat also carries a `target_solved`
  flag, true once *any* range of that target has been completed with a match.
  A client still searching a different range of an already-solved target sees
  this on its next heartbeat (so within `HEARTBEAT_INTERVAL`, 60s) and kills
  its running `namebreak` - the whole process *group*, not just the direct
  child, so this also works if `--namebreak-bin` ever points at a wrapper
  script rather than the real binary directly. That range is closed out
  server-side at the same moment (no `/complete` round-trip - there's nothing
  meaningful to report), and the client moves straight on to its next `/claim`.
- **Storage**: SQLite on a single Fly Volume. One server instance only - range
  assignment has to be centrally coordinated anyway, so this isn't a real
  limitation.

## Running locally

```sh
cd coordinator
ADMIN_TOKEN=devsecret DATABASE_URL=sqlite://namebreak.db cargo run -p namebreak-server
```

Add a target (the operator-only side, protected by `ADMIN_TOKEN`):

```sh
curl -X POST localhost:8080/api/v1/admin/targets \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{
    "name": "rez-finz09bx",
    "prefix": "REZ\\", "suffix": ".TXT",
    "hash_a_hex": "0xF60F5D90", "hash_b_hex": "0xCE0A9BDB",
    "lower_bound": "FINZ09BX",
    "upper_bound": "GAMEMENU",
    "prune_symbol_runs": true,
    "alphabet_name": "size49",
    "max_backslash_count": 0
  }'
```

`lower_bound`/`upper_bound` tighten the search to exactly that alphabetical
range at *every* candidate length that gets searched - not just their own
literal length. They don't need to be the same length as each other (e.g.
`"BLACKSMITH"` to `"CATAPULT"`, 10 and 8 characters, is valid: at length 10,
candidates are bounded between `BLACKSMITH` and `CATAPULT` + padding). They
also don't need to relate to this target's own `prefix`/`suffix` at all, or
even be candidates a real match of this target could ever equal - a bound is
often a *different*, already-known filename (a neighboring entry from a
listfile, or from an adjacent hash-table slot) used purely for its
alphabetical position, e.g. a target with `suffix: ".WAV"` can legitimately
have `lower_bound: "GLUE\\PALCS\\DLG.GRP"` and
`upper_bound: "MUSIC\\MENGSKVICTORY.WAV"` even though neither one ends in
`.WAV` or has anything to do with this target's actual prefix. A bound longer
than this server's supported maximum length (below) is fine too - only its
first that-many characters are ever consulted, the rest is simply never
truncated into relevance.

There's no `min_len`: the server always starts at length 1 and no `max_len`:
it always searches up to as long as the chosen alphabet supports (capped at
whatever length still fits a flat 64-bit range index, `alphabet_size^len <=
i64::MAX` - 11 for every alphabet currently in `PREDEFINED_ALPHABETS`, since
they're all close enough in size to land on the same cap; a genuinely smaller
alphabet, e.g. a hex-only one, would push it noticeably higher) - tightened at
every length in between by `lower_bound`/`upper_bound`. `alphabet_name`
defaults to `"size49"` if omitted; see `GET /api/v1/alphabets` for the full
list.

Check progress:

```sh
curl localhost:8080/api/v1/status
```

Pause/resume a target:

```sh
curl -X PATCH localhost:8080/api/v1/admin/targets/1 \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"status": "paused"}'
```

Delete a target permanently (also removes its ranges and carving cursor - not
reversible):

```sh
curl -X DELETE localhost:8080/api/v1/admin/targets/1 -H 'X-Admin-Token: devsecret'
```

## Running a client

Build `namebreak` as usual first (see `../namebreaker-cuda/Makefile`), then:

```sh
cargo run -p namebreak-client -- \
  --server-url http://localhost:8080 \
  --username yourname \
  --namebreak-bin ../namebreaker-cuda/namebreak \
  --workdir ./run
```

`--hostname` defaults to the machine's actual hostname. The client
re-registers (idempotently) on every start, claims a range, runs `namebreak
bounded` against exactly that range, reports the result, and loops. If
`namebreak` doesn't exit cleanly (crash, CUDA error, wrong args), the client
skips reporting completion and lets the range's lease expire so the server
reassigns it - it won't report success or silently drop bad work.

## Server configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite://namebreak.db` | SQLite connection string |
| `ADMIN_TOKEN` | *(required)* | protects `/admin/*` |
| `BIND_ADDR` | `0.0.0.0:8080` | listen address |
| `TARGET_CHUNK_SECONDS` | `900` | desired wall-clock time per range |
| `DEFAULT_RATE_PER_SEC` | `500000000` | assumed candidates/sec until a user's first completed range refines it |
| `MIN_CHUNK_CANDIDATES` / `MAX_CHUNK_CANDIDATES` | `1000000` / `200000000000` | clamp on carved chunk size |
| `LEASE_GRACE_MULTIPLIER` | `3.0` | lease length = this × expected chunk duration |
| `RECLAIM_INTERVAL_SECS` | `30` | how often expired leases are swept back to pending |
| `EMA_ALPHA` | `0.3` | smoothing factor for each user's observed-rate average |

## Deploying to fly.io

```sh
cd coordinator
fly launch --no-deploy   # picks up fly.toml; say no to Postgres/Redis add-ons
fly volumes create namebreak_data --size 1   # matches the [[mounts]] in fly.toml
fly secrets set ADMIN_TOKEN=<a real secret>
fly deploy
```

The client is not part of the server image - volunteers build/run it locally
against `--server-url https://<your-app>.fly.dev`.
