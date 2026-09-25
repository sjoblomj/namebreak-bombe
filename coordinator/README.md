# namebreak coordinator

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

- **Auth**: `/register {username, hostname, protocol_version}` (no password)
  hands back an opaque bearer token. Every other endpoint requires it - this
  is the only thing standing between a real client and a generic bot scraping
  the API, so it's intentionally simple rather than absent.
- **Protocol versioning**: every client declares its own `protocol_version`
  (`X.Y.Z`) at `/register`; the server rejects registration outright on a
  MAJOR version mismatch against its own `PROTOCOL_VERSION`
  (`coordinator/protocol/src/lib.rs`), asking for an upgrade. A client's
  MINOR version also gates which `PREDEFINED_ALPHABETS` it's offered -
  `GET /api/v1/alphabets` and range-claiming both hide an alphabet from a
  client whose declared version predates that alphabet's own
  `introduced_in` version (`server/src/alphabet.rs`), so an old client never
  gets handed a range in a newer alphabet it doesn't understand.
- **Ranges**: a target's candidate space is carved into contiguous chunks sized
  from each user's observed candidates/sec, so a chunk takes roughly
  `TARGET_CHUNK_SECONDS` regardless of GPU speed. A range that isn't completed
  or heartbeated before its lease expires is released back to pending, to be
  reassigned to someone else. Until someone else actually claims it, the
  client that lost it (e.g. a flaky connection that kept searching offline)
  can take it back just by heartbeating again - it keeps the same range id and
  its recorded progress - or, if it finished the range while offline, have its
  completion accepted as normal. Once another client has claimed it, the
  returning client's heartbeat or not-found completion is rejected and its
  reported progress discarded.
- **Priority ranges**: an operator can flag specific candidate prefixes within
  a target (at one exact length each) as claimable ahead of everything else -
  useful for testing a hunch about the answer without waiting for ordinary
  carving to reach it. `POST /api/v1/admin/targets/{id}/priority-ranges`
  takes a small regex-*style* `pattern` (literal characters, `.`, backslash
  escapes, and `[...]` bracket classes - no quantifiers, alternation, groups
  or anchors), a `length`, and a `priority`; a pattern matching more than one
  character at some position expands into one priority range per concrete
  prefix (capped at `MAX_PRIORITY_PATTERN_EXPANSIONS`, 200, in
  `server/src/alphabet.rs`), each claimed the same way an ordinary range is.
  `DELETE /api/v1/admin/priority-ranges/{id}` removes one (or, if work on it
  already started, retires it instead of deleting it outright).
- **Alphabets**: each target picks one of a small set of predefined alphabets
  (`server/src/alphabet.rs`'s `PREDEFINED_ALPHABETS`, also listable via
  `GET /api/v1/alphabets`) - variations on the default 49-character set, with or
  without brackets/backslash and with a reduced punctuation set, currently
  `size50`, `size49`, `size48`, `size47`, `size43` and `size42`. The set of
  distinct *sizes* (42/43/47/48/49/50) is compiled into namebreak's CUDA backend as
  separate template instantiations (the same zero-cost trick already used for
  `--prune-symbol-runs`), so picking a different alphabet costs no performance -
  but it does mean a genuinely new *size* (not just a new named profile at an
  existing size) requires editing its dispatch and
  recompiling/redistributing the binary to volunteers.
- **Backslash limiting**: each target also has a `max_backslash_count` (default
  `0` = unlimited). `namebreak` discards any candidate with more `\` occurrences
  than this before spending a hash chain on it - the same style of cheap
  pre-filter as `--prune-symbol-runs`'s "no 3 consecutive symbols" rule, just
  targeting one specific character instead. To forbid `\` entirely, use an
  alphabet that doesn't contain it rather than `max_backslash_count: 0` - `0` is
  the "no limit" sentinel, not "zero allowed".
- **Bracket pruning**: a target with `prune_unopened_brackets` (default
  `false`) has `namebreak` discard candidates that close a bracket never
  opened - a `)` or `]` at a point where more brackets have been closed than
  opened - `(`/`)` and `[`/`]` counted separately. Brackets left open by the
  target's prefix count as opened.
- **Progress checkpointing**: every 60s the client heartbeats the most recent
  partial (Hash A only) match `namebreak` has printed for its current range, if
  any. `namebreak` only logs a match after the CUDA batch containing it has
  finished, so everything up to that candidate is known to be searched - the
  server records it as the range's `progress_index`. If the range is later
  reassigned to another client (lease expired, client disconnected) with real
  progress recorded,
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
  this on its next heartbeat (so within 60s) and aborts its in-progress
  search. That range is closed out server-side at the same moment (no
  `/complete` round-trip - there's nothing meaningful to report), and the
  client moves straight on to its next `/claim`.
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
    "prune_unopened_brackets": true,
    "alphabet_name": "size49",
    "max_backslash_count": 0,
    "priority": 0,
    "description": "From the <b>1998</b> demo listing",
    "skip_regex": "[M-Q]"
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

`priority` (default 0, omit for normal priority) decides which target's work
gets claimed first when more than one is active: a worker's `/claim` always
takes a higher-priority target's claimable work - a pending range, or fresh
space still to carve - over a lower-priority target's, no matter which target
is older. It's strict, not weighted: a lower-priority target can go completely
untouched for as long as a higher-priority one still has anything claimable.
Targets sharing a priority (including the default) fall back to plain
creation-order FIFO among themselves.

`description` (optional) is an operator note shown in the target's card
header on the dashboard. It's rendered there as raw HTML, not escaped - tags
like `<b>` come out formatted - so only ever set it from text you trust,
since it's never sanitized.

`skip_regex` (optional) excludes part of the search space from ever being
carved out and handed to a worker. It's matched only against a candidate's
*leading character*, independent of candidate length - `"[M-Q]"` skips every
candidate starting with `M` through `Q`, at every length the target searches.
It must compile as a regex or the request is rejected; it can't match
anything deeper than the first character (e.g. `"^AB"` behaves exactly like
`"^A"` - only the leading character is ever tested). A skipped stretch is
carved as its own range with `status: "skipped"` (shown as "Skip" on the
dashboard) the moment carving actually reaches it - never in advance, and
never retroactively against ranges carved before the regex was set.

Check progress:

```sh
curl localhost:8080/api/v1/status
```

Pause/resume a target, and/or change its priority, description, skip_regex,
alphabet_name, prune_symbol_runs, prune_unopened_brackets or max_backslash_count:

```sh
curl -X PATCH localhost:8080/api/v1/admin/targets/1 \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"status": "paused", "priority": 5, "description": "<b>Bumped</b> for the weekend"}'
```

Any field can be omitted to leave it unchanged (pass `"description": ""` or
`"skip_regex": ""` to clear an existing one), but at least one must be given.
A changed `skip_regex` only affects ranges carved after the patch. A changed
`prune_symbol_runs`/`prune_unopened_brackets`/`max_backslash_count` affects every range claimed
after the patch (including already-carved pending ones); ranges already in
progress finish with the old setting.

Delete a target permanently (also removes its ranges and carving cursor - not
reversible):

```sh
curl -X DELETE localhost:8080/api/v1/admin/targets/1 -H 'X-Admin-Token: devsecret'
```

## Running a client

Build `namebreak` as usual first (see `../client/README.md` - the
default build includes coordinator support; `-DNAMEBREAK_NETWORK=OFF` omits
it). Then, in the directory you want its `matches/` directory in, create a
`config.conf` (see `../client/src/common/config.h`) with a
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

`hostname` (optional) defaults to the machine's actual hostname;
`poll_interval_secs` (optional) defaults to 30. namebreak re-registers
(idempotently) on every start, claims a range, searches it in-process against
exactly that range, reports the result, and loops. If the search fails to
even start (bad claim data, matches file not writable), it skips reporting
completion and lets the range's lease expire so the server reassigns it - it
won't report success or silently drop bad work.

## Server configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite://namebreak.db` | SQLite connection string |
| `ADMIN_TOKEN` | *(required)* | protects `/admin/*` |
| `BIND_ADDR` | `0.0.0.0:8080` | listen address |
| `TARGET_CHUNK_SECONDS` | `900` | desired wall-clock time per range |
| `DEFAULT_RATE_PER_SEC` | `500000000` | assumed candidates/sec until a user's first completed range refines it |
| `MIN_CHUNK_CANDIDATES` / `MAX_CHUNK_CANDIDATES` | `1000000` / `1000000000000000` | clamp on carved chunk size |
| `LEASE_SECONDS` | `21600` (6 hours) | how long a claimed range stays leased after the last sign of life from its client (the claim, then each heartbeat) |
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

The client is not part of the server image - volunteers build `namebreak`
(`../client`) locally and run its `coordinator` mode with
`server_url = https://<your-app>.fly.dev` in their `config.conf`.
