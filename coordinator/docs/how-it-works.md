# How the coordinator works

The design, piece by piece. [Managing targets](targets.md) has the admin
API that sets these up.

- **Auth**: `/register {username, hostname, protocol_version}` (no password)
  hands back an opaque bearer token. Every other endpoint requires it - this
  is the only thing standing between a real client and a generic bot scraping
  the API, so it's intentionally simple rather than absent.
- **Protocol versioning** (now 1.7.0 - dictionary targets came with 1.5,
  meet-in-the-middle targets with 1.6, false matches with 1.7): every client declares its own `protocol_version`
  (`X.Y.Z`) at `/register`; the server rejects registration outright on a
  MAJOR version mismatch against its own `PROTOCOL_VERSION`
  (`coordinator/protocol/src/lib.rs`), asking for an upgrade. A client's
  MINOR version also gates which `PREDEFINED_ALPHABETS` it's given: an old
  client is never handed a range in an alphabet newer than its declared
  version. It searches such a range in the smallest alphabet it does know
  that contains every character of the range's (`client_alphabet_for` in
  `server/src/alphabet.rs`), with the same first and last candidate as
  bounds - so it covers the whole range, plus some candidates only the
  bigger alphabet has, and its chunks are shrunk to match. The range itself
  stays in its own alphabet; a progress checkpoint the client reports is
  rounded down to the last candidate of that alphabet at or before it
  (`floor_index`), so nothing gets skipped. A client that knows no such
  alphabet gets no work from that target at all.
- **Client releases**: clients from protocol 1.1.0 on also send which
  release they are (`client_release`, the GitHub tag they were built for,
  e.g. `v2026-09-26`, or `dev`). `PUT /api/v1/admin/client-releases` sets
  `{"minimum": "v2026-09-26"}` (null or left out clears it; `GET` reads it
  back). A client older than `minimum` is refused at `/register` and
  `/claim` with HTTP 426 and a message saying what to get, which the client
  shows before quitting - a range it already has can still be finished. That
  includes clients from before 1.1.0, which don't say what release they are,
  and already show the server's error message when registration fails. `dev`
  builds are never refused. The setting lives in the database, so changing
  it needs no restart.
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
  reported progress discarded. Heartbeats alone don't hold a range forever:
  one whose heartbeats have reported no new progress for
  `STALL_RELEASE_SECONDS` (a day, by default) - a client paused, or stuck -
  is released back to pending all the same.
- **Priority ranges**: an operator can flag specific candidate prefixes within
  a target (at one exact length each) as claimable ahead of everything else -
  useful for testing a hunch about the answer without waiting for ordinary
  carving to reach it. `POST /api/v1/admin/targets/{id}/priority-ranges`
  takes a small regex-*style* `pattern` (literal characters, `.`, backslash
  escapes, and `[...]` bracket classes - no quantifiers, alternation, groups
  or anchors), a `length`, and a `priority`, and creates one priority range
  (see **Patterns** below), carved and claimed the same way ordinary work is.
  A new priority range may not overlap one that's still handing out work.
  It may start inside work a finished (or deleted) one already handed out,
  and then simply starts right after that work - handy for narrowing in on
  what's left after deleting a priority range partway through.
  `DELETE /api/v1/admin/priority-ranges/{id}` removes one without losing any
  candidates: work already handed out from it stays recorded, the rest goes
  back to the main sweep - except any part the main sweep has already jumped
  past (it never goes back), which stays as priority work so it still gets
  searched. The response says how much went each way
  (`{"deleted": false, "returned_to_main_sweep": 21, "kept": 0}`); `deleted`
  is true only when nothing was ever handed out and nothing had to be kept.
- **Automatic priority ranges**: with `"auto_priority": true` on a target,
  the server makes priority ranges itself, from the leading characters real
  names most often start with - the first 2 characters at length 11, the
  first 3 from length 12 (shorter lengths are quick enough to sweep as they
  are). The prefixes are ranked at startup and again after every find:
  first by how many words in real filenames start with them - those of
  `server/data/sc-listfile.txt` (a copy of `tools/sc-listfile.txt`) and
  every name found so far, split into words and counted like
  `tools/prefix_counter.py` -
  then, for prefixes no filename starts with, by how many English words do
  (`/usr/share/dict/words`, or `DICTIONARY_PATH`; counted like
  `tools/next_letters.py --rank`). Whenever the target has no other priority
  work left, a claim makes one priority range, for the most likely prefix
  not made into one yet at the length the main sweep is on, covering only
  what's within the bounds, ahead of the sweep and not already another
  priority range's - so a target bounded from `J` to `M` only gets prefixes
  from `J` to `M`. Skip ranges still apply inside them. One at a time, so claims stay quick and the dashboard
  shows them as they come. They show as `auto` in the Prio column, with how
  many words start with the prefix on hover; it's also their `priority`.
  An operator's own priority ranges always come first. Once the prefixes run
  out at a length, the main sweep carries on there, jumping over them, and
  the next length starts over. Deleting an automatic one works like any
  other, and its prefix isn't used again. Turning `auto_priority` off makes
  no new ones but leaves those already made.
- **Skip ranges**: the opposite - candidates an operator never wants handed
  out. `POST /api/v1/admin/targets/{id}/skip-ranges` takes the same kind of
  `pattern`, a `length` and a plain-text `reason` (required), and returns
  `{"skip_range_id": 3}`. Everything it matches within the target's bounds
  that no range covers yet is recorded right away as ranges with
  `status: "skipped"` ("Skip" on the dashboard's Progress tab, where
  expanding the row shows the skip range's pattern and reason), and carving
  (the main sweep or a priority range) jumps straight over them. Ranges
  already carved, even pending ones, are left alone. Where a skip range
  overlaps a priority range, skipping wins.
  `DELETE /api/v1/admin/skip-ranges/{id}` deletes it, and nothing only it
  matched stays skipped: what another of the target's skip ranges matches
  stays skipped, as that one's. So to narrow a skip range - `[B-M][ -9]` to
  all but `I` - add the narrower ones (`[B-H][ -9]` and `[J-M][ -9]`) first,
  then delete it. Of the rest, its rows that carving hasn't reached yet are
  removed, so carving searches those candidates as it gets there. The ones
  the main sweep (or, inside a priority range, the priority range) has
  already passed - carving never goes back - are requeued as pending
  ranges, which are handed out a chunk at a time before any fresh carving.
  The response says how many candidates were requeued
  (`{"requeued": 1250}`). A priority range can't be used for this instead:
  one never starts behind the main sweep. On a change of the target's
  alphabet, the rows carving hasn't reached are rewritten in the new
  alphabet the same way.
- **Patterns**: a skip or priority range's `pattern` pins down a candidate's
  leading characters, one per position - `"_[A-Z]"` is every candidate of
  that length starting with an underscore and then a letter. Candidates are
  numbered in alphabet order, leading character first, so everything a
  pattern matches is a few contiguous stretches of positions, and the
  pattern is stored as one range with one segment per stretch. Characters
  that sit next to each other in the alphabet at the pattern's last
  constrained position form one stretch, and anything after that position
  (including trailing `.`s) never splits one - so `"_[A-Z]"` and `"[A-Z]."`
  are one stretch each, while `"[A-Z]_"` is 26. A pattern needing more than
  1000 stretches (`MAX_PATTERN_SPANS` in `server/src/alphabet.rs`) is
  rejected. On an alphabet change, the pattern is matched afresh against the
  new alphabet.
- **Alphabets**: each target picks one of a small set of predefined alphabets
  - or a custom one (`alphabet` instead of `alphabet_name`, see [Managing
  targets](targets.md))
  (`server/src/alphabet.rs`'s `PREDEFINED_ALPHABETS`, also listable via
  `GET /api/v1/alphabets`) - variations on the default 49-character set, with or
  without brackets/backslash and with a reduced punctuation set, currently
  `size50`, `size49`, `size48`, `size47`, `size43`, `size42`, `size41`,
  `size40`, `size30` and `size29`. Clients of protocol 1.3 and later search
  an alphabet of any size from 1 to 63; older ones' CUDA backend only the sizes
  above, which it had compiled in. So a new alphabet of a new size is a new
  `PREDEFINED_ALPHABETS` entry tagged `(1, 3)`, or a custom alphabet: older
  clients are then given a larger alphabet they know instead (see
  `alphabet::client_alphabet_for`), or no work from that target. The client's
  CUDA kernel has the sizes 40, 42 and 43 compiled in and takes any other
  at runtime, which costs about 2% with `prune_whole_candidate` and
  nothing without it.
- **Backslash limiting**: each target also has a `max_backslash_count` (default
  `0` = unlimited). `namebreak` discards any candidate with more `\` occurrences
  than this before spending a hash chain on it - the same style of cheap
  pre-filter as `--prune-symbol-runs`'s "no 3 consecutive symbols" rule, just
  targeting one specific character instead. To forbid `\` entirely, use an
  alphabet that doesn't contain it rather than `max_backslash_count: 0` - `0` is
  the "no limit" sentinel, not "zero allowed". A target can also ask for a
  `min_backslash_count` (default `0` = none needed, at most
  `max_backslash_count` unless that's `0`): a candidate is discarded once
  the characters the client checks leave too few after them to make up the
  difference - so without `prune_whole_candidate`, only those needing more
  `\` than the last five or so characters can hold. And with
  `prune_adjacent_backslashes` (default `false`), candidates with two `\`
  next to each other are discarded - counting a `\` the prefix ends with.
  The prefix's own backslashes never count towards either count. A canary
  copies `prune_adjacent_backslashes` but not `min_backslash_count`, since
  the candidate it plants has no `\`. Clients older than protocol 1.4
  search as if both were off.
- **Bracket pruning**: a target with `prune_unopened_brackets` (default
  `false`) has `namebreak` discard candidates that close a bracket never
  opened - a `)` or `]` at a point where more brackets have been closed than
  opened - `(`/`)` and `[`/`]` counted separately. Brackets left open by the
  target's prefix count as opened.
- **Inserted text**: a target can insert fixed text into every candidate
  long enough for it - `insert_from_start: ["\\", 3]` puts a backslash
  after a candidate's first three characters, `insert_from_end: ["\\", 4]`
  before its last four (1 to 16 printable ASCII characters, at a position
  from 0 to 16). A candidate shorter than the position gets nothing
  inserted; where the two meet, `insert_from_start` comes first. Bounds,
  ranges, positions and counts are all without it - the text only changes
  what a candidate hashes as - while the filenames clients report (their
  heartbeats' progress checkpoints, names found) have it; the server takes it
  back out to read progress. The pruning rules check it like the
  candidate's own characters, except text inserted after a candidate's last
  character, which counts as part of the suffix. Unlike the pruning
  settings, a client too old to know it couldn't just search more: it would
  miss the target. So a target with inserted text gets no clients older than
  protocol 1.4 (`alphabet::INSERTIONS_SINCE`). Canaries insert it too. Where
  it lands in a candidate decides what it costs a client - see the client's
  [Inserted text](../../client/docs/pruning.md#inserted-text).
- **Whole-candidate pruning**: by default, those rules only look at a
  candidate's leading characters - all but the last five or so, which the
  client enumerates on its CPU. A target with `prune_whole_candidate`
  (default `false`) has them look at every character but the last: about a
  fifth fewer candidates are searched at every length, and a GPU client
  searches about 10% faster. Of the real names in the client's StarCraft
  listfile, one breaks a rule before its last character (see the client's
  [`prune_whole_candidate`](../../client/docs/pruning.md#prune_whole_candidate-the-rules-on-every-character-but-the-last)). A client too old to know the setting
  searches as if it were off - more than it has to, never less.
- **Progress checkpointing**: every 60s the client heartbeats the most recent
  partial (Hash A only) match `namebreak` has printed for its current range, if
  any. `namebreak` only logs a match after the GPU batch containing it has
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
- **Quitting**: a client that quits while it has a range (from protocol
  1.3.0 on) says so with `POST /api/v1/ranges/{id}/quit`, sending its last
  partial match like a heartbeat does. The range is split there and then:
  everything up to and including that candidate (or an earlier heartbeat's,
  whichever is further) becomes a `completed` range credited to the client,
  and the rest a `pending` one, handed out ahead of fresh carving. Without
  any partial match, the whole range simply goes back to `pending`. Either
  way the work is available again straight away instead of once the lease
  expires.
- **Stopping on a find**: the same heartbeat also carries a `target_solved`
  flag, true once *any* range of that target has been completed with a match.
  A client still searching a different range of an already-solved target sees
  this on its next heartbeat (so within 60s) and aborts its in-progress
  search. That range is closed out server-side at the same moment (no
  `/complete` round-trip - there's nothing meaningful to report), and the
  client moves straight on to its next `/claim`.
- **Dictionary targets**: a target can be searched by words instead -
  candidates made of words from word lists the server stores, with
  separators between them, as the client's dictionary mode makes them (see
  [Dictionary targets](dictionary-targets.md)). Its ranges are ranges of candidate numbers,
  numbered as the client numbers them (`server/src/dictionary.rs`, a copy of
  the client's `engine/dictionary.h`). Only clients of protocol 1.5 and later
  get their work. With `send_basenames`, clients also send every candidate
  whose basename matches the target's encryption key, which the server
  keeps and the dashboard shows.
- **Meet-in-the-middle targets**: a target can be a file whose *directory* is unknown,
  where the hash state right after that directory - the *meeting point* - is
  known: the client then searches the directory from both ends at once,
  meeting in the middle, which buys back one character of brute force for
  every character it can keep a table of (see [Meet-in-the-middle
  targets](meet-in-the-middle-targets.md) and the client's [Meet-in-the-middle
  mode](../../client/docs/meet-in-the-middle.md)). Its ranges are stretches of one
  directory length's *tails* - the directory without the table's characters
  - numbered as the client numbers them (`server/src/meet_in_the_middle.rs`, the
  server's half of the client's `engine/meet_in_the_middle.h`). The table is what the
  search costs in memory, and it's the same for every client of a target,
  since a tail is whatever the table doesn't cover: a client says at
  `/register` how much it will give one (`meet_in_the_middle_table_bytes`), and gets
  no work from a target whose table wouldn't fit in it - as a client that
  can't search a target's alphabet gets none. Only clients of protocol 1.6
  and later get any of it.
- **Storage**: SQLite on a single Fly Volume. One server instance only - range
  assignment has to be centrally coordinated anyway, so this isn't a real
  limitation.

## Canaries

Now and then (`CANARY_PROBABILITY` of claims, while there's real work), a
claim gets a canary instead: a small range, a few seconds' worth at the
client's measured rate (`CANARY_SECONDS`), of a virtual target named
`virtual-canary` whose hashes are those of a filename planted in that range.
To the client it's ordinary work, so finding it checks each volunteer's
actual hardware, driver and build: bit flips (consumer GPUs have no ECC
memory), a driver bug, a broken release, a modified client. The canary
copies a real target's prefix, suffix, alphabet and pruning, so it's
searched the way real work is; the planted candidate has only letters and
digits, so no pruning rule skips it. When the client's first target by
priority is a dictionary target, the canary copies that instead (once the
client's dictionary rate has been measured): its word lists, separators,
word counts, tails, prefix and suffix, at the word count its sweep is at,
and - if it has an encryption key - a key made from the planted name's
basename, so the client takes the same key-only path real work does (as well
as `send_basenames`; the basenames sent for a canary aren't kept). It leaves
out the target's bounds, which could leave out the planted name. When that
first target is a meet-in-the-middle target, the canary copies it instead:
its alphabet, prefix, suffix, pruning rules and - so that it costs the
client exactly what the real work does, within the memory that client
declared - the same table length. Its meeting point is the state after a
directory planted in the range, and its hashes are that directory's
filename's, so finding it is finding that directory; it searches one length
only, the planted directory's. Its result goes in the `canaries` table
(`found` or `missed`; a missed one is also logged as a warning). Virtual
targets (`targets.is_virtual`) are left out of everything else - other
claims, the dashboard's targets and ranges, `/status`, the rate estimate,
and a volunteer's candidates, ranges and names found - and the Volunteers
tab shows them only as "Canaries found/total". A canary still being searched
isn't counted yet; one given up on (released or reclaimed) counts as not
found.
