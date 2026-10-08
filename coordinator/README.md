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

Visit the server's base URL in a browser (`GET /`) for a live dashboard - every
target with its bound filenames, prefix, suffix, pruning rules and insertions, its ranges, each range's status and who worked on it (from
`last_assigned_user_id`, which - unlike `assigned_user_id` - is never cleared on
reclaim), and a solved target's find called out with a prominent banner. If a
range was reassigned partway through (see progress checkpointing below), each
finished portion shows up as its own row credited to whoever actually searched
it, rather than the whole thing appearing under just the most recent claimer.
That's the **Targets** tab. The **Matches** tab lists every name found, newest
first - who found it and when, its Hash A and Hash B, and the targets it
solved (targets sharing a Hash A/Hash B pair share a row). The **Volunteers** tab ranks every username that
has claimed work or found a name by candidates searched (the sizes of its
completed ranges, across all its hostnames - a little generous, since a range
closed early by a find still counts in full), alongside its completed ranges,
hostnames and names found (targets sharing a Hash A/Hash B pair are one find),
and its canaries found out of those handed to it (see below).
The **Introduction** tab explains what namebreaking is. The open tab is kept
in the URL's `#fragment`, so `/#volunteers` links straight to it.
Plain HTML/CSS/JS (`server/static/dashboard.html`, embedded into the binary at
compile time), which loads `GET /api/v1/dashboard` once each time it's opened
(or reloaded) - no build step, no framework. With desktop notifications on,
it checks the much smaller `GET /api/v1/status` every minute for finds, and
loads the dashboard's data again when there's one. `GET /api/v1/dashboard`
and `GET /` are both public (no auth), matching `GET /api/v1/status`.

See the top-level plan/design notes for the full rationale; the short version:

- **Auth**: `/register {username, hostname, protocol_version}` (no password)
  hands back an opaque bearer token. Every other endpoint requires it - this
  is the only thing standing between a real client and a generic bot scraping
  the API, so it's intentionally simple rather than absent.
- **Protocol versioning**: every client declares its own `protocol_version`
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
  reported progress discarded.
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
  `DELETE /api/v1/admin/skip-ranges/{id}` deletes it, and nothing it
  matched stays skipped. Its rows that carving hasn't reached yet are
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
  - or a custom one (`alphabet` instead of `alphabet_name`, see below)
  (`server/src/alphabet.rs`'s `PREDEFINED_ALPHABETS`, also listable via
  `GET /api/v1/alphabets`) - variations on the default 49-character set, with or
  without brackets/backslash and with a reduced punctuation set, currently
  `size50`, `size49`, `size48`, `size47`, `size43`, `size42`, `size41`,
  `size40`, `size30` and `size29`. Clients of protocol 1.3 and later search
  an alphabet of any size from 1 to 63; older ones' CUDA backend only the sizes
  above, which it had compiled in. So a new alphabet of a new size is a new
  `PREDEFINED_ALPHABETS` entry tagged `(1, 3)`, or a custom alphabet: older
  clients are then given a larger alphabet they know instead (see
  `alphabet::client_alphabet_for`), or no work from that target. The client's CUDA kernel has the sizes 42 and 43
  compiled in and takes any other at runtime, which
  costs about 2% with `prune_whole_candidate` and nothing without it.
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
  it lands in a candidate decides what it costs a client - see the client
  README's "Inserted text".
- **Whole-candidate pruning**: by default, those rules only look at a
  candidate's leading characters - all but the last five or so, which the
  client enumerates on its CPU. A target with `prune_whole_candidate`
  (default `false`) has them look at every character but the last: about a
  fifth fewer candidates are searched at every length, and a GPU client
  searches about 10% faster. Of the real names in the client's StarCraft
  listfile, one breaks a rule before its last character (see the client
  README's "Design decisions"). A client too old to know the setting
  searches as if it were off - more than it has to, never less.
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
  **Dictionary targets** below). Its ranges are ranges of candidate numbers,
  numbered as the client numbers them (`server/src/dictionary.rs`, a copy of
  the client's `engine/dictionary.h`). Only clients of protocol 1.5 and later
  get their work. With `send_basenames`, clients also send every candidate
  whose basename matches the target's encryption key, which the server
  keeps and the dashboard shows.
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
    "prune_whole_candidate": true,
    "alphabet_name": "size49",
    "max_backslash_count": 0,
    "min_backslash_count": 0,
    "prune_adjacent_backslashes": true,
    "insert_from_end": ["\\", 4],
    "priority": 0,
    "description": "From the <b>1998</b> demo listing",
    "start_len": 1,
    "auto_priority": false,
    "encryption_key_hex": "0xD9AC2EFF"
  }'
```

For a custom alphabet, give `"alphabet"` with its characters instead of
`"alphabet_name"` - e.g. `"alphabet": " -.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'"`
(1 to 63 of them, printable ASCII, no lowercase letters, none twice; in any
order, as they're stored sorted). It gets a name made from its characters,
`custom-<size>-<hash>`, which the dashboard shows. Only clients of protocol
1.3 or later are given it as it is; an older client gets the smallest
predefined alphabet it knows that has all of its characters - searching more
than it needs to, never less - or no work from that target. The same goes for
`PATCH`: `"alphabet"` changes a target to a custom alphabet the way
`"alphabet_name"` changes it to a predefined one.

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

`start_len` (default 1) is the shortest candidate length the server carves -
every shorter candidate is left out entirely. It must be between 1 and the
max supported length. There's no `max_len`: the server always searches up to
length 16, the longest candidate the client will search (`MAX_CANDIDATE_LEN`
in `client/src/engine/limits.h`, mirrored by `alphabet::MAX_CANDIDATE_LEN`) -
tightened at every length in between by `lower_bound`/`upper_bound`, which is
what keeps the longer lengths searchable at all. Past length 11 a length has
more candidates than a 64-bit integer holds, so the server stores each
position per leading prefix - which prefix (a block number) plus an index
among that prefix's candidates - the same leading/trailing split the client
makes. `alphabet_name` defaults to `"size49"` if omitted; see
`GET /api/v1/alphabets` for the full list.

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

`encryption_key_hex` and `base_file_name` (both optional) record what's known
about an encrypted target file beyond its two hashes. They're shown on the
target's dashboard card; the search doesn't use them. `encryption_key_hex` is
the file's encryption key, in hex like the hashes, without the adjustment
some files' keys get for their position and size in the archive - mpqcli's
`encryption-key-raw`. That key is made from the file's name without its
directory, so it's the same for every file of that name, in any directory.
`base_file_name` is that name without the directory, when it's known but the
directory isn't - e.g. because the key is that of a known name. It can't
have a `\` or `/`.

### Dictionary targets

A dictionary target's candidates are made of words: one to `max_words` of
them, with a separator between each two - every word from its word lists,
merged (normalized as Storm hashes names - letters uppercase, `/` as `\` -
sorted, and without duplicates), every separator from its `separators`. A
filename is prefix + candidate + suffix, so `MUSIC\` + `BATTLE` + `_` +
`THEME` + `.WAV`. It's the client's dictionary mode (see the client README's
"Dictionary mode"), shared out.

`english-1`, the dictionary compiled into the client, is built into the
server too (`server/data/english-1.txt`, a copy of
`client/data/english-1.txt` - a test checks the two are the same), and
stored as a word list whenever it starts, so a target can use it as it is.
Store any other word list first, one word per line - read as the client
reads them, so blank lines, lines starting with `#` and lines with anything
but printable ASCII are left out (the answer says which lines were):

```sh
curl -X PUT localhost:8080/api/v1/admin/word-lists/sc-units \
  -H 'X-Admin-Token: devsecret' --data-binary @sc-units.txt
# {"name":"sc-units","word_count":212,"checksum":"5e0c...","created_at":...}
```

A word list never changes once stored: clients keep a copy of each one they
download, under its name, so the same name again is only accepted with the
very same file, and a different list needs a different name (409
otherwise). The `english-` names are kept for the dictionaries built into
the server and the client: they can't be uploaded or deleted. A client uses
its own `english-1` rather than download it. A name is 1 to 64 letters,
digits, `.`, `-` and `_`.
`GET /api/v1/admin/word-lists` lists them; `DELETE
/api/v1/admin/word-lists/{name}` deletes one no target uses. Clients
download them from `GET /api/v1/word-lists/{name}`.

Then the target - `dictionary` instead of the alphabet and its settings:

```sh
curl -X POST localhost:8080/api/v1/admin/targets \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{
    "name": "music-words",
    "prefix": "music\\", "suffix": ".wav",
    "hash_a_hex": "0x216A81D3", "hash_b_hex": "0x5A1F2C3B",
    "dictionary": {
      "word_lists": ["english-1", "sc-units"],
      "separators": ["", "_", "-", " "],
      "min_words": 1, "max_words": 2
    },
    "lower_bound": "MUSIC\\BG", "upper_bound": "MUSIC\\BH",
    "encryption_key_hex": "0x1D5AD26C", "send_basenames": true
  }'
```

- `word_lists`: stored lists' names. `separators`: at least one, each once,
  printable ASCII - `""` writes words together. `min_words` (default 1) to
  `max_words`: at most 8, and no more candidates than 2^63 - two words of
  `english-1` with four separators are 16.3 billion, three 4.2 * 10^15.
- `lower_bound`/`upper_bound`: whole filenames this time (prefix, candidate
  and suffix), inclusive, either or both left out for none. Candidates are
  numbered with their first word slowest, so bounds mostly pick out first
  words: the server hands out, for each number of words, the numbers from
  the first first word the bounds let in to the last (exactly, for one-word
  candidates), and the client skips any candidate in between that's outside
  (`AB_` sorts after `ABC`). A target whose bounds leave no candidate is
  refused.
- `prefix`, `suffix`, separators and bounds are stored normalized, as the
  client normalizes them.
- `send_basenames` (default false, needs `encryption_key_hex`): clients
  compare every candidate's basename - what follows its last `\` - to the
  key and send every one that matches, so a file's name is found even when
  the directory searched is the wrong one. About one candidate in 2^32
  matches by chance: some four in two words of `english-1` with four
  separators, about a million in three (see **Basenames** below). Without
  it, clients don't compare basenames at all.
- `name`, `priority`, `description`, `encryption_key_hex` and
  `base_file_name` are as for any target; the alphabet, the pruning rules,
  the insertions, `start_len` and `auto_priority` aren't for a dictionary
  target (400), and nor are priority and skip ranges.

`PATCH` takes `name`, `status`, `priority`, `description`,
`encryption_key_hex`, `base_file_name` and `send_basenames` - the words,
separators, word counts, prefix, suffix and bounds can't be changed.

A dictionary target's ranges are ranges of candidate numbers, sized by each
client's rate at dictionary searches, measured apart from its rate at
alphabet ones (`users.ema_dictionary_rate_per_sec`; a GPU searches some 20
billion a second, a CPU some 0.2 billion) - `DEFAULT_DICTIONARY_RATE_PER_SEC`
until it's measured. A range never spans two numbers of words, which is its
`candidate_len` (the dashboard's Words column). A claim of one carries the
word lists' names and checksums, the separators, the word counts, the range's
first and end numbers, the bounds and, with `send_basenames`, the key (see
`ClaimResponse::dictionary` in `protocol/src/lib.rs`); the client checks its
copy of each word list against its checksum, and the merged words against
theirs, before it searches. Heartbeats and a quit report progress as a
candidate number (`next_candidate_number` - every candidate numbered below it
has been searched) rather than a Hash A match, so a dictionary range is
checkpointed at every heartbeat, and a quit splits it exactly there. Clients
older than protocol 1.5 never get a dictionary target's work; nor is a
canary made from one.

**Basenames.** A client sends the basenames it finds with its next
heartbeat, quit or completion, at most 5,000 a report
(`MAX_BASENAMES_PER_REPORT`), keeping the ones the server hasn't got yet in
memory until a report gets through. A report never says the search has got
further than a basename it doesn't carry: its `next_candidate_number` is at
most where the first one it leaves waiting was found. A completion says the
whole range was searched, so the client sends more than it can carry in
heartbeats first, one straight after another. So whatever happens to the
client - a crash, a quit, a lost connection - a basename the server never
got lies in a part of the range it will hand out again, and is found again:
none is ever missed, and the client keeps no files of them. The
server keeps each basename once per target, with who sent it first and in
which range, whatever else the report gets - a 409 included, as it's a fact
about the target, like a find - so a client counts them delivered on any
answer but an error. A basename that can't be one (empty, longer than 255
characters, with a `\` or anything but printable ASCII) is left out, and
nothing else is done with them: they're kept for whoever searches them. The
dashboard marks a dictionary target with a Dictionary badge, and gives it a
Basenames tab - all of them, up to the first 100,000 - and a Dictionary tab
with its word lists' words (`english-1` is only mentioned). `GET
/api/v1/targets/{id}/basenames` (public, like the dashboard) lists all of
them, one per line, in the order they came, and `GET
/api/v1/targets/{id}/word-lists/{name}` (public too) a word list the target
uses, as uploaded.

There can be a lot of them: about one candidate in 2^32 matches a key by
chance, so a target of three words of `english-1` with four separators
(4.2 * 10^15 candidates) gets about a million - 31 MB as text. So they're
kept compressed (`server/src/basenames.rs`): each report's new ones as one
gzip member (`basename_reports`), and an 8-byte hash of each
(`basename_hashes`) to keep each once. Measured with a million of them sent
240 a heartbeat, as a GPU client sends them: 33 MB in the database - 17 MB
of reports and 15 MB of hashes - where a row each took 100 MB; the Fly
volume is 1 GB. The server is built for that many: each report adds at most
5,000, in one transaction; a target's count is kept as they come
(`targets.basename_count`) rather than counted; the dashboard reads only a
target's latest reports, for its 20 latest basenames; and the full list is
read and sent 20 reports at a time (`BASENAME_REPORTS_PAGE`), so it's never
all in memory, and the database - one connection, which every claim and
heartbeat needs too - is only held for a page's query, however slowly the
list is downloaded. Two words make only a handful; `send_basenames` is off
by default, so a target only collects them when it's asked to.

Every answer is gzipped for a client that says it takes gzip
(`Accept-Encoding` - a browser, `curl --compressed`, the client's word list
downloads): the full list of a million basenames is then 11 MB rather than
31, and the dashboard's JSON and the word lists shrink about as much.

Skip part of a target's search space (see **Skip ranges** above), or
prioritize part of it:

```sh
curl -X POST localhost:8080/api/v1/admin/targets/1/skip-ranges \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"pattern": "[M-Q]", "length": 8, "reason": "Searched offline in 2024"}'
curl -X POST localhost:8080/api/v1/admin/targets/1/priority-ranges \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"pattern": "_[A-Z]", "length": 9, "priority": 10}'
```

Check progress:

```sh
curl localhost:8080/api/v1/status
```

Pause/resume a target, and/or change its name, priority, description,
alphabet_name (or a custom alphabet), prune_symbol_runs, prune_unopened_brackets,
prune_whole_candidate, max_backslash_count, min_backslash_count,
prune_adjacent_backslashes, insert_from_start, insert_from_end, start_len,
auto_priority, encryption_key_hex or base_file_name:

```sh
curl -X PATCH localhost:8080/api/v1/admin/targets/1 \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"status": "paused", "priority": 5, "description": "<b>Bumped</b> for the weekend"}'
```

Any field can be omitted to leave it unchanged (pass `"description": ""` to
clear an existing one, and `null` to remove `insert_from_start`,
`insert_from_end`, `encryption_key_hex` or `base_file_name`), but at least one
must be given. A changed
`prune_symbol_runs`/`prune_unopened_brackets`/`prune_whole_candidate`/`max_backslash_count`/`min_backslash_count`/`prune_adjacent_backslashes` affects every range claimed
after the patch (including already-carved pending ones); ranges already in
progress finish with the old setting. So does a changed `insert_from_start` or
`insert_from_end`, except that the progress such a range reports, with the old
text, is then ignored. Raising `start_len` past where carving
has reached makes it jump straight to the start of the new length the next
time it carves, leaving the rest of the shorter lengths uncarved (shown as a
gap on the dashboard). Lowering it never moves carving back, so lengths it
already passed or jumped over stay that way. Ranges already carved, including
pending ones at shorter lengths, and priority ranges are unaffected.

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

### Canaries

Now and then (`CANARY_PROBABILITY` of claims, while there's real work), a
claim gets a canary instead: a small range, a few seconds' worth at the
client's measured rate (`CANARY_SECONDS`), of a virtual target named
`virtual-canary` whose hashes are those of a filename planted in that range.
To the client it's ordinary work, so finding it checks each volunteer's actual
hardware, driver and build: bit flips (consumer GPUs have no ECC memory), a
driver bug, a broken release, a modified client. The canary copies a real
target's prefix, suffix, alphabet and pruning, so it's searched the way real
work is; the planted candidate has only letters and digits, so no pruning rule
skips it. Its result goes in the `canaries` table (`found` or `missed`; a missed
one is also logged as a warning). Virtual targets (`targets.is_virtual`) are
left out of everything else - other claims, the dashboard's targets and
ranges, `/status`, the rate estimate, and a volunteer's candidates, ranges and
names found - and the Volunteers tab shows them only as "Canaries
found/total". A canary still being searched isn't counted yet; one given up on
(released or reclaimed) counts as not found.

## Server configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite://namebreak.db` | SQLite connection string |
| `ADMIN_TOKEN` | *(required)* | protects `/admin/*` |
| `BIND_ADDR` | `0.0.0.0:8080` | listen address |
| `TARGET_CHUNK_SECONDS` | `900` | desired wall-clock time per range |
| `DEFAULT_RATE_PER_SEC` | `500000000` | assumed candidates/sec until a user's first completed range refines it |
| `DEFAULT_DICTIONARY_RATE_PER_SEC` | `1000000000` | the same for a dictionary target's ranges, measured apart |
| `MIN_CHUNK_CANDIDATES` / `MAX_CHUNK_CANDIDATES` | `1000000` / `100000000000000000` | clamp on carved chunk size |
| `LEASE_SECONDS` | `21600` (6 hours) | how long a claimed range stays leased after the last sign of life from its client (the claim, then each heartbeat) |
| `RECLAIM_INTERVAL_SECS` | `30` | how often expired leases are swept back to pending |
| `EMA_ALPHA` | `0.3` | smoothing factor for each user's observed-rate average |
| `CANARY_PROBABILITY` | `0.33` | chance that a claim gets a canary instead of real work; `0` turns them off |
| `CANARY_SECONDS` | `5` | how long a canary should take, at the client's measured rate |

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
