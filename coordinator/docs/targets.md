# Managing targets

Everything under `/api/v1/admin/` is the operator-only side, protected by
`ADMIN_TOKEN` (sent as `X-Admin-Token`). The examples talk to a server
running locally (see [the README](../README.md#running-locally)).

## Adding a target

An alphabet target - [dictionary](dictionary-targets.md) and
[meet-in-the-middle](meet-in-the-middle-targets.md) targets have their own
pages:

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
target's dashboard card, and the key is what a match of both hashes is
checked against before it's taken for the file (see [False matches](#false-matches) below) -
a dictionary search also uses it to pick which candidates to hash at all
(see `send_basenames` in [Dictionary
targets](dictionary-targets.md)). `encryption_key_hex` is
the file's encryption key, in hex like the hashes, without the adjustment
some files' keys get for their position and size in the archive - mpqcli's
`encryption-key-raw`. That key is made from the file's name without its
directory, so it's the same for every file of that name, in any directory.
`base_file_name` is that name without the directory, when it's known but the
directory isn't - e.g. because the key is that of a known name. It can't
have a `\` or `/`.

`hash_table_position` (optional, any kind of target) says where the file's
entry is in its archive's hash table:
`{"table_size": 2048, "first_start_slot": 849, "last_start_slot": 851}`.
Storm looks a name up from slot hash type 0 % `table_size` onwards, slot by
slot (wrapping past the end), and gives up at the first slot that has never
been used - so the file's name has to start its lookup somewhere from the
slot just after the nearest never-used slot before the entry
(`first_start_slot`) up to the entry's own slot (`last_start_slot`). The
first is the greater if they wrap past the table's end. `table_size` is a
power of two, as an archive's hash table's is. Shown on the dashboard card,
and checked like the key.

## Skip and priority ranges

Skip part of a target's search space (see **Skip ranges** in [How the coordinator
works](how-it-works.md)), or
prioritize part of it:

```sh
curl -X POST localhost:8080/api/v1/admin/targets/1/skip-ranges \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"pattern": "[M-Q]", "length": 8, "reason": "Searched offline in 2024"}'
curl -X POST localhost:8080/api/v1/admin/targets/1/priority-ranges \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"pattern": "_[A-Z]", "length": 9, "priority": 10}'
```

## Checking progress

```sh
curl localhost:8080/api/v1/status
```

## Changing a target

Pause/resume a target, and/or change its name, priority, description,
alphabet_name (or a custom alphabet), prune_symbol_runs, prune_unopened_brackets,
prune_whole_candidate, max_backslash_count, min_backslash_count,
prune_adjacent_backslashes, insert_from_start, insert_from_end, start_len,
auto_priority, encryption_key_hex, base_file_name or hash_table_position:

```sh
curl -X PATCH localhost:8080/api/v1/admin/targets/1 \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{"status": "paused", "priority": 5, "description": "<b>Bumped</b> for the weekend"}'
```

Any field can be omitted to leave it unchanged (pass `"description": ""` to
clear an existing one, and `null` to remove `insert_from_start`,
`insert_from_end`, `encryption_key_hex`, `base_file_name` or `hash_table_position`), but at least one
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

## Deleting a target

Delete a target permanently (also removes its ranges and carving cursor - not
reversible):

```sh
curl -X DELETE localhost:8080/api/v1/admin/targets/1 -H 'X-Admin-Token: devsecret'
```

## False matches

Two 32-bit hashes match by chance about once in 2^64 names, and a long
search hashes that many: `REZ\SFZSSP050WSRL.WAV` matched a file's Hash A and
Hash B and was nothing like its name. So a match of both hashes is checked
against `encryption_key_hex` (hash type 3 of the name's basename must be
it) and `hash_table_position`, whichever the target has. A name that fails
either is a *false match*, not the file.

Clients of protocol 1.7 and later are given both with every claim, whatever
kind of target it's of, check every match of both hashes against them, and
search on past one that fails - reporting it with their next heartbeat,
quit or completion (`false_matches`). The server keeps each one once per
target; should one pass the target's checks as they are when it's reported
(they were patched since the range was claimed), it's the find, and solves
the target. Older clients are given neither - their claims are as they were
- and take the first match of both hashes for the file, so the server checks
every find itself, whoever reports it. One that fails is kept as a false
match, the target stays unsolved, and the range wasn't searched past it: an
alphabet range's rest, after the name, is handed out again; a dictionary or
meet-in-the-middle range's progress is a number a completion doesn't carry,
so all of its unsearched part goes back - to clients of 1.7 and later only,
as an older one would stop at the same name again.

A target with false matches gets a "False matches" tab on its dashboard
card, listing each one, why it isn't the file, and who reported it when.
