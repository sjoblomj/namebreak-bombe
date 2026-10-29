# Meet-in-the-middle targets

A meet-in-the-middle target is for a file whose name is a known prefix, an
**unknown directory** and a known suffix, where the MPQ hash state right
after that directory is already known - for Hash A and for Hash B. That
state is the *meeting point*. It comes from the other files of the same
unknown directory: two whose basenames are known pin it down, by trying all
2^32 values of `seed2` against the first name's hash and checking each
survivor against the second.

The client searches such a directory from both ends at once (the client's
[Meet-in-the-middle mode](../../client/docs/meet-in-the-middle.md) has the whole argument): forward from the
prefix into a **table** of every directory of its first `table_length`
characters, by the hash state each one reaches, and backward from the
meeting point, every **tail** - the directory's last characters - rewound
character by character and looked up in that table. A directory of *n*
characters is therefore one of the table's prefixes followed by a tail of *n
- table_length* characters, and the work is the tails: `alphabet^(n -
table_length)` of them, not `alphabet^n` candidates.

```sh
curl -X POST localhost:8080/api/v1/admin/targets \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{
    "name": "setupdat-bwunin",
    "prefix": "", "suffix": "\\BWUNIN.PIF",
    "hash_a_hex": "0x5BAB63FE", "hash_b_hex": "0x8DF2CB08",
    "alphabet": " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_",
    "prune_symbol_runs": true, "prune_adjacent_backslashes": true,
    "meet_in_the_middle": {
      "meeting_point_a_seed1_hex": "0x92F62345", "meeting_point_a_seed2_hex": "0xB9F61953",
      "meeting_point_b_seed1_hex": "0xB95115E6", "meeting_point_b_seed2_hex": "0x7609B4F4",
      "min_length": 6, "max_length": 9, "table_length": 4
    }
  }'
```

- `meeting_point_a_seed1_hex`/`meeting_point_a_seed2_hex` and the `_b_` pair: the hash
  state after the directory, for Hash A and Hash B. Hashing the suffix onto
  them must give the target's two hashes, or the target is refused, saying
  which doesn't fit - a meeting point for the wrong position in the name would
  otherwise have every client search the whole space and find nothing.
- `min_length` (default 1) to `max_length`: how long the directory may be,
  at most 16 characters, and no more tails of one length than 2^63.
  `min_length` can't be less than `table_length`: a directory shorter than
  the table has no tail to hand out, and there are few enough of those to
  sweep with an ordinary alphabet target.
- `table_length`: how many of a directory's first characters the table
  holds, 0 (no table at all - the walk does everything) to `min_length`.
  Every character of it divides the tails by the alphabet's size and
  multiplies the table's memory by it. For the 43-character alphabet above:

  | `table_length` | Table | Tails for a 9-character directory |
  |---|---|---|
  | 3 | 0.7 MiB | 271 G |
  | 4 | 30 MiB | 6.3 G |
  | 5 | 1.4 GiB | 147 M |

  A table of more than 64 GiB is refused: no volunteer would hold one, and a
  table one character longer than meant is an easy mistake to make.

The target's `alphabet` (predefined or custom), `prefix`, `suffix` and
pruning rules work as an alphabet target's, except that the rules always
apply to the whole directory - `prune_whole_candidate` isn't a meeting point
target's, nor are the bounds, the inserted text, `start_len`,
`auto_priority`, `send_basenames`, or priority and skip ranges (what it
hands out is tails, not candidates a pattern can pin down). An
`encryption_key_hex` and a `hash_table_position` are, as for any target: a
match of both hashes is checked against them (see [False matches](targets.md#false-matches)). The
key is made from the basename, which is the suffix's here, so it can't tell
one directory from another - the hash table position is what can.
`PATCH` can change its priority, description, status and pruning rules, but
not its alphabet or lengths: those are what its tails are numbered by, so
searching the same file another way means another target.

**Memory.** The table is the whole memory cost, and the same for every
client of the target - the ranges mean nothing without it. A client says how
much it will give one when it registers; a client that won't give this
target's table that much gets no work from it at all, and neither does one
from before protocol 1.6, which doesn't say. For volunteers with less
memory, make a second target on the same hashes with a shorter table: the
dashboard already shows targets that share a Hash A/Hash B pair as one find.

**Ranges.** A range is a stretch of one directory length's tails -
`candidate_len` is the length, and the positions are tail indices, numbered
by the tail *reversed* (the directory's last character turning slowest, as
the client rewinds it first). The lengths are swept shortest first, a chunk
at a time, and a chunk is sized from the client's measured tails a second
(`ema_meet_in_the_middle_rate_per_sec`, measured apart from the other kinds
of search). A heartbeat's `next_tail_index` is the progress checkpoint, as a
dictionary range's `next_candidate_number` is - and unlike an alphabet
search, a meet-in-the-middle search always knows exactly how far it has got,
so every heartbeat carries real progress and a reassigned range is always
split where its client actually left off.

The client builds the table afresh for every range it claims (a few seconds
for four characters, a quarter of a minute for five), so
`TARGET_CHUNK_SECONDS` wants to stay comfortably above that - the default
900 is fine.
