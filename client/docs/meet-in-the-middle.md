# Meet-in-the-middle mode

`mode = meet-in-the-middle` is for a file whose name is a known prefix, an
**unknown directory**, and a known suffix - and where the MPQ hash state
right after that directory is already known, for Hash A and for Hash B. That
state is the *meeting point*, and it, rather than the file's hashes, is what
the search looks for: the directory is whatever string takes the hash from
the prefix's state to it.

A meeting point comes from the files *around* the one being searched. When
an archive holds a run of files whose basenames are known but whose
directory isn't - the install scripts of Brood War's `INSTALL.EXE`, say,
where the basenames are in the scripts themselves - then every one of them
is `<the same unknown directory>` + `\` + `<a known basename>`. Two of those
are enough to pin down the state after the directory, by trying all 2^32
values of `seed2` against the first name and checking each survivor against
the second. The four numbers that comes to - `seed1` and `seed2` after the
directory, for Hash A and for Hash B - are what this mode is given.

## Both ends at once

Every step of the MPQ hash

```
seed1 = cryptTable[hash + ch] ^ (seed1 + seed2)
seed2 = ch + seed1 + seed2 + (seed2 << 5) + 3
```

can be *undone* when the character is known. `seed2`'s step multiplies it by
`1 + (1 << 5)`, and 33 is odd, so dividing by it is a plain multiplication
by 33's inverse modulo 2^32 (`0x3E0F83E1`):

```
seed2_before = (seed2_after - ch - seed1_after - 3) * 0x3E0F83E1
seed1_before = (seed1_after ^ cryptTable[hash + ch]) - seed2_before
```

So the search can work from both ends of the directory and meet in the
middle:

- **forward**, from the state after the prefix: every directory of
  `table_length` characters, kept in a table by the state it reaches;
- **backward**, from the meeting point: every *tail* - a directory's last
  characters - rewound character by character, and the state it reaches
  looked up in that table.

A tail of *j* characters whose state is in the table means a directory of
`table_length` + *j* characters: the table's prefix, then the tail. So the
work is about `alphabet^(n - table_length)` tails for a directory of *n*
characters, instead of the `alphabet^n` candidates a plain search hashes -
the table buys back one character of brute force for every character it
holds. Directories shorter than the table are found while it's being built,
by the same walk, and cost nothing extra.

What the table costs is memory: eight bytes per directory of its length,
plus about one more for the filter that keeps most lookups away from the
keys. With the 43-character alphabet above:

| `table_length` | Memory | Tails for an 8-character directory | for 10 | for 12 |
|---|---|---|---|---|
| 3 | 0.7 MB | 147 M | 272 G | 503 T |
| 4 | 31 MB | 3.4 M | 6.3 G | 11.7 T |
| 5 | 1.4 GB | 79.5 k | 147 M | 272 G |
| *(a plain search)* | - | 11.7 T | 21.6 P | 40.0 E |

`table_memory` is what the table may take, and the longest length that fits
in it is used; `table_length` sets it outright instead. A character more
costs as many times more memory as the alphabet has characters, so in
practice the table is 4 or 5 characters long and the walk does the rest.

Every backend searches meet-in-the-middle targets: the CUDA/HIP one,
OpenCL, Metal, `cpu` and `reference`. On the RTX 3080 Ti Laptop, both GPU
backends rewind about 2.2 billion tails a second with the 31 MB table (CUDA
2.20, OpenCL 2.37) and about half that with the 1.4 GB one (1.13 and 1.22),
whose lookups miss every cache; the `cpu` backend does about 66 million.
Halving the lookups' speed to hold one character more is a bargain: that
character divides the tails by 43.

So an 11-character directory takes a 5-character table (13 s to build on the
CPU, 1.4 GB) and 6.5 billion tails (5 s on the GPU): under twenty seconds in
all, where the 43^11 = 9.3 * 10^17 names of that space are about a week of
the same GPU's time to search through. One character further on, where a
plain search is a year, a meet-in-the-middle search is four minutes.

The table lives in the device's own memory, so it has to fit there. A search
whose table is bigger than the GPU has says so and stops, rather than
crawling through host memory - give it less (`table_memory`), or run it on
`--backend cpu`, which uses the machine's own.

## On the GPU

One work-item takes one *row*: the tails that share every character but the
one rewound last - one per character of the alphabet - which is the first
character of the tail. The rest of the rewinding, and the pruning rules over
it, are then done once for the whole row - the mirror of the search kernel's
rows, which share everything but the candidate's *last* character, because
the walk rewinds a tail's last character first and that one the most tails
have in common.

Each of the row's tails then costs one step of the hash undone, one bit of
the table's filter, and - for about one in eight - a binary search of its
keys. A launch covers about 67 million tails
(`NAMEBREAK_MEET_IN_THE_MIDDLE_TAILS_PER_LAUNCH`), reports how many it
looked up and the tails it found, and is launched again with more room if it
found more than it had room for - as a dictionary launch is. The kernels are
`meetInTheMiddleKernel` in `src/backends/cuda/cuda_backend.cu`,
`src/backends/opencl/meet_in_the_middle.cl` and
`src/backends/metal/meet_in_the_middle.metal` (the OpenCL one in Metal's
shading language); `src/backends/common/meet_in_the_middle_launch.h` is
what they share. On a Mac the table lives in memory the CPU and GPU share,
but Metal still caps it: a table bigger than one buffer may be, or than
the GPU's recommended working set, is refused the same way.

## An example

The directory of Brood War's `SETUPDAT\BWUNIN.PIF`, had it not been known,
from the meeting point a recovery like that gives for it:

```ini
mode = meet-in-the-middle

[meet-in-the-middle]
alphabet = " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\_"
suffix = \BWUNIN.PIF
meeting_point_a_seed1 = 0x92F62345
meeting_point_a_seed2 = 0xB9F61953
meeting_point_b_seed1 = 0xB95115E6
meeting_point_b_seed2 = 0x7609B4F4
hash_a = 0x5BAB63FE
hash_b = 0x8DF2CB08
max_length = 8
table_memory = 1G
prune_symbol_runs = true
prune_adjacent_backslashes = true
matches_name = bwunin
```

That finds `SETUPDAT\BWUNIN.PIF` in about half a second, on the CPU or on
the GPU: a 4-character table (31 MB) and 3.5 million tails, where a plain
8-character search of the same alphabet is 43^8 = 11.7 trillion candidates,
about eight seconds of this GPU's own time.

`[meet-in-the-middle]` keys:

| Key | Required | Meaning |
|---|---|---|
| `alphabet` | yes | The characters the unknown directory is made of (at most 63, each once). A `\` among them searches directories with subdirectories in them. |
| `prefix` | no (default none) | Known text before the unknown directory. The meeting point must be the state after the prefix *and* the directory. |
| `suffix` | yes | Known text after it - for a file in that directory, the `\` and the basename. |
| `meeting_point_a_seed1`, `meeting_point_a_seed2` | yes | Hash A's state after the directory, hex - the state the search walks to. |
| `meeting_point_b_seed1`, `meeting_point_b_seed2` | no (both or neither) | Hash B's state after the directory, hex. The search never walks it; it only checks the meeting point against `hash_b` before starting (see below). |
| `hash_a`, `hash_b` | yes | The file's own two hashes, hex. Hashing `suffix` onto the meeting point must give them, or the search refuses to start - see below. |
| `min_length` | no (default 1) | The shortest directory to report. |
| `max_length` | yes | The longest to search, up to `MAX_CANDIDATE_LEN` (16). |
| `table_length` | no | How many of a directory's first characters the table holds. Left out, the longest that fits in `table_memory`. |
| `table_memory` | no (default `1G`) | How much memory the table may take: bytes, or with a `K`, `M` or `G` after it. |
| `prune_symbol_runs`, `prune_unopened_brackets`, `prune_adjacent_backslashes`, `max_backslash_count`, `min_backslash_count` | no | The same cheap rules as `[search]` - but applied to *every* character of the directory, not only its leading ones, and `prune_adjacent_backslashes` also counts a `\` the suffix begins with next to one the directory ends with. |
| `encryption_key`, `hash_table_size`, `hash_table_first_start_slot`, `hash_table_last_start_slot` | no | As in `[search]`. |
| `matches_name` | no | Names the files it writes: `matches-<name>.txt` and `meet-in-the-middle-<name>.txt` instead of `matches.txt` and `meet-in-the-middle.txt`. |
| `resume_from_last_candidate` | no (default false) | Carry on from the progress file - unless that was written by a search with other settings, which it refuses to resume. |

## The meeting point has to fit the hashes

The meeting point and the file's hashes come from different places - the
meeting point from a key recovery, the hashes from the archive - so before
anything is searched, hashing the suffix onto the meeting point must give
the file's two hashes. A meeting point for the state after the wrong
position in the name, or for another file's directory, would otherwise have
the search walk the whole space and quietly find nothing. The search says
which of the two doesn't fit and stops.

It means the two halves of the search are checked against each other too: a
directory that reaches the meeting point's 64-bit state necessarily hashes
to Hash A, and the search reports it, then checks Hash B in full. Both
together leave about one directory in 2^96 to match by chance - the search
says how many it expects before it starts, which for a real search is 0.

Only Hash A's half of the meeting point is actually searched for, so Hash
B's is optional: leave `meeting_point_b_seed1`/`meeting_point_b_seed2` out
(both, or neither - half a state means nothing) and the search runs on Hash
A's state alone. Nothing about what it finds changes, since every directory
that reaches the meeting point is hashed in full and compared to both of
the file's hashes anyway. What is lost is the check above, for the half of
it that catches the subtler mistake: a meeting point taken at the wrong
place in the name still hashes to *something*, and without Hash B's state
to compare it against, such a search walks everything and finds nothing -
which looks exactly like a file that isn't there. The search says so on
startup when they are missing.

## Pruning, and what it may skip

The rules work on the two halves separately. A run of three symbols, two
backslashes in a row and `max_backslash_count` can each be judged from a
piece of the directory on its own, so the table skips a first part that
breaks one, and the walk skips a tail that does - and with it every longer
tail that ends in those characters, which is what makes the rules worth
having here: rejecting one character of a tail skips the whole subtree
behind it.

`min_backslash_count` and `prune_unopened_brackets` can't be judged from a
piece: characters further left could always supply the missing backslash, or
open the bracket the piece closes. Those are checked once a whole directory
is in hand, as every rule is before anything is reported - so a meeting
point search reports exactly the directories that match the meeting point
and pass the rules in full, and the halves' own skipping only ever drops
directories that check would reject anyway. That property is what
`tests/meet_in_the_middle_test.cpp` checks exhaustively, over every
directory of a small alphabet at every split between the two halves.

The Windows GUI's setup has a Meet in the Middle tab with these keys, and
checks them as a search would before it writes them - including that hashing
the suffix onto the meeting point really gives the file's hashes, which
is the mistake worth catching before a search walks the whole space for
nothing. While it searches, its progress bar is how many of the tails it has
rewound, and a directory that matches the meeting point is listed with the
Hash-A matches.

## In coordinator mode

A meet-in-the-middle target can be shared out (the coordinator's
[Meet-in-the-middle targets](../../coordinator/docs/meet-in-the-middle-targets.md)): the server fixes how many characters the
table holds - what a tail even is depends on it - and hands out a stretch of
one directory length's tails at a time. What it costs in memory is therefore
the same for every client of that target, so a client says how much it will
give one in `[hardware]`'s `meet_in_the_middle_table_memory` - the memory
being part of the machine it agrees to spare, as its GPUs are:

```ini
[hardware]
meet_in_the_middle_table_memory = 1G
```

(v2026-10-22 had it in `[coordinator]`, where it is still read; `[hardware]`'s
is used if both have it.)

Left out (or 0), this client is handed no meet-in-the-middle target's work
at all - a table is memory you have to agree to spare, and on a GPU it has
to fit in the device's own as well. It's per GPU: a client searching on
several (see [Choosing and tuning the
hardware](hardware.md)) may have a table this big for
each. The client says which it is when it
registers, and a server hands out no target whose table is bigger than it.
The Windows GUI's first-run setup proposes a size for the machine - an
eighth of its memory, at most 4 GB - and its Hardware window (see
[Choosing and tuning the hardware](hardware.md)) has
it as a slider over the sizes a table can usefully take (None, then powers of two from 256 MB up
to what the machine has), with what this machine has and has free above it
and, under it, how long a table the chosen size holds.
The table is built afresh for every range claimed (a few seconds for four
characters of 43, a quarter of a minute for five), and progress is reported
as the tail the walk has got to, so a range taken over by someone else
resumes exactly there. A range's own percentage is how many of its tails
have been walked - the Windows GUI shows it the same way it does for a
local meet-in-the-middle search, since neither has a candidate order for
the matches file to be measured against.

## Progress, resuming and ranges

The search goes through the backward walk's *nodes*: one per tail, shortest
first, and within a length in the order of the tail *reversed* - the
directory's last character changes slowest, because it's the first the walk
rewinds and so the one whose state the most tails share. A node's number is
what the progress file (`matches/meet-in-the-middle.txt`) records, and what
a range of a search is given in, so `resume_from_last_candidate = true`
carries on exactly where the last run stopped.

Note that the node order is not the order the directories themselves sort
in, as the other modes' candidates are searched - the search reports a
directory when the walk reaches *its tail*, and the tails of one length come
before every longer one. It is also the numbering the coordinator hands out
ranges in (see "In coordinator mode" above).
