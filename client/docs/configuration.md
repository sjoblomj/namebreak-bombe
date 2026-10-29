# Configuring namebreak

`namebreak` reads `config.conf` from the current working directory, or the
file given with `--config <file>`, every time it starts; without one it
stops and says what the smallest working one looks like (the Windows GUI
runs its setup dialog instead, and writes one; its tray menu's
`Settings...` brings that dialog back later, and what you accept there
replaces what the file said - a field left blank takes its key out
altogether. The run in progress carries on with the settings it started
with, since the file is only read at startup). It's a flat `key = value`
file with up to five `[section]` blocks (`[search]`, `[coordinator]`,
`[dictionary]`, `[meet-in-the-middle]` and `[hardware]`); `#`-led lines and blank lines are
ignored, and a value only needs `"..."` quoting if it has meaningful
leading/trailing whitespace (a literal `\` needs no escaping).

The [`config.conf`](../config.conf) beside the client's source is an example
with every key of every section, each explained in a comment. The sections
of this page cover the top-level keys, `[search]` and `[coordinator]`;
[Dictionary mode](dictionary-mode.md), [Meet-in-the-middle
mode](meet-in-the-middle.md) and [Choosing and tuning the
hardware](hardware.md) have the other sections.

## Top-level keys

These go before the first section.

| Key | Required | Meaning |
|---|---|---|
| `mode` | yes | `bounded`, `continuous`, `coordinator`, `dictionary` or `meet-in-the-middle` - see [Modes](../README.md#modes). `--mode <mode>` overrides it. |
| `backend` | no | The backend to search with - `cuda`, `hip`, `metal`, `opencl` or `cpu`, whichever the build has. Unset, the first that can run on the machine. `--backend <name>` overrides it. |
| `matches_dir` | no (default `matches`) | Where the matches files and the rest the search writes go (see [Matches files](#matches-files)). |
| `check_for_updates` | no (default `true`) | Whether a release build looks for a newer release at startup (see [Version and updates](../README.md#version-and-updates)). |

## `[search]`

An example, for `continuous` mode:

```ini
mode = continuous

[search]
alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"
max_backslash_count = 0
prefix = REZ\
suffix = .WAV
lower_bound = REZ\FINZ09BX.TXT
upper_bound = REZ\GAMEMENU.BIN
hash_a = 0xF60F5D90
hash_b = 0xCE0A9BDB
prune_symbol_runs = true
prune_unopened_brackets = true
resume_from_last_candidate = true
```

`[search]` keys (used by both `bounded` and `continuous` mode):

| Key | Required | Meaning |
|---|---|---|
| `alphabet` | yes | Every character a candidate may contain: any number of them from 1 to `MAX_ALPHABET_SIZE` (63). |
| `max_backslash_count` | yes | Max `\` occurrences allowed in a candidate before it's skipped; `0` means unlimited. To forbid `\` entirely, leave it out of `alphabet` instead - `0` is "no limit", not "zero allowed". |
| `prefix` / `suffix` | yes | The fixed parts of the filename around the candidate. |
| `start_candidate` | no | Full filename (prefix+candidate+suffix) to begin searching from. Without it, the search starts from the beginning: in `continuous` mode the shortest candidates (as if it were just prefix+suffix - that filename itself isn't checked), in `bounded` mode, which searches only one candidate length, `lower_bound`. |
| `resume_from_last_candidate` | no (default `false`) | Resume from the last line of the matches file (see below), the most recent Hash-A match - so a stopped search can simply be restarted to carry on from about where it was. If there's also a `start_candidate`, it starts from whichever of the two the search reaches later (longer candidates come after shorter ones). A last line that doesn't belong to this search - another prefix or suffix, or characters outside `alphabet`, since `bounded`/`continuous` searches share one matches file - is ignored, with a note. Either way, the search prints the candidate it starts from. |
| `lower_bound` / `upper_bound` | yes | Full filenames (inclusive) bounding the search alphabetically, at every candidate length searched. |
| `hash_a` / `hash_b` | yes | The two target MPQ hashes, hex (`0x` prefix optional). |
| `prune_symbol_runs` | no (default `false`) | Skip candidates containing 3+ consecutive non-alphanumeric, non-space characters (real MPQ filenames essentially never have runs like that) - see [Pruning](pruning.md#prune_symbol_runs-prune_unopened_brackets-and-max_backslash_count-only-run-on-the-cpu). |
| `prune_unopened_brackets` | no (default `false`) | Skip candidates that close a bracket never opened: reading left to right, a `)` at a point where more `)` than `(` have been seen, or likewise a `]` with `[`. The two kinds are counted separately (how they nest within each other isn't checked). Brackets left open by `prefix` count as opened, so a candidate may close those - see [Pruning](pruning.md#prune_symbol_runs-prune_unopened_brackets-and-max_backslash_count-only-run-on-the-cpu). |
| `min_backslash_count` | no (default `0`) | Min `\` occurrences a candidate must have; `0` means none needed. Like `max_backslash_count`, the prefix's own don't count, and it must not be more than `max_backslash_count` (unless that's `0`). A candidate is skipped once the characters checked (see `prune_whole_candidate`) leave too few after them to make up the difference, even if every one were a `\` (every other one, with `prune_adjacent_backslashes`) - so without `prune_whole_candidate`, only candidates that need more `\` than the last five or so characters can hold. |
| `prune_adjacent_backslashes` | no (default `false`) | Skip candidates with two `\` next to each other - including a `\` at the very start of the candidate when `prefix` ends with one. |
| `insert_from_start` / `insert_from_end` | no | Text inserted into every candidate, as `text, position` (e.g. `\, 3`; quote the text if it has leading or trailing spaces, `" A ", 3`; at most 16 characters): after the candidate's first `position` characters, or before its last `position`. A candidate shorter than the position gets nothing inserted; both are placed on the candidate's own characters, and where they meet `insert_from_start` comes first. `lower_bound`, `upper_bound` and the length a search covers are without them; the filenames it reports, `start_candidate` and the matches file it resumes from are with them. The pruning rules check inserted text like the candidate's own characters, except text inserted after its last character, which is part of the suffix. See [Inserted text](pruning.md#inserted-text) for what it costs. |
| `prune_whole_candidate` | no (default `false`) | Apply `prune_symbol_runs`, `prune_unopened_brackets`, `max_backslash_count`, `min_backslash_count` and `prune_adjacent_backslashes` to every character of a candidate but the last, instead of only to its leading characters (all but the last five or so). About a fifth fewer candidates are searched, and a search is about 10% faster on a GPU - see [`prune_whole_candidate`](pruning.md#prune_whole_candidate-the-rules-on-every-character-but-the-last). |
| `matches_name` | no | Names the matches file: `matches-<matches_name>.txt` instead of `matches.txt`, so that searches for different files don't share one. Characters other than letters, digits, `.`, `-` and `_` become `_`. |
| `encryption_key` | no | The file's raw encryption key, hex (mpqcli's `encryption-key-raw`): a match of both hashes whose basename doesn't hash to it is a false match (see below). |
| `hash_table_size`, `hash_table_first_start_slot`, `hash_table_last_start_slot` | no (all three or none) | Where the file's entry is in its archive's hash table: a match of both hashes whose lookup doesn't start at a slot from the first to the last is a false match (see below). The size is a power of two. |

## `[coordinator]`

`[coordinator]` keys (`coordinator` mode only) are `server_url` (required),
plus optional `username`, `hostname` (auto-detected and interactively
confirmed if omitted), and `poll_interval_secs` (default `30`) - see the
[coordinator README](../../coordinator/README.md#running-a-client) for the
full picture.

## Matches files

The most recent Hash-A match (a candidate matching the first hash, whether
or not it matches the second) is kept in a matches file: one filename on one
line, replaced as the search goes - at most once a second - so the file
stays a few dozen bytes however long the search runs. A match of both hashes
is always the line it ends with: the search stops there. It's also appended
to `found.txt` beside the matches files, which gets nothing else and is
never replaced, so a later search writing the same matches file can't lose
it. They all go in one directory - `matches/` in the current directory, or
whatever a top-level `matches_dir = <directory>` line (next to `mode`) says;
it's created if missing. `bounded`/`continuous` mode writes `matches.txt`
there (or `matches-<matches_name>.txt`), `dictionary` mode the same and two
more files (see [Dictionary mode](dictionary-mode.md)), `meet-in-the-middle`
mode the same and a progress file (see [Meet-in-the-middle
mode](meet-in-the-middle.md)), and `coordinator` mode one
`matches-<target name>.txt` per target - and, for a dictionary target, the
basenames not sent yet and the word lists it downloaded (see [Dictionary
targets in coordinator mode](dictionary-mode.md#dictionary-targets-in-coordinator-mode)).

## False matches

Two 32-bit hashes match by chance about once in 2^64 names, and a long
search hashes that many - `REZ\SFZSSP050WSRL.WAV` matched both hashes of a
file in the Brazilian Brood War installer and was nothing like its name. So
a match of both hashes is also checked against what's known about the file,
if anything: its encryption key, which the name's basename must hash to
(hash type 3), and where its entry is in the archive's hash table, which
the name's lookup (hash type 0 modulo the table's size) must start close
enough to for Storm to reach. A section of the config gives them as
`encryption_key` and as `hash_table_size`, `hash_table_first_start_slot`
and `hash_table_last_start_slot` (see the tables of keys; in coordinator
mode, protocol 1.7 and later, the coordinator gives them). `first` is the
slot just after the nearest never-used slot before the file's entry, `last`
the entry's own, and the first is the greater if they wrap past the table's
end. A name that fails is a *false match*: printed as `FALSE MATCH`, with why,
appended to `false_matches.txt` beside the matches files (the name, a tab,
and why) - and, in coordinator mode, reported to the coordinator, which
lists it on the target's dashboard card - and the search goes on past it as
if it were any other Hash-A match. Only these checks run on a match of both
hashes, so they cost nothing.
