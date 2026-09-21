# namebreak

A GPU-accelerated MPQ filename brute-forcer. MPQ archives (used by Blizzard
games like StarCraft, Diablo, and Warcraft III) index their files by a pair
of 32-bit hashes of the filename rather than storing filenames directly, so
recovering an unknown filename means finding a candidate string that hashes
to both target values. `namebreak` does that by generating every candidate
in a given alphabet and length range and hashing each one on the GPU via
CUDA. For background on MPQ hashing and why this works at all, see
[zezula.net's writeup](http://zezula.net/en/mpq/namebreak.html).

Every filename `namebreak` searches has the shape **prefix + candidate +
suffix** - the prefix and suffix are fixed, known strings (e.g. `REZ\` and
`.WAV`), and the candidate is the unknown part being brute-forced.

## Modes

`namebreak` runs in one of three modes, set via `config.conf`'s `mode = ...`
(or overridden by passing `--mode <mode>`, e.g. `build/namebreak --mode bounded`).
A different config file can be used instead of `config.conf` via
`--config <file>`.

- **`bounded`** - exhaustively searches candidates of exactly
  `start_candidate`'s length, between `lower_bound` and `upper_bound`, then
  exits.
- **`continuous`** - same as `bounded`, but once a length is exhausted it
  moves on to the next candidate length and keeps going indefinitely (up to
  `MAX_CANDIDATE_LEN`, 16 characters).
- **`coordinator`** - registers with a central coordinator server, claims a
  range of a shared target's search space, searches it, and reports back;
  repeats until stopped. See [`../coordinator/README.md`](../coordinator/README.md)
  for the server side and how this mode is configured.

Exit code is `0` if both hashes matched, `2` if the search space was
exhausted without a match, `1` on any setup/config error.

## Pausing

Press `p` (or `P`) at any point while `namebreak` is running interactively
to pause the search - the same key cgminer/xmrig and other long-running GPU
compute tools use for this. The GPU batch already in flight always finishes
normally; no new one starts until `p` is pressed again to resume. Ctrl+C
pauses too, on the first press - a reflexive Ctrl+C doesn't lose the run
outright, since a pause is trivially undone. A second Ctrl+C, while already
paused by either method, actually quits. In `coordinator` mode,
heartbeating keeps going while paused (it runs on its own thread,
independent of the search itself), so a paused client doesn't lose its
claimed range to a lease timeout - it just makes no progress on it until
resumed. (Pausing doesn't stop the coordinator loop from claiming a *new*
range if there isn't one in progress yet, though - if you pause between
ranges, the next one still gets claimed, and then sits idle, heartbeated
but unworked, until you resume; it just isn't reassigned to someone else in
the meantime.) Only available when stdin is an actual interactive terminal
(not when redirected/piped, or with no controlling terminal at all, e.g. a
cron job or systemd service) - `namebreak` prints `Press 'p' to
pause/resume the search.` on startup when it is.

## Configuring

`namebreak` reads `config.conf` from the current working directory (not a
path you pass in) every time it starts. It's a flat `key = value` file with
up to two `[section]` blocks; `#`-led lines and blank lines are ignored, and
a value only needs `"..."` quoting if it has meaningful leading/trailing
whitespace (a literal `\` needs no escaping).

```ini
mode = continuous

[search]
alphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"
max_backslash_count = 0
prefix = "REZ\\"
suffix = ".WAV"
start_candidate = "REZ\\ .WAV"
lower_bound = "REZ\\FINZ09BX.TXT"
upper_bound = "REZ\\GAMEMENU.BIN"
hash_a = 0xF60F5D90
hash_b = 0xCE0A9BDB
prune_symbol_runs = true
```

`[search]` keys (used by both `bounded` and `continuous` mode):

| Key | Required | Meaning |
|---|---|---|
| `alphabet` | yes | Every character a candidate may contain. Size must be one of `42, 43, 47, 48, 49, 50` (see [Compiling](#compiling) for why) and at most `MAX_ALPHABET_SIZE` (50). |
| `max_backslash_count` | yes | Max `\` occurrences allowed in a candidate before it's skipped; `0` means unlimited. To forbid `\` entirely, leave it out of `alphabet` instead - `0` is "no limit", not "zero allowed". |
| `prefix` / `suffix` | yes | The fixed parts of the filename around the candidate. |
| `start_candidate` | yes | Full filename (prefix+candidate+suffix) to begin searching from - lets a long search resume where it left off. |
| `lower_bound` / `upper_bound` | yes | Full filenames (inclusive) bounding the search alphabetically, at every candidate length searched. |
| `hash_a` / `hash_b` | yes | The two target MPQ hashes, hex (`0x` prefix optional). |
| `prune_symbol_runs` | no (default `false`) | Skip candidates containing 3+ consecutive non-alphanumeric, non-space characters (real MPQ filenames essentially never have runs like that) - see [Design decisions](#design-decisions). |

`[coordinator]` keys (`coordinator` mode only) are `server_url` (required),
plus optional `username`, `hostname` (auto-detected and interactively
confirmed if omitted), and `poll_interval_secs` (default `30`) - see the
coordinator README linked above for the full picture.

Every Hash-A match (a candidate matching the first hash, whether or not it
matches the second) is appended to a matches file, one filename per line.
They all go in one directory - `matches/` in the current directory, or
whatever a top-level `matches_dir = <directory>` line (next to `mode`) says;
it's created if missing. `bounded`/`continuous` mode writes `matches.txt`
there, and `coordinator` mode one `matches-<target name>.txt` per target.

`scripts/run.sh` is a working example: it regenerates `config.conf`'s `[search]`
section from a few shell variables (recomputing `start_candidate` from the
last line of `matches/matches.txt` each time) and launches `continuous` mode.

## Compiling

Requires the CUDA Toolkit (`nvcc`) and a CUDA-capable GPU.

```sh
make                          # builds build/namebreak
make ARCH=sm_75               # for a different GPU (see nvidia-smi --query-gpu=compute_cap --format=csv)
make NETWORK=0                # omit coordinator mode / the libcurl dependency
make BACKEND=cpu              # no CUDA at all: g++ only, searching on the CPU (see Backends below)
```

The alphabet's *size* (not its exact characters) is baked into the binary as
a compile-time template instantiation per size, for performance - the fixed
set of sizes this build supports (`42, 43, 47, 48, 49, 50`) is checked early
and fails fast with a clear message if `config.conf`'s alphabet doesn't
match one of them. Supporting a new size means adding it to
`SupportedAlphabetSizes` in `src/backends/cuda/cuda_backend.cu` and
recompiling.

`make test` runs the correctness test suite: a pure-CPU unit test, plus
end-to-end tests of every backend that compare real search results against
an independent brute-force reference (thousands of cases: every supported alphabet
size, every last-character position, ranges starting/ending mid-row and
crossing launch boundaries, prefix/suffix lengths 0-63 including bytes >=
0x80, the both-hashes-match path, and a seeded fuzzer). The same test source
is built in several configurations (different GPU window / launch sizes),
and each takes about a minute to compile, so `make -j test` is much faster.
`make search_bench` times the real search over a fixed range; `make
WINDOW=6 search_bench` re-runs it with a different GPU window. Everything is
built into `build/`; `make clean` removes it.

## Source layout

| Directory | What's in it |
|---|---|
| `src/engine/` | The search itself: `runSearch()`, candidate/bound arithmetic, hashing on the CPU |
| `src/backends/` | What the search runs its batches on: `cuda/`, and the `cpu/` reference (see Backends below) |
| `src/common/` | The config file, and the few OS-specific helpers (terminal, hostname) |
| `src/net/` | The coordinator client: HTTP, the wire protocol, the claim/heartbeat loop (left out by `make NETWORK=0`) |
| `src/cli/` | The console program's `main()` |
| `src/gui/win32/` | The Windows GUI (`make gui`, Windows only) |
| `tests/` | Correctness tests and benchmarks (`make test`, `make search_bench`) |
| `scripts/` | `run.sh`, and the generator for the GUI's embedded icon |

Includes are written relative to `src/` (e.g. `#include "engine/search.h"`).

### Backends

`runSearch()` (`src/engine/search.cpp`) walks the search space and does
everything that isn't hashing - validation, bounds, the leading/trailing
split, pruning, pause/abort, writing matches. It hands each chunk of
candidates to a *backend* (`SearchBackend`, `src/engine/backend.h`), which
hashes them and returns the hits. `src/backends/backends.cpp` lists the
backends a build has; the program uses the first:

- `cuda/` - the CUDA kernels and the code that launches them.
- `cpu/` - a reference backend: one thread, every candidate hashed from
  scratch. Far too slow for real searches, but simple enough to be obviously
  right. Every build has it, and `make test` runs the end-to-end tests
  against it as well as against CUDA. `make BACKEND=cpu` builds with it
  alone, without any CUDA code - e.g. to check the rest builds and works
  without the CUDA Toolkit.

A new backend (say, for AMD GPUs) implements `SearchBackend`, gets a
directory under `src/backends/` and an entry in `backends.cpp`, and is
covered by the same tests: they take a `--backend <name>` argument.

## Design decisions

### The GPU doesn't brute-force the whole candidate

Only a small, fixed-size trailing window of each candidate
(`NAMEBREAK_GPU_WINDOW_CHARS` characters, currently 5, set in
`src/backends/cuda/tuning.h`) is enumerated directly on the GPU. Anything beyond that is treated as an extension of the prefix: every
combination of those leading characters is enumerated on the CPU and its
hash contribution folded in once (incrementally - O(1) amortized per step,
not a full rehash) before the GPU launches for that leading value. The
window is bounded above (`kMaxTrailingLen`, 6, in the same file) because the
kernel indexes rows with 32-bit integers.

With the current kernel the window is a launch-size knob more than a
per-thread-cost knob (see below), and 5 measured best on the reference
hardware (about 155 G candidates/s at a window of 4, 222 at 5, 216 at 6):
a window of 4 makes each leading value's launch so short (~25 us) that
per-launch overhead becomes a large fraction of the runtime. Note the window
also decides how much `prune_symbol_runs`/`max_backslash_count` can see (they
only examine the leading characters, see below): a larger window means fewer
characters are pruned on, so those two settings skip slightly fewer
candidates than they did at a window of 4 (never more).

### How the GPU kernel is structured

Hashing a candidate means feeding its characters through the hash function
one at a time, each step depending on the one before it. That means two
candidates sharing the same first few characters redo the exact same first
few steps, if hashed independently - wasted, repeated work.

The kernel avoids that by working on **rows** instead of individual
candidates. A row is one fixed value for every trailing character *except
the last* - e.g. with a 3-character trailing window and alphabet `A..Z`,
the row `XY*` covers the 26 candidates `XYA`, `XYB`, `XYC`, ... `XYZ`. One
GPU thread owns one whole row: it hashes the shared part (`XY`) exactly
once, then loops just the last character over the alphabet, reusing that
one result for every candidate in the row. So each of those 26 candidates
costs only one more character step plus the suffix, instead of every one
of them separately re-hashing the whole trailing window from scratch.

One kernel launch doesn't necessarily cover a whole leading value's
trailing space at once, though - it covers at most
`NAMEBREAK_ROWS_PER_LAUNCH` rows (8,388,608 by default, set in
`src/backends/cuda/tuning.h`), and always a whole number of them: a launch's
boundaries land on row boundaries, except possibly at the very start or end of the
range being searched. That's what makes the window above a *launch-size*
knob rather than only a per-thread-cost one - and it's also what bounds how
long any single launch can run for, since pause and abort are only checked
*between* launches, not in the middle of one.

Two more things make that per-row loop fast:

- **It's fully unrolled**, with the last character's alphabet position
  known at compile time for every iteration of the loop. That lets the
  compiler bake each character's hash-table value directly into the
  generated instructions, instead of looking it up from memory at runtime.
  This matters because GPU threads execute in lockstep, 32 at a time (a
  "warp") - a runtime lookup where each of those 32 threads needs a
  different table entry serializes the whole warp, one lookup at a time,
  while a compile-time constant costs nothing extra.
- **The suffix length is a compile-time parameter too** (for lengths 0-8;
  longer suffixes fall back to a slower runtime-length loop), for the same
  reason.

Together, this row-based design measured about 16x faster than the
previous one-thread-per-candidate kernel, while still producing
bit-for-bit identical hashes.

That fast kernel only *records where* each hashA hit is - it doesn't build
the filename or check hashB, keeping its hot path as small as possible. A
second, much smaller kernel (`verifyMatchesKernel`) runs only when a batch
actually had a hit: it rebuilds that candidate's complete filename and
checks hashB - using the *original*, independent hashing code (not
the row-based fast path), so every hit gets cross-checked by a second
implementation at runtime. Any disagreement between the two prints a
`WARNING`, which the test suite treats as a failure.

Only the first `MAX_MATCHES` (1024) hits from one launch can be recorded
at all. If a launch somehow has more than that, its range is searched
again as two halves (recursively, if needed), so every hit still gets
checked against hashB rather than silently lost. With a real 32-bit hash
this essentially never happens in practice (it would take over a thousand
collisions in a single launch), but it's handled explicitly and tested
(`tests/search_overflow_test.cpp`).

### `prune_symbol_runs` and `max_backslash_count` only run on the CPU

Both are cheap heuristics for skipping implausible candidates before
spending a hash chain on them - not correctness rules (a real match could in
principle violate either one; these just trade a small amount of
completeness for a lot of throughput). They're applied only to the CPU-
enumerated *leading* characters described above, never to the GPU-brute-
forced trailing window - and this was a deliberate choice, tried and
measured, not an oversight:

- **Skipping an entire leading value on the CPU is essentially free.** It
  skips that value's whole trailing batch (potentially millions of
  candidates) before any GPU work is launched at all - one CPU-side decision
  is all it costs.
- **Checking the trailing window on the GPU measured as a consistent net
  loss**, at every depth tried, including a narrower version that only
  checked whether the trailing window's first character or two continued a
  run carried over from the leading part (which looked promising on paper -
  that position stays identical across most of a warp, so a `return` there
  was expected to often save a whole warp's worth of work). The reason it
  didn't pay off: GPU threads are grouped into 32-wide warps that execute in
  lockstep, so a `return` only actually saves time if it takes an *entire*
  warp with it - a warp with even one thread still needing to continue pays
  the same cost as if all 32 did. And regardless of whether the check ever
  fires, its own fixed cost (a memory read, a comparison) is paid by every
  thread on every candidate. Against this project's real alphabet, a
  candidate actually tripping either rule is rare enough that this fixed
  cost is paid far more often than it's ever repaid - measured consistently
  slower (2-9%, worse as more of the trailing window was checked) once
  benchmarked properly (warmed-up GPU, multiple samples, no confounding
  cold-start effects from comparing freshly-recompiled binaries).

The upshot: `prune_symbol_runs`/`max_backslash_count` only ever examine a
candidate's leading characters (everything except the trailing window).
