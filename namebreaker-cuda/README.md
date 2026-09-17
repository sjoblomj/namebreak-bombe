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
(or overridden by passing it as the program's first argument, e.g.
`./namebreak bounded`):

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

`run.sh` is a working example: it regenerates `config.conf`'s `[search]`
section from a few shell variables (recomputing `start_candidate` from the
last line of `matches.txt` each time) and launches `continuous` mode.

## Compiling

Requires the CUDA Toolkit (`nvcc`) and a CUDA-capable GPU.

```sh
make                          # builds ./namebreak
make ARCH=sm_75               # for a different GPU (see nvidia-smi --query-gpu=compute_cap --format=csv)
make NETWORK=0                # omit coordinator mode / the libcurl dependency
```

The alphabet's *size* (not its exact characters) is baked into the binary as
a compile-time template instantiation per size, for performance - the fixed
set of sizes this build supports (`42, 43, 47, 48, 49, 50`) is checked early
and fails fast with a clear message if `config.conf`'s alphabet doesn't
match one of them. Supporting a new size means editing the dispatch table in
`namebreak.cu` and recompiling.

`make test` runs the correctness test suite (a pure-CPU unit test plus an
end-to-end GPU test comparing real search results against an independent
brute-force reference); `make bench` / `make search_bench` run throughput
benchmarks used to tune the GPU/CPU split described below.

## Design decisions

### The GPU doesn't brute-force the whole candidate

Only a small, fixed-size trailing window of each candidate (`gpuWindowChars`
characters, currently 4) is generated and hashed directly on the GPU with a
native 64-bit index. Anything beyond that is treated as an extension of the
prefix: every combination of those leading characters is enumerated on the
CPU and its hash contribution folded in once (incrementally - O(1)
amortized per step, not a full rehash) before a batch of GPU threads ever
launches. This isn't primarily about correctness (a much larger window would
still fit in a `uint64_t`) - it's throughput: a wide window means every GPU
thread in a batch redundantly re-hashes whatever leading characters are
actually constant across that whole batch, and shrinking it to 4 roughly
doubled measured throughput for this project's alphabet.

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
candidate's leading characters (everything except the last `gpuWindowChars`
of it).
