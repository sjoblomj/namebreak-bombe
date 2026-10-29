# namebreak-bombe

[Client for volunteers](https://github.com/sjoblomj/namebreak-bombe/releases) |
[Dashboard for community progress](https://namebreak-coordinator.fly.dev/)

A GPU-accelerated MPQ filename brute-forcer. MPQ archives (used by Blizzard
games like StarCraft, Diablo, and Warcraft III) index their files by a pair
of 32-bit hashes of the filename rather than storing filenames directly, so
recovering an unknown filename means finding a candidate string that hashes
to both target values. `namebreak` does that by generating every candidate
in a given alphabet and length range and hashing each one on the GPU (CUDA,
HIP, OpenCL or Metal) or the CPU. For background on MPQ hashing and why this works at all, see
[zezula.net's writeup](http://zezula.net/en/mpq/namebreak.html).

Every filename `namebreak` searches has the shape **prefix + candidate +
suffix** - the prefix and suffix are fixed, known strings (e.g. `REZ\` and
`.WAV`), and the candidate is the unknown part being brute-forced.

## Getting started

Volunteers can download a ready-built client from the
[releases page](https://github.com/sjoblomj/namebreak-bombe/releases); to
build it yourself, see [Building namebreak](docs/building.md).

`namebreak` reads its settings from `config.conf` in the current directory.
The [`config.conf`](config.conf) in this directory is an example with every
key explained in comments; the smallest file that works is the coordinator
mode one:

```ini
mode = coordinator

[coordinator]
server_url = https://namebreak-coordinator.fly.dev
```

Then run `namebreak` (or, on Windows, `namebreak-gui.exe`, whose first-run
setup writes that file for you). It registers with the coordinator, claims
work, searches it, and reports what it finds - see [Configuring
namebreak](docs/configuration.md) for searching on your own instead.

## Modes

`namebreak` runs in one of five modes, set via `config.conf`'s `mode = ...`
(or overridden by passing `--mode <mode>`, e.g. `build/namebreak --mode bounded`).
A different config file can be used instead of `config.conf` via
`--config <file>`.

It searches on the first backend (see [Backends](docs/backends.md#backends)) that can run
on the machine - normally the GPU. A top-level `backend = <name>` line in
`config.conf`, or `--backend <name>`, picks one instead; running with an
unknown name lists the ones the build has. Which of the machine's GPUs it
searches on - in `coordinator` mode all of them at once - is the
`[hardware]` section's (see [Choosing and tuning the
hardware](docs/hardware.md)).

- **`bounded`** - exhaustively searches candidates of exactly
  `start_candidate`'s length (or `lower_bound`'s, without one), between
  `lower_bound` and `upper_bound`, then exits.
- **`continuous`** - same as `bounded`, but once a length is exhausted it
  moves on to the next candidate length and keeps going indefinitely (up to
  `MAX_CANDIDATE_LEN`, 16 characters).
- **`coordinator`** - registers with a central coordinator server, claims a
  range of a shared target's search space, searches it, and reports back;
  repeats until stopped. A target is searched by alphabet, as `bounded` mode
  searches, by words, as `dictionary` mode does (see [Dictionary targets
  in coordinator mode](docs/dictionary-mode.md#dictionary-targets-in-coordinator-mode)), or from a
  meeting point, as `meet-in-the-middle` mode does - for which `[hardware]`'s
  `meet_in_the_middle_table_memory` says how much memory you'll give its
  table, and without which the server hands out none of that work (see
  [Meet-in-the-middle mode](docs/meet-in-the-middle.md#in-coordinator-mode)).
  See [`../coordinator/README.md`](../coordinator/README.md) for the server
  side and how this mode is configured.
- **`dictionary`** - searches candidates made of words: one word, or several
  with separators between them, from a dictionary compiled into the program
  (`english-1`) and word lists of your own - and, given the file's
  encryption key, records every candidate whose basename matches it. See
  [Dictionary mode](docs/dictionary-mode.md).
- **`meet-in-the-middle`** - for a file whose directory is unknown, but where the hash
  state right after that directory is known: it searches the directory from
  both ends at once, which buys back one character of brute force for every
  character it can keep a table of. See [Meet-in-the-middle
  mode](docs/meet-in-the-middle.md).

Other options: `--list-gpus` lists the GPUs each backend can search on (see
[Choosing and tuning the hardware](docs/hardware.md)), and `--autotune`
measures the OpenCL or Metal backend's launch settings on them and writes
the fastest to `config.conf`.

Exit code is `0` if both hashes matched, `2` if the search space was
exhausted without a match, `1` on any setup/config error.

## Version and updates

`namebreak -v` (or `--version`) prints which release this is - the GitHub
release tag it was built for, like `v2026-09-26`, or `dev` for a build that
isn't a release (see `NAMEBREAK_VERSION` in [Building](docs/building.md)).

At startup, a release build looks up the latest release on
[GitHub](https://github.com/sjoblomj/namebreak-bombe/releases) and, if
there's a newer one, says so and where to get it. It never downloads
anything, and says nothing if the lookup fails. A top-level
`check_for_updates = false` line in `config.conf` turns this off. A build
without networking (`NAMEBREAK_NETWORK=OFF`) never looks.

In `coordinator` mode, the server can refuse a release it considers too old.
namebreak then shows the server's message, saying what to get, and quits.

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
resumed. (A range paused for a day is released all the same, so that the
work isn't held up forever - see `STALL_RELEASE_SECONDS` in the
coordinator's [configuration](../coordinator/docs/deploying.md).) No new range is claimed while paused.

Also in `coordinator` mode, `f` (or `F`) means "finish the current range,
then pause": the client carries on with the range in hand, reports it
complete, and pauses before claiming new work. Pressed while paused, it
resumes the search for the rest of that range; pressed while running, it
just arranges the pause. `p` still pauses and resumes in the meantime
without cancelling it, and pressing `f` again cancels it, so the client
carries on claiming new work as usual. With no range in hand (paused
between ranges, or waiting for work), it pauses right away. The Windows GUI
has the same option as a "Finish current range, then pause" checkbox, and
in its tray menu.

Keys are only available when stdin is an actual interactive terminal
(not when redirected/piped, or with no controlling terminal at all, e.g. a
cron job or systemd service) - `namebreak` prints `Press 'p' to
pause/resume the search.` on startup when it is.

## Documentation

Using the client:

- [Configuring namebreak](docs/configuration.md) - `config.conf`, the
  `[search]` section of `bounded`/`continuous` mode, the matches files and
  false matches.
- [Choosing and tuning the hardware](docs/hardware.md) - the `[hardware]`
  section: which GPUs, how many CPU threads, launch settings and
  `--autotune`.
- [Dictionary mode](docs/dictionary-mode.md) - searching candidates made of
  words, standalone and in coordinator mode, and how the GPU does it.
- [Meet-in-the-middle mode](docs/meet-in-the-middle.md) - finding an unknown
  directory from the hash state after it.
- [Tuning the CUDA or HIP client](scripts/autotune/README.md) - building a
  client with launch settings measured on your own GPU.

Working on the client:

- [Building namebreak](docs/building.md) - compiling, build options and
  releases.
- [Tests and benchmarks](docs/testing.md) - what `ctest` covers, the
  benchmarks, and the mutation experiment that checks the tests.
- [Source layout and backends](docs/backends.md) - where things are, and
  what each backend is.
- [How a search works](docs/design.md) - the leading/trailing split, rows,
  row groups and how the GPU kernels are structured.
- [The lookup filter](docs/lookup-filter.md) - why most candidates are
  never hashed, and why that can't miss a match.
- [Pruning and inserted text](docs/pruning.md) - where the pruning rules
  run, `prune_whole_candidate`, and what inserted text costs.
- [PERFORMANCE.md](PERFORMANCE.md) - speedups tried, measured and still
  open.
