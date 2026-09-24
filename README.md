# namebreak-bombe

![namebreak-bombe](namebreak.png)

A GPU-accelerated MPQ filename brute-forcer, plus the tooling around running
it at scale. MPQ archives (used by Blizzard games like StarCraft, Diablo, and
Warcraft III) index their files by a pair of 32-bit hashes of the filename
rather than storing filenames directly, so recovering an unknown filename
means finding a candidate string that hashes to both target values - that's
what the programs here do, by trying every combination of characters in a
given alphabet.

For more info on what name-breaking is, see [here](http://zezula.net/en/mpq/namebreak.html).

## Layout

- [`client/`](client/README.md) - the actively maintained client. Generates
  every candidate in a configured alphabet and length range and hashes each
  one on the GPU (CUDA, HIP, OpenCL or Metal) or the CPU. Runs standalone
  (`bounded`/`continuous` mode) against a fixed range, or as a worker for the
  coordinator below (`coordinator` mode).
- [`coordinator/`](coordinator/README.md) - a server for distributing a
  search across multiple volunteers' GPUs. Tracks targets, carves each one's
  candidate space into time-boxed ranges, and hands them out to
  `client` clients over HTTP, with a live dashboard showing
  progress and who found what.
- `namechecker/` - a small standalone CPU tool for checking whether a given
  name (or every name in a wordlist file), once prefixed/suffixed, matches a
  target hash pair - useful for verifying a candidate by hand, or scoring a
  wordlist offline, without spinning up the GPU search.
- `portrait-cuda/` - an earlier, single-file prototype of the GPU search (no
  config file, no coordinator support, no CPU/GPU candidate split).
  Superseded by `client`; kept around for reference, not actively
  maintained.

## Getting started

Most work happens in [`client/`](client/README.md) (see
its README for configuring and compiling the client) and, for a distributed
search across multiple machines, [`coordinator/`](coordinator/README.md)
(see its README for running the server and pointing clients at it).
