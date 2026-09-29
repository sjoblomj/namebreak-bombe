# Performance task list

Speedups still open for the client, found while adding the lookup filter to
the CUDA backend (see README.md's [The lookup
filter](README.md#the-lookup-filter-most-candidates-are-never-hashed)), plus
the safeguards that would make a missed match even less likely. Each item
says what's known - **measured**, or **estimated** where it hasn't been tried
- and what it would take.

Numbers are from the RTX 3080 Ti Laptop GPU and i9-12900H this was developed
on. Nothing here counts as done until it's measured with `search_bench
--scale 20` (warmed up, and without `noprune` too, so that pruning doesn't
blur a kernel comparison) and the whole `ctest` passes - the dense-hit
`stress-*` tests above all, since every item below touches code that could
silently drop candidates.

Where things stand (measured): the CUDA backend searches 1,084-1,130 G
candidates/s, up from 209-218 before the filter. It issues about 150
instructions per row of 49 candidates, down from about 1,130. Nsight
Systems puts about 5% of the GPU's time between launches (478 ms of kernel
time in a 0.5 s search: 2,044 launches of 233 us each).

## The lookup filter on the other backends

- [ ] **CPU backend.** A prototype (the cpu backend's row walk, with each
  row's last character looked up in the table instead of hashed with SIMD)
  passed the whole test suite and measured **4.6 times as fast**: 19.5 G
  candidates/s against 4.3, on a machine busy enough that both were below
  the README's 9. Tables of 8 or 9 bits per seed were best (512 KB / 2 MB,
  from the CPU's L2); every width from 6 to 10 beat the current backend by
  3-5 times. It's also simpler code than the SIMD it would replace. Later,
  the row states themselves could be computed eight at a time with AVX2,
  and their table entries gathered.
- [ ] **OpenCL.** Port `filteredRowsKernel` to `search.cl`, with the table in
  a global buffer, and the same checks as the CUDA backend (a sample of the
  table checked, and read back, before every search). *Estimated* about the
  same factor as CUDA's; it ran at 88% of the unfiltered CUDA kernel.
- [ ] **Metal.** The same port. It can't be run without a Mac, so it would
  lean on the self-test, which runs on the Mac itself. Its kernel should
  take `HASHA_MATCH_MASK` the way the OpenCL one does, so that the stress
  tests cover it too.
- [ ] **HIP.** Gets the filter with the CUDA code, but has never run on an
  AMD GPU. The self-test now runs there before every use, which lowers the
  risk - but measuring it on real AMD hardware is still to do.

## CUDA backend

- [ ] **Incremental row decoding.** Decoding a row - four divisions by the
  alphabet's size, eight shared-memory loads, four hash steps - is now about
  50 of the 150 instructions a row costs (**measured**, SASS of
  `filteredRowsKernel<49, 4>`). Consecutive rows differ only in their last
  shared character 48 times out of 49, so a thread walking several rows in
  a row would pay one hash step per row instead. *Estimated* +20-30%.
- [ ] **Less divergence in the verify loop.** A warp repeats the loop over
  flagged candidates as many times as its busiest lane needs: about 1.4
  times at 8 bits (*estimated* from the statistics). Options: a 9-bit table
  (2 MB; the GPU's L2 is 4 MB) or 10-bit, or warp-cooperative verification
  (collect the flagged candidates of all 32 lanes with a ballot, then have
  each lane verify one). *Estimated* +10-30%. Sweep
  `NAMEBREAK_LOWBITS_FILTER_BITS` from 6 to 10 first.
- [ ] **Where the table lives.** 512 KB is read through the read-only cache,
  mostly from L2. A 6-bit table (32 KB) would fit in shared memory, if
  blocks were long-lived enough to load it once (a grid-stride loop).
  Measure against the above.
- [ ] **Gaps between launches (~5%, measured).** One launch per leading
  value, synchronized and read back before the next. Either launch several
  leading values at once (a 2D grid over an array of their seeds and
  prefixes), or keep two launches in flight with double-buffered results.
  *Estimated* +5%, and it makes the next item cheap.
- [ ] **Re-tune the window and launch size.** `NAMEBREAK_GPU_WINDOW_CHARS`
  (5) and `NAMEBREAK_ROWS_PER_LAUNCH` were chosen with the kernel as it was
  before the filter, whose launches took five times as long.
- [ ] **Stop spinning a CPU core.** `cudaDeviceSynchronize` busy-waits: the
  running client keeps one core at 100% (**measured** with `ps`/`top`). On a
  laptop, CPU and GPU share one power and cooling budget, so that core may
  cost GPU clock. `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)`
  would free it, at the price of some wake-up latency per launch - fine
  once launches are longer (above). Measure throughput and GPU clocks.
- [ ] **Build the table faster, or less often.** Building it takes 4 ms for
  `.WAV` and 10 ms for a 17-byte suffix, and the check before every search
  0.4-1.6 ms (**measured**) - nothing next to a real range, but it's most
  of the cost of the test suite's thousands of tiny searches. Building it
  on the GPU would take microseconds; or cache it, keyed by *everything* it
  depends on (alphabet, suffix, target hashA, and the filter/match bit
  counts) - in coordinator mode, range after range has the same target.
  Keep the check either way.
- [ ] **Any alphabet size.** The per-candidate loop that needed the alphabet
  size at compile time is gone; it's now only used for the row decode's
  divisions, which could use multipliers computed on the host (as a
  compiler does for constants). That would drop `SupportedAlphabetSizes`
  and its "rebuild for a new size" step, and cut compile time. A feature
  more than a speedup.
- [ ] **Profile.** Nothing here has been checked with Nsight Compute yet:
  whether the filtered kernel is now bound by integer throughput, L2
  latency, or divergence would rank the items above.
- [ ] **Micro-optimizations in the verify path.** The last suffix step's XOR
  and `+ 3` could be folded into the constant it's compared with. They were
  worth about a quarter of the old kernel's instructions, but they're only
  paid on the ~0.2 candidates per row that get through the filter now.

## The search space (a decision, not only a speedup)

- [ ] **Prune the whole candidate.** `prune_symbol_runs` and
  `prune_unopened_brackets` only look at the leading characters. Applied to
  the whole candidate, they'd skip about a fifth more of it at every length
  (**computed** exactly for the real alphabet, both rules on):

  | Length | Searched now | Rules on every character but the last | On every character |
  |---|---|---|---|
  | 8 | 87.59% | 72.80% (-16.9%) | 69.63% (-20.5%) |
  | 9 | 83.55% | 69.63% (-16.7%) | 66.64% (-20.2%) |
  | 10 | 79.75% | 66.64% (-16.4%) | 63.82% (-20.0%) |
  | 11 | 76.17% | 63.82% (-16.2%) | 61.15% (-19.7%) |

  With the filter, leaving out a rejected last character is one AND on a
  row's mask. Rows are where most of it is, and skipping them needs a list
  of the rows that survive for each state the leading characters leave
  behind, instead of every row. It changes which candidates a search
  covers, so it needs deciding - and every client has to agree on it for a
  target (it would belong with the prune flags the coordinator sends).

## Tooling

- [ ] **`search_bench`** counts pruned candidates as searched and prunes
  symbol runs but not brackets, unlike the real configuration. That's right
  for "how fast does a search finish", but compare kernels with `noprune`.
- [ ] **CPU and GPU together.** Run the CPU backend on the cores the GPU
  backend leaves idle. *Estimated* a few percent at best - only worth it
  once the CPU backend has the filter.

## Safeguards against missing a match

The filter is proven, tested densely, checked before every search, and every
backend is self-tested before use (see the README). What's still open:

- [ ] **Canaries from the coordinator.** Now and then, the server hands out
  a range with a target planted in it - the hashes of a known candidate in
  that range - indistinguishable from real work, and checks that the client
  reports it. That checks each volunteer's actual hardware, driver and
  build, continuously: bit flips (consumer GPUs have no ECC memory), a
  driver bug, a broken release, a modified client. Nothing on the client
  side can do that. Needs a protocol and server change.
- [ ] **Double-checking.** Re-issue a small random sample of finished ranges
  to a different client, and compare the hashA hits both reported (not only
  "found"). A cheaper, statistical form of the above.
- [ ] **GPU tests in CI.** The release workflow's runners have no GPU, so the
  GPU backends' tests only run when someone runs `ctest` on a GPU machine.
  A self-hosted runner with a GPU would run them on every change.
- [ ] **Longer stress runs.** `search_stress_test --budget <candidates>`
  runs longer than ctest's default of 300 million (about 60,000-73,000 hits
  in the default geometry); worth a long run after any change to a kernel.
- [ ] **Mutation testing as a script.** The experiment in the README (nine
  deliberately broken CUDA kernels, each of which the self-test, the
  integration test and the stress test must catch on their own) was run by
  hand, from copies of the source. A script that applies each mutation to a
  copy, switches off `createBackend`'s self-test there, builds, and checks
  that every test fails would make it repeatable after any kernel change -
  and should fail loudly when a mutation no longer applies. The mutations:
  `mask &= (uint64_t(1) << AlphabetSize) - 1` to `mask &= 0xFFFFFFFFull`;
  `lowBitsFilterIndex(seed1, seed2)` to `(seed2, seed1)` in the kernel;
  `<< lastRowEndK` to `<< (lastRowEndK - 1)`; `<< firstRowStartK` to
  `<< (firstRowStartK + 1)`; the first row's condition `t == 0` to `true`;
  both edge conditions to `false`; `while (mask != 0)` to
  `while ((mask & (mask - 1)) != 0)`; the suffix loop to `i + 1 < SuffixLen`;
  and `targetHashA` to `targetHashA ^ 1` in `buildLowBitsFilterTable`.
  The same for the other backends as they get the filter.
