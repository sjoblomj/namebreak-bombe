# Performance task list

Speedups still open for the client, found while adding the lookup filter to
the CUDA backend (see README.md's [The lookup
filter](README.md#the-lookup-filter-most-candidates-are-never-hashed)), plus
the safeguards that would make a missed match even less likely. Each item
says what's known (**measured**, or **estimated** where it hasn't been
tried) and what it would take.

Numbers are from the RTX 3080 Ti Laptop GPU and i9-12900H this was developed
on. Nothing here counts as done until it's measured with `search_bench
--scale 20` (warmed up, and without `noprune` too, so that pruning doesn't
blur a kernel comparison) and the whole `ctest` passes - the dense-hit
`stress-*` tests above all, since every item below touches code that could
silently drop candidates.

Where things stand: the CUDA backend searches 1,434-1,467 G candidates/s
with a hot GPU, and up to about 1,700 with a cool one (**measured**), from
209-218 before the lookup filter and 1,088-1,124 with the filter but one row
per thread. The OpenCL backend, with the filter and row groups too, searches
about 1,300-1,460, from 202-205 before the filter and 1,082-1,086 with it
alone. The CUDA kernel issues about 100 instructions per row of 49 candidates
(*estimated* from its SASS: about 30 per row, the verify loop's, and a
share of its thread's setup), down from about 1,130 before the filter.
Before the rows were split into chunks, Nsight Systems put about 5% of the
GPU's time between launches (478 ms of kernel time in a 0.5 s search: 2,044
launches of 233 us each); with launches a quarter shorter since, that share
will have grown (not measured again).

## The lookup filter on the other backends

- [ ] **CPU backend.** A prototype (the cpu backend's row walk, with each
  row's last character looked up in the table instead of hashed with SIMD)
  passed the whole test suite and measured **4.6 times as fast**: 19.5 G
  candidates/s against 4.3, on a machine busy enough that both were below
  the README's 9. Tables of 8 or 9 bits per seed were best (512 KB / 2 MB,
  from the CPU's L2); every width from 6 to 10 beat the current backend by
  3-5 times. It's also simpler code than the SIMD it would replace. Later,
  the row states themselves could be computed eight at a time with AVX2,
  and their table entries gathered. The backend already walks its rows
  incrementally (`searchRowsWith`), and so did the prototype: the port must
  keep that, so there's no row decoding to add here.
- [x] **OpenCL.** `filteredRowsKernel` ported to `search.cl`, with the table
  in a global buffer and the same checks as the CUDA backend (a sample of
  the table checked, and read back, before every search). **Measured** 5.3
  times as fast: 1,082-1,086 G candidates/s, from 202-205 - about 72% of the
  CUDA backend, which also has its row groups split into chunks. Its tests
  run about four times faster too (a whole ctest of OpenCL in 45 s, from
  about 3.5 minutes): the driver no longer compiles a fully unrolled loop
  over every candidate of a row. `run_mutation_test_opencl` covers it.
  - [x] **Incremental row decoding**, as the CUDA kernel's: row groups split
    into chunks, one per work-item, walked with a loop the launch size
    can't cut short. **Measured** +35%: tuned separately from CUDA, a whole
    row group per work-item (64 rows, so one chunk whatever the alphabet)
    did best - about 1,460 G candidates/s against 1,435 at CUDA's 25 and 880
    at 1, in the same run; 1,082-1,086 before, with one row per work-item.
- [x] **Metal.** The same port, with one thread per row, and its kernel
  takes `HASHA_MATCH_MASK` too, so the stress tests cover Metal on a Mac.
  There's no Mac here, so it was tested by compiling `search.metal` as C++
  against a stand-in for `<metal_stdlib>` and running it one GPU thread at a
  time (a threadgroup in two passes around its barrier) behind a C++
  transcription of `metal_backend.mm`: every integration and stress test
  configuration and the overflow test pass that way, and all twelve of its
  kernel mutations are caught by each check. `metal_backend.mm` itself has
  never been compiled.
  - [ ] **Run it on a Mac**, or on the release workflow's macOS runner (run
    it by hand from the Actions tab): that compiles `metal_backend.mm` and
    runs `ctest` there, and `tests/mutation_test.py --backend metal` covers
    the host-side mutations the emulation can't.
  - [x] **Incremental row decoding**, as OpenCL's: row groups split into
    chunks, one per thread, walked with a loop the grid's size can't cut
    short. Tested the same way as the filter port: every configuration
    passes, with half as many threads and with a threadgroup too many as
    well, and each of its twenty-one kernel mutations is caught by all
    three checks. **Not measured.** It takes CUDA's 25 rows per thread (two
    chunks per group) rather than OpenCL's whole group, so that a batch - a
    quarter of CUDA's - still has enough threads for the largest Apple GPUs.
  - [ ] **Tune the chunk size on a Mac.** Sweep `NAMEBREAK_ROWS_PER_THREAD`
    (1, 7, 25, 49 and 64, say) with `search_bench --scale 20`, as for CUDA
    and OpenCL.
- [ ] **HIP.** Gets the filter with the CUDA code, but has never run on an
  AMD GPU. The self-test now runs there before every use, which lowers the
  risk - but measuring it on real AMD hardware is still to do.
  - [x] **Tested through HIP's NVIDIA platform**, filter and row groups
    and all: built against ROCm 6.1.2's own HIP headers (compiled by nvcc)
    and run as `--backend hip` on the RTX 3080 Ti, every test passes - the
    integration test in its four geometries, the stress test in its five
    configurations and the overflow test. That tests the HIP API mapping,
    not AMD's compiler, and the NVIDIA platform would even compile a CUDA
    runtime call `gpu_runtime.h` doesn't map, so two more things were
    checked by hand: every `cuda*` name the backend uses is mapped, and the
    device functions it uses beyond CUDA's syntax exist in AMD's HIP
    headers with the same meaning - `__syncthreads`, `__ldg`, which has a
    `const unsigned long*` overload for the table, and `__ffsll`, whose two
    AMD overloads the `(long long)` cast chooses between.
  - [ ] **Run it on an AMD GPU**: `ctest`, `search_bench --scale 20` and
    `tests/mutation_test.py --backend hip`. Until then, the release
    workflow's `linux-hip` job (run it by hand from the Actions tab) at
    least compiles it with ROCm's own compiler for RDNA2-4.
  - [ ] **Tune the chunk size on AMD.** HIP has the CUDA kernel's
    incremental row decoding already, but its 25 rows per thread was tuned
    on an NVIDIA GPU. Many AMD GPUs run 64 threads in lockstep rather than
    32, with different occupancy limits, so the best
    `NAMEBREAK_ROWS_PER_THREAD` may differ: re-sweep it (the README's kernel
    section has the NVIDIA numbers to compare with).

## CUDA backend

- [x] **Incremental row decoding.** Decoding a row - four divisions by the
  alphabet's size, eight shared-memory loads, four hash steps - was about
  50 of the 150 instructions a row cost (SASS of `filteredRowsKernel<49, 4>`).
  Now a thread takes a chunk of about 25 consecutive rows of one row group
  (`NAMEBREAK_ROWS_PER_THREAD`, see the README's kernel section): the
  group's characters are hashed once, and each row costs about 30
  instructions besides its verify loop. **Measured** +32% side by side
  (1,088-1,124 to 1,434-1,467 G candidates/s), +36% in a cooler sweep, where
  1 row per thread did 1,133, 7 did 1,575, 10-16 about 1,605, 25 did 1,659
  and 49 did 1,611. The HIP backend is this same code, so it has the change
  too. The OpenCL and Metal kernels got it after their filter ports (see
  above): without the filter a row there still hashed all its candidates,
  and decoding was only about 5% of that (*estimated* from the unfiltered
  CUDA kernel they were ported from), so it wasn't worth doing before. The
  CPU backend already walked rows incrementally (`searchRowsWith`).
- [x] **Less divergence in the verify loop.** A warp repeats the loop over
  flagged candidates as many times as its busiest lane needs: about 1.4
  times at 8 bits (*estimated* from the statistics). The sweep of
  `NAMEBREAK_LOWBITS_FILTER_BITS` it called for first showed the opposite of
  what was expected: wider tables are slower, narrower ones faster -
  **measured** (`search_bench --scale 20`, interleaved, 3-4 runs each) 6
  bits about 1,770 G candidates/s, 7 about 1,800, 8 about 1,550, 9 about
  1,580, 10 about 390. The table's reads cost more than the divergence, so
  the default is now 7 (a 128 KB table): **+16-18%**, in every pair of runs.
  - [ ] **Warp-cooperative verification** (collect the flagged candidates
    of all 32 lanes with a ballot, then have each lane verify one) is still
    possible, but now works against a table at 7 bits, where twice as many
    candidates are flagged: worth trying only after **Where the table
    lives**, which the sweep shows matters more.
  - [ ] **Re-sweep on the other backends.** 7 applies to all of them (the
    table is shared). OpenCL on this GPU gains too - **measured** about
    1,453 G candidates/s against 1,325 at 8 bits (+9-10%, three pairs) - but
    Metal on a Mac and HIP on AMD may each prefer another width.
- [x] **Where the table lives.** Tried: a 6-bit table (32 KB) copied into
  each block's shared memory, with the grid capped at 116, 174 or 348
  blocks (2, 3 or 6 per SM) so each block loads it once and walks many
  chunks with the kernel's grid-stride loop. **Measured slower**
  (`search_bench --scale 20`, interleaved, three runs each): 1,385, 1,220
  and 1,480 G candidates/s, against 1,880 for the same 6-bit table read
  through the read-only cache and 1,790-1,970 for the shipped 7 bits.
  32 KB of shared memory per block leaves at most 3 blocks (768 threads) per
  SM, too few to hide the hash's latency, and random 64-bit reads from
  shared memory conflict between a warp's lanes - while the read-only
  cache already keeps most of a 32-128 KB table close. Reverted; the table
  stays in global memory, read with `__ldg`. (6 bits through the cache
  measured close to 7 again - within the runs' noise.)
- [ ] **Gaps between launches (~5%, measured).** One launch per leading
  value, synchronized and read back before the next. Either launch several
  leading values at once (a 2D grid over an array of their seeds and
  prefixes), or keep two launches in flight with double-buffered results.
  *Estimated* +5%, and it makes the next item cheap.
- [ ] **Re-tune the window and launch size.** `NAMEBREAK_GPU_WINDOW_CHARS`
  (5) and `NAMEBREAK_ROWS_PER_LAUNCH` were chosen with the kernel as it was
  before the filter, whose launches took about seven times as long as now.
  `NAMEBREAK_ROWS_PER_THREAD` (25) interacts with both: a smaller window
  means fewer rows per launch, so fewer threads.
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
- [x] **Mutation testing as a script.** The experiment in the README
  (deliberately broken CUDA kernels, each of which the self-test, the
  integration test and the stress test must catch on their own) was run by
  hand in three rounds, and each round found a gap in the checks - the best
  argument for running it after every kernel change. It's now
  `tests/mutation_test.py` (`--target run_mutation_test`): it copies the
  source, applies each mutation as an exact string replacement (stopping if
  one no longer matches exactly once), switches off `createBackend`'s
  self-test in the copy, builds, and checks that the self-test,
  `search_integration_test` and `search_stress_test` each fail - after
  checking that the unchanged code passes them, as does the one mutation
  that must *not* be caught (one thread too few launched). `--list` shows the
  mutations.
  - [x] **The other backends**: OpenCL (`--backend opencl`) and Metal
    (`--backend metal`, on a Mac) have their own mutations, and
    `--backend hip` runs CUDA's on the CUDA code compiled as HIP.
