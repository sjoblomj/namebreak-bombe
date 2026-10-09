# Performance task list

Speedups still open for the client, found while adding the lookup filter to
the CUDA backend (see README.md's [The lookup
filter](README.md#the-lookup-filter-most-candidates-are-never-hashed)), plus
the safeguards that would make a missed match even less likely. Each item
says what's known (**measured**, or **estimated** where it hasn't been
tried) and what it would take.

Numbers are from the RTX 3080 Ti Laptop GPU and i9-12900H this was developed
on. Nothing here counts as done until it's measured with `search_bench
--scale 20` (warmed up; compare kernels by its search rate, and anything
that changes the window by its projection for a real search) and the whole
`ctest` passes - the dense-hit
`stress-*` tests above all, since every item below touches code that could
silently drop candidates.

Where things stand (before **Verify a chunk's candidates after its rows**,
which added another 14-35%, see there): the CUDA backend searches
1,434-1,467 G candidates/s with a hot GPU, and up to about 1,700 with a
cool one (**measured**), from
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

- [x] **CPU backend.** A prototype (the cpu backend's row walk, with each
  row's last character looked up in the table instead of hashed with SIMD)
  passed the whole test suite and measured 4.6 times as fast, on a busy
  machine. Now the backend itself: it keeps its incremental row walk, looks
  each row up in the search's table (built and sample-checked in
  `beginSearch`, as the GPU backends do), and hashes only the flagged
  candidates in full - plain C++, the SIMD versions (AVX2, SSE2/NEON, and
  plain loops) gone. It also takes 16 batches per call (`runBatches`), its
  rows cut into work items - about eight per thread - that the threads take
  in turn, so that they start once per 16 leading values instead of once
  per batch, and batches of any size share out evenly (with one call per
  batch, and a leading value split into a batch of 2^22 rows and a smaller
  one, it measured about 26-28 G candidates/s). **Measured**
  (`search_bench --backend cpu`, i9-12900H, 20 threads): about **50 G
  candidates/s, against 10.6** - 4.7 times as fast - at the shared 7-bit
  table (the prototype found 8-9 bits best on the CPU, but every width from
  6 to 10 within a few tens of percent). The mutation script now has a
  `cpu` list too.
  - [ ] **Later**: the row states computed eight at a time with AVX2, and
    their table entries gathered; a table width of its own, if 8 or 9 bits
    still measure better here.
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
  - [x] **Tested through HIP's NVIDIA platform**, filter, row groups and
    16 batches per launch and all: built against ROCm 6.1.2's own HIP
    headers (compiled by nvcc) and run as `--backend hip` on the RTX 3080
    Ti, with `NAMEBREAK_REQUIRE_GPU` on, every test passes - the integration
    test in its four geometries, the stress test in its six configurations
    and the overflow test, whose batches searched together overflowed and
    were searched again one by one, as they should. That tests the HIP API mapping,
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
  - [x] **Warp-cooperative verification** (collect the flagged candidates
    of all 32 lanes with a ballot, then have each lane verify one): tried
    after the profile (see **Profile**), which has the verify loop at about
    half of the kernel's instructions with most of a warp's lanes idle in
    it. **Measured slower**, in both forms, with Nsight Compute on one
    launch of 16 leading values (the kernel as it is: 374M instructions,
    3.62 ms at the profiler's fixed clocks) and `search_bench --scale 40`:
    - A queue per warp in shared memory: each row, every lane appends its
      flagged candidates with a ballot (a round per candidate a lane has -
      2.05 rounds a row, as a row's 32 lanes nearly always include one with
      two), and the warp verifies them 32 at a time. 597M instructions, 5.63
      ms; **39% slower**. A round of appending costs about 45 instructions,
      more than verifying the candidate (about 22), and the queue's 7 KB a
      block took L1 from the table: 22% hits instead of 58%.
    - Just the rows in step: every lane of a warp going through the same
      number of rows, and the verify loop gone round by the whole warp
      (`__any_sync`) or followed by `__syncwarp()`. The rows then ran at
      full width (2.94M table reads a launch instead of 5.61M) - but 500M
      instructions, 4.8 ms: **17% slower**. Keeping a warp together cost
      more (the warp-wide loop, a vote each round, bookkeeping every lane
      executes) than the split warps' idle lanes did: 9.7G thread
      instructions against 5.3G.
    So the kernel's split warps are cheaper than they look. Not tried then:
    a verify loop over two or more rows' candidates together - now done,
    see **Verify a chunk's candidates after its rows**.
  - [ ] **Re-sweep on the other backends.** 7 applies to all of them (the
    table is shared). OpenCL on this GPU gains too - **measured** about
    1,453 G candidates/s against 1,325 at 8 bits (+9-10%, three pairs) - but
    Metal on a Mac and HIP on AMD may each prefer another width.
- [x] **Verify a chunk's candidates after its rows.** Each lane's rows are
  looked up first, noting only which have flagged candidates (a bit per
  row); a second loop then hashes them one candidate per round, a lane that
  runs out of one row's taking its next flagged row (its state and mask
  worked out again). A warp's verify rounds go from the sum over its rows of
  the busiest lane's candidates (about two a row) to the busiest lane's
  candidates in the whole chunk - no ballots, shared memory or warp syncs,
  unlike the warp-cooperative tries above. It pays most where a candidate
  costs most, with a long suffix: one of the coordinator's active targets
  has `\BWUNIN.EXE` (11 characters), which searched a third slower than
  `.WAV`. **Measured** (`search_bench --scale 20 --size 43`, a new
  `--suffix` option, alternating pairs): +17% with `\BWUNIN.EXE`, the same
  with `.WAV`. It also changed what the tuning knobs prefer, so they were
  swept again on top of it (sizes 40, 43, 49; suffixes `.WAV` and
  `\BWUNIN.EXE`):
  - `NAMEBREAK_ROWS_PER_THREAD` **64** (a whole row group per thread) instead
    of 25: 2-6% faster in every configuration (43, also a whole group at
    size 43, about 4%), and 15 about 6-9% slower than 25.
  - **Suffix lengths 9-12 compiled in** (`dispatchSuffixLen`): the runtime
    loop for longer suffixes cost about 10% with a 9- or 11-character one.
    32 more kernel instantiations to compile.
  - **A 6-bit table for CUDA** (`kCudaLowBitsFilterBits`, 32 KB): 13% faster
    than 7 bits with `.WAV`, the same with 9-11 characters (and slower
    with the runtime suffix loop, before lengths 9-12 were compiled in). 8
    bits was 11-22% slower. The OpenCL backend measured 3% (`.WAV`) to 15%
    (`\BWUNIN.EXE`) slower at 6 bits, the CPU backend 17%, so the width
    became a parameter of `buildLowBitsFilterTable`,
    `checkLowBitsFilterTable` and `lowBitsFilterIndex`, defaulting to the
    shared 7 - only CUDA (and HIP) pass their own. A build that sets
    `NAMEBREAK_LOWBITS_FILTER_BITS` (the stress tests' 1-bit filter) sets
    CUDA's too; `lowbits_filter_test` now also runs at 6 bits.
  All together, **measured** against the code before (three alternating
  rounds, every pair faster): 49 characters `.WAV` +14% (about 2,140
  against 1,875 G candidates/s), 43 `.WAV` +18%, 40 `.WAV` with `--whole`
  +22%, 49 `.WAV` with `--whole` +16%, 43 `\BWUNIN.EXE` and `BWUNIN.EXE`
  +35% (about 1,620 against 1,195).
  - [ ] **The same on OpenCL and Metal**, whose kernels still hash each
    row's candidates right after its lookup - and then re-sweep their
    table width and chunk size, as above.
  - [ ] **HIP on AMD** gets all of this with the CUDA code, unmeasured: a
    64-lane wavefront may like other values.
- [x] **Threads per block**, and the other knobs re-checked at the
  coordinator's targets (alphabet sizes 40, 42 and 43, `.WAV` and
  `\BWUNIN.EXE`; `search_bench --scale 60` with their `--alphabet`, four to
  eight interleaved rounds). **Measured**:
  - 512 and 1,024 threads per block (`kThreadsPerBlock`) instead of 256:
    +1-2% and +1-4% at the GPU's own clocks - but at Nsight Compute's fixed
    clocks 1,024 is **6% slower** (2.76 ms instead of 2.60 for the same
    launch, the same 273M instructions, 62% of the warps a scheduler could
    hold instead of 92%). This GPU runs at its 80 W power cap ("SW power
    cap", 1.0-1.1 GHz): a less occupied kernel draws less and gets a higher
    clock, which a GPU with power to spare wouldn't give it. **Unchanged**.
  - The lookup filter: 5 bits 31-38% slower than 6, 7 bits 11-15% slower
    (the same with `\BWUNIN.EXE`). 6 stays.
  - 32 batches per launch instead of 16: within the noise. Between two full
    launches the GPU idles 1.5-1.7% (Nsight Systems, 14 us median gaps,
    launches of 0.8-1 ms at sizes 40-42), so there's little left to win.
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
- [x] **Gaps between launches (~9%, measured).** One launch per leading
  value, synchronized and read back before the next. Measured again after
  the filter, the row groups and the 7-bit table (Nsight Systems,
  `search_bench --scale 5`): a launch - one leading value, 49^4 rows -
  takes 0.13 ms, and the GPU then idles 11.6 us (median) before the next,
  9.7% of the time. Almost all of it is the round trip: the kernel ends,
  the host wakes, reads the 8-byte result and launches the next; the
  host's own work in between is under a microsecond.
  - [x] **Queue the result's copy behind the kernel**, into pinned memory,
    instead of a blocking `cudaMemcpy` after the wait: **measured** 1.6 us
    off the gap (10.0 us, 9.1% idle).
  - [x] **Fewer, longer launches**, from a wider window: **measured** a
    wash - the gaps it recovers are paid back in pruning (see the next
    item).
  - [x] **Several leading values per launch.** The engine collects up to
    `SearchBackend::maxBatchesPerCall()` consecutive batches - of one
    leading value or several, with pruned ones left out - and hands them to
    `runBatches()` together; the CUDA/HIP backend searches up to
    `NAMEBREAK_BATCHES_PER_LAUNCH` (16) in one launch, a row of thread
    blocks (`blockIdx.y`) per batch, whose rows, edges and seeds are a
    by-value kernel argument (indexed loads from the constant bank - no
    local-memory copy, and 32 registers instead of 37). A hit records its
    batch, and is rebuilt with that batch's prefix.
    If a launch's hits together overflow, each batch is searched again on
    its own. **Measured**: in the timed search the GPU idles 1.6% of the
    time instead of 9.4% (Nsight Systems; the kernels themselves ran 2%
    faster, fewer launches' tails), and `search_bench --scale 40` was
    faster in six alternating pairs out of six, +5.0 to +11.6%, about
    **+6.7%** on average. 4, 8, 16 and 32 batches per launch were within
    the noise of each other; 16 makes a launch about 2.3 ms. Other
    backends keep one batch per call (the default), for now.
  - [x] **The same for OpenCL**: a `runBatches()` with the batches in a
    small constant buffer, written without waiting before each launch, a
    row of work-groups per batch (`get_group_id(1)`), and hits read back
    with the count and checked on the CPU, as CUDA's (the next item). Its
    mutation list got the same new mutations. **Measured**, together with
    the sleeping wait below: faster in six alternating pairs out of six,
    about **+2.1%** (+1.9 to +2.5%, leaving out a first pair inflated by a
    cold start), on a GPU power-capped by the heat of a long session.
  - [ ] **The same for Metal**: a batch array in a buffer, the grid's y
    as the batch, the hits read back with the count - and measured on a
    Mac.
  - [x] **A cheaper hit path.** A launch of 16 leading values (4.5G
    candidates) has a hashA hit about two times in three, and each cost
    about 35 us more of GPU idle: the batches' prefixes uploaded, a second
    kernel (`verifyMatchesKernel`) to rebuild and check the hits, a
    synchronize, and three blocking copies. Queuing all that would still
    have left the verify kernel and its wait, so instead the kernel records
    each hit's trailing index and batch in the results buffer, right after
    the count, and the copy that follows every launch brings back the count
    and the first 16 hits; the CPU checks them with `HitVerifier`, as every
    other backend already did (from scratch, on another processor - as
    independent a cross-check as the verify kernel was). A launch with more
    than 16 hits (only the stress test's) costs one more copy.
    **Measured**: the median gap between launches went from 47 to 16 us
    and the GPU's idle time from 1.5% to 0.7% (Nsight Systems), and
    `search_bench --scale 40` was faster in six alternating pairs out of
    six, about **+1.5%** (+1.1 to +2.0%, leaving out a first pair inflated
    by a cold start).
  - [ ] **Keep the next launch queued** while this one's results are read:
    a launch/collect pair beside `runBatch`, with two sets of result
    buffers, so the GPU never waits. It reshapes the engine's search loop
    (the split on too many hits, pause, abort, a match found with a batch
    in flight) and needs its own tests and mutations - with 0.7% idle
    left, worth little now.
- [x] **Re-tune the window and launch size.** `NAMEBREAK_GPU_WINDOW_CHARS`
  (5) and `NAMEBREAK_ROWS_PER_LAUNCH` were chosen with the kernel as it was
  before the filter, whose launches took about seven times as long as now.
  Measured with the fixed `search_bench` (`--scale 40`): a 6-character
  window, with 2^23, 2^25, 2^27 or 2^29 rows per launch (the last is one
  launch per leading value, about 7 ms). Longer launches do recover the
  gaps - window 6 with 2^29 rows searched **+4.5 to +5.7% faster** than
  window 5 in each of six alternating pairs - but a window one wider leaves
  one leading character fewer for the CPU to prune: 16.45% of 4-character
  leading values are pruned, against 20.25% of 5-character ones (and
  12.41% against 16.45% at the real search's length of 9), so the GPU has
  4.8% more to search. The projected rate for a real search came out even:
  -0.3% to +0.9% per pair, about +0.3% on average. **Unchanged**: window 5,
  2^23 rows per launch (which a window-5 launch, 49^4 rows, never reaches).
  The gaps are still worth closing, but only in a way that keeps the
  pruning - several leading values per launch, or the next launch queued
  (see "Gaps between launches").
- [x] **Stop spinning a CPU core.** `cudaDeviceSynchronize` busy-waits: the
  running client keeps one core at 100% (**measured**: `search_bench` now
  reports the CPU time of its timed search). On a laptop, CPU and GPU share
  one power and cooling budget, so that core may cost GPU clock.
  `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` frees it - 5-6% of
  a core - but **measured 3.6% slower** (six alternating pairs, all
  slower): the thread wakes about 460 us after a launch ends (Nsight
  Systems: median gap 16 us before, 461 after), though the kernels
  themselves ran about 10% faster, the GPU getting the power the CPU no
  longer used. So the CUDA backend now sleeps through most of each launch
  and spins only for the end of it (`waitForLaunch`): it predicts a launch's
  time from the time a candidate took in the search's recent large
  launches, timed on the host, and wakes a margin early - twice its recent
  sleep overshoot, so that where sleeps are coarse (Windows' default timer)
  it never sleeps. **Measured**: 24% of a core instead of 100%, and the
  same speed (six alternating pairs, -1.0 to +1.6%, about +0.2% on
  average): its gaps are longer (median 32 us), as the CPU wakes up slower,
  but its kernels shorter. It can only change when the host looks, never
  what's found - it still waits with `cudaDeviceSynchronize`.
  - [x] **The same for OpenCL**: NVIDIA's OpenCL spins too - its blocking
    read kept a core at 100% (**measured**). Its launches were one leading
    value, about 0.16 ms - too short to sleep in, with a wake-up margin of
    about 0.2 ms - so it got 16 leading values per launch first (above),
    and then the same `LaunchWaiter`, now shared (after a `clFlush`, so the
    launch is on its way before the host sleeps). **Measured**: 21% of a
    core instead of 100%.
  - [ ] **Short launches never sleep.** A launch of 16 batches is 16 x
    size^5 candidates - about 0.8 ms at size 40 and 1 ms at 42, against 2.3
    at 49 - and the waiter starts out assuming a sleep overshoots by 0.5 ms,
    a margin of 1 ms, so it never sleeps through one, and never learns that
    sleeps here overshoot by far less: those searches keep a core at 100%
    (**measured**: `sc-ptbr-poor-fellars-dog`'s and `rez-wav1`'s settings;
    `\BWUNIN.EXE`'s longer launches sleep, at about 30%). Starting from
    0.1 ms took it to 33-35% - and the same speed (six alternating rounds
    each, -0.3 and -0.1%; a laptop whose GPU sat at its own 80 W cap
    throughout). Worth doing for the core it frees, not for speed; on
    Windows' default 15.6 ms timer its first sleep would overshoot once,
    and it would stop sleeping, as now.
  - [ ] **Metal**: `waitUntilCompleted` is documented as a blocking wait, so
    it may not spin at all - measure on a Mac before changing anything; its
    launches, one batch of 2^21 rows, are short too.
- [x] **Build the table faster, or less often.** At 8 bits, building it
  took 4 ms for `.WAV` and 10 ms for a 17-byte suffix, and the check before
  every search 0.4-1.6 ms - nothing next to a real range, but most of the
  cost of the test suite's thousands of tiny searches. At today's 7 bits
  (a quarter of the entries) it was 0.78 and 1.9 ms, the check 0.31 and
  1.1 ms (**measured**, best of 20) - still 45% of the integration test's
  time on its own (2,296 searches), 69% of OpenCL's.
  - [x] **Faster**: the hash step is a T-function, so a table of n-bit
    states can be computed in the narrowest integer that holds n bits,
    modulo its size, and its low bits come out exactly as the 32-bit hash's:
    8-bit lanes up to 8 bits, 16-bit ones up to 10 - four or two times as
    many per vector instruction - with each block's masks gathered locally
    and written once. **Measured**: 0.42 and 0.86 ms (1.9-2.2 times as
    fast), and the integration test on CUDA 2.96 s instead of 4.14. The
    check is unchanged - 32-bit, from random high bits - and
    `lowbits_filter_test` now also runs at 1, 8 and 10 bits, to cover both
    lane widths at their narrowest and widest (a builder shifting by 4
    instead of 5, using 8-bit lanes at 10 bits, or truncating a key, fails
    it).
  - Less often: **measured** not worth it. The integration test's 2,296
    searches have 2,179 different alphabet/suffix/target combinations, the
    stress test's 379 have 375, so a cache would save about 5% of the tests'
    builds; in coordinator mode, where range after range shares a target,
    it would save 0.4 ms per range.
  - Building it on the GPU would take microseconds, but it would be a
    second builder per backend - three, with Metal's - for what is now half
    a millisecond.
- [x] **Any alphabet size.** The CUDA kernel had the alphabet's size
  compiled in, one instantiation per size of a fixed list (29, 30, 40-43,
  47-50); now it takes any size from 1 to 63 (`MAX_ALPHABET_SIZE`, raised from 50) at runtime (the other backends
  already did). What it divides by the size - a row group's characters, a
  chunk's rows - it divides with multipliers worked out on the host
  (`FastDivisor`, exact for every 32-bit numerator); a batch's first and last
  row are split into group and row on the host; and a hit records its row and
  last character, the host working out its index - a multiplication the
  kernel had done for every row with a flagged candidate. The default
  alphabet's size, 49, and 42 stayed compiled in (`CompiledAlphabetSizes`): 60
  instantiations instead of 200, and the CUDA file compiles in about 40% of
  the time (4.1 s instead of 10.7). **Measured** (Nsight Compute, one launch of 16 leading values;
  `search_bench --scale 40`, five pairs each, alternating with the code before;
  `search_bench --size <n>` times other sizes):
  - at 49, compiled in: 336M instructions instead of 374M, 3.37 ms instead of
    3.62 (323M instead of 333M with `--whole`); **3.2% faster** (all five
    pairs), **2.1%** with `--whole` (all five).
  - at 50 and 42, taken at runtime: the same speed (0.6% and 0.2%
    slower, within the noise), and **2.2-2.3% slower** with `--whole` (all
    pairs) - a few instructions a row the compiler saves when it knows the
    size. So 42 is compiled in too: **4% faster** than before (five pairs,
    all faster; on a GPU hot enough to run a third slower than usual) and
    **2%** with `--whole`. Another size is a number in
    `CompiledAlphabetSizes`, and 20 instantiations more.
  - Then the compiled-in sizes became 42 and 43, and 49 is taken at runtime.
    **Measured** against the code before any of this (four pairs each): 49
    the same speed (0.5% slower, within the noise) and **2.1% slower** with
    `--whole` (all four pairs); 43 **4.5% faster** (all four) and **1.3%**
    with `--whole`.
  - Then 40 compiled in too, the size of the coordinator's
    `sc-ptbr-poor-fellars-dog`: with a whole row group per thread, a known
    size of 40 is also one chunk per group known. **Measured** (Nsight
    Compute, the same launch, walking every group): 149M instructions
    instead of 156M, 1.50 ms instead of 1.53; `search_bench` with that
    target's alphabet **+4.1%** without `--whole` (eight rounds, all faster,
    +3.5 to +6.7%), and about +1.3% with it, walking the lists. 140 kernel
    instantiations instead of 112; the CUDA file compiles in 13.3 s instead
    of 11.8 here, and the self-test's five cases at 40 (and its pruning and
    grouped cases) take it from 143 to 157 ms.
  - The first runtime version issued 25% more instructions: with the size
    unknown, the compiler worked the batch's first and last row out again
    for every row (reading the batch from the kernel's arguments by
    `blockIdx.y`, and dividing again) rather than keep them in registers -
    and `keepInRegister`, an empty `asm` that makes a value opaque, is what
    stopped it. The row's hash step written out with its invariant parts
    kept the same way saved the list-walking kernel another 2-3%; the same
    step as a select rather than an `if` lost that and more, and folding
    the alphabet's mask into the first row's made no difference (the
    compiler had done it already).
- [x] **Profile.** Nsight Compute 2025.2 (`--set full`) on single
  launches of 16 leading values of `filteredRowsKernel<49, 4>`, walking every
  row group and walking lists (`search_bench --scale 2`, with and without
  `--whole`; ncu locks the clocks at the base 585 MHz, so only ratios
  count). Reading the counters needs root unless
  `options nvidia NVreg_RestrictProfilingToAdminUsers=0` is in
  `/etc/modprobe.d/` (it is on the development machine now). **Measured**:
  - **Bound by instruction issue**, not memory: the SMs are busy 80% of the
    time, at 3.05 instructions a cycle of 4, and the top stall is "not
    selected" (3.1 warps waiting per instruction issued - there's always
    another warp ready), then math pipe throttle and the table lookup's
    latency (the AND after the `LDG` of the table gets 21% of the stall
    samples, but other warps hide it). Occupancy 93%, 30 registers, L2 59%
    busy and all hits, L1 58% hits, DRAM idle. So what counts is how many
    instructions are issued - and **Where the table lives** can't gain much.
  - **Most lanes are idle**: 374M instructions a launch, 5.27G thread
    instructions - 14 of a warp's 32 threads active on average. The rows
    run at half width: 5.61M table reads a launch, 47.7 per warp for the
    24 or 25 rows of each lane, so the lanes with a group's first chunk and
    those with its second go through their rows separately; at 29
    instructions a row that's 44% of the instructions, and the verify loop
    most of the rest. (The source view's per-instruction counts, which come
    from instrumenting the code, add up to 538M for this kernel - more than
    it could issue in the time it took - so only the hardware counters are
    quoted here.) Filling the lanes measured slower both ways it was tried:
    see **Warp-cooperative verification**.
  - **The list-walking kernel** (`Listed`) issued 35 instructions per row
    where the other issues 29: the 64-bit test of a row's bit in the row
    mask, most of its 5%. Now a chunk's row bits are shifted into a mask of
    their own once (32 bits on CUDA, whose chunks have at most 25 rows) and
    it moves on one bit per row. **Measured**: 333M instructions a launch
    instead of 342M, 3.26 ms instead of 3.37; `search_bench --whole` 1.3%
    faster on CUDA (five pairs, all faster) and 1.2% on OpenCL (four
    pairs, all faster), and the same without `--whole`. Metal has it too.
  - Two cheaper rows and verify rounds, **measured**: the first and last
    row's cut applied only in a batch's first and last group (a branch per
    chunk rather than 8 instructions per row): 378M instructions, 3.68 ms -
    no gain. The highest flagged bit instead of the lowest (one bit scan of
    whichever half isn't zero, and an XOR to clear it): 372M, 3.53 ms, but
    in `search_bench` 0.4% faster and 1.6% with `--whole` - within the
    noise, so left out.
- [ ] **Micro-optimizations in the verify path.** The last suffix step's XOR
  and `+ 3` could be folded into the constant it's compared with. They were
  worth about a quarter of the old kernel's instructions, but they're only
  paid on the ~0.2 candidates per row that get through the filter now.

## The search space (a decision, not only a speedup)

- [x] **Prune the whole candidate.** `prune_symbol_runs` and
  `prune_unopened_brackets` only looked at the leading characters. Applied to
  the whole candidate, they'd skip about a fifth more of it at every length
  (**computed** exactly for the real alphabet, both rules on):

  | Length | Searched now | Rules on every character but the last | On every character |
  |---|---|---|---|
  | 8 | 87.59% | 72.80% (-16.9%) | 69.63% (-20.5%) |
  | 9 | 83.55% | 69.63% (-16.7%) | 66.64% (-20.2%) |
  | 10 | 79.75% | 66.64% (-16.4%) | 63.82% (-20.0%) |
  | 11 | 76.17% | 63.82% (-16.2%) | 61.15% (-19.7%) |

  Now a setting, `prune_whole_candidate` (a `[search]` key, and a target's
  setting on the coordinator, sent with every claim; off by default): the
  rules, `max_backslash_count` too, at every character **but the last** -
  the middle column. Leaving out a last character saves nothing (its row is
  hashed anyway), and the host can't drop the hits it would leave out
  without dropping a real match too. For each state the leading characters
  leave (32 occur in a real search), the host lists the row groups that
  survive, with a bit per class of the rows' own last characters its
  characters leave (`backends/common/row_pruning.h`); the GPU kernels
  walk their batch's list, and a pruned row inside a surviving group gets
  no candidates - skipping it wouldn't save its lane anything, as the warp
  steps through its rows together. See the README's "Design decisions".
  **Measured** (`search_bench --scale 40`, alternating with the code before
  it): a real search projected **+10.8%** on CUDA (2,639 against 2,382 G
  candidates/s, five runs each) and **+9.8%** on OpenCL (2,265 against
  2,063). Tried on the way:
  - One kernel for both, walking a list of every group when nothing is
    pruned: **5% slower** than before on CUDA (six pairs; 6.5% on OpenCL),
    whatever was tried - reading each chunk's entry before the barrier,
    no per-row check at all (slower still, about 20%: the compiler
    schedules the loop differently), the group computed as before instead
    of read (also about 20% slower). So a search that doesn't prune the
    whole candidate still gets the kernel without lists (`Listed` in the
    CUDA kernel, `LISTED` in the OpenCL and Metal ones), and measures the
    same as before (0.4% and 0.7% slower, within the noise).
  - Iterating only a chunk's surviving rows (their bits): 13% slower with
    nothing pruned. A branch around each pruned row: 4.5% slower with
    nothing pruned. Two loops, one without a check for chunks with no
    pruned rows: as fast with nothing pruned, but with the whole candidate
    pruned slower than a single loop, as a warp with both kinds of chunk
    runs both.
  - [ ] **Get the rest.** 20% fewer candidates would be about +20%
    (2,870 projected). The list-walking kernel's 5% is part of it - Nsight
    Compute should say what it spends that on - and the pruned rows that
    still take their turn the rest: of the rows asked for, 12.7% are left
    out with their whole group, 3.8% only by their own last character
    (*estimated* from 3 million random 10-character candidates).
  - [ ] **Metal** has it too, walking its batch's list with `LISTED`
    (compiled here only as C++ against a stand-in for `<metal_stdlib>`, for
    its syntax): run the tests on a Mac.
  - [x] **Walk the lists only where they pay** (CUDA). How much the rules
    prune depends on the alphabet's characters: the coordinator's
    `sc-ptbr-poor-fellars-dog` searches `" -.0-9A-Z_"` with symbol runs
    pruned, where a run of three needs three of `-`, `.` and `_` in a row -
    0.3% of its candidates - so the lists left out almost nothing, and cost
    the list-walking kernel's 5% (**measured**, Nsight Compute, the same
    launch: 165M instructions and 1.60 ms walking the lists, 156M and 1.53 ms
    walking every group). Now a launch walks its batches' lists only if
    they leave out at least `NAMEBREAK_LIST_MIN_PRUNED_PERCENT` (5) of its
    row groups; if not, it walks every group, and the hits in rows the lists
    leave out are dropped on the host (`RowPruning::survives` - a launch has
    about one hit), so a search reports exactly what it did. The tests run
    both ways: small_launch always walks the lists, window2 and the stress
    test's overflow variant never do, the rest take the default. So does the
    self-test, whose ranges of a few row groups would otherwise never walk
    the lists where only a row is pruned (the mutation script's `rowbit` got
    past it): it asks for each way (`SearchConstants::listWalking`, which
    the engine leaves to the backend), and takes 207 ms instead of 157. The
    mutation script has four mutations of the host's check, and two that
    force either way and must change nothing. **Measured** (`search_bench
    --scale 60`, six rounds interleaved with the code before, with the
    `--alphabet`, `--prefix` and `--suffix` of the coordinator's targets;
    the backend's search rate), against the code before, which always
    walked the lists:

    | Search, with `--whole` | Pruned by the backend | At 5% (shipped) | Never walking them |
    |---|---|---|---|
    | `" -.0-9A-Z_"`, `MUSIC\`, `.WAV`, symbol runs | 0.14% | **+4.8%** | +4.4% |
    | `" ()-.0-9A-Z_"`, `REZ\`, `.WAV`, all rules | 4.9% | +1.1% | +0.9% |
    | `" ()-.0-9A-Z\_"`, `\BWUNIN.EXE`, all rules | 5.1% | -0.2% | -4.0% |
    | the real 49 characters, all rules | 17.4% | +0.1% | -13.1% |

    The 11-character suffix makes the rows pruned inside surviving groups
    worth more (a flagged candidate costs more to hash), so walking the
    lists still pays at 5% there - the threshold counts groups only. With
    size 40 compiled in too (above), which also counts for more once its
    launches walk every group, `sc-ptbr-poor-fellars-dog`'s search is
    **8.6-9.0% faster** than before both (two runs of four and six
    interleaved rounds, every round faster); the coordinator's other
    targets, which don't prune the whole candidate, the same.
  - [x] **The same for OpenCL** (`kListMinPrunedPercent`, the same 5%;
    the tests' variants and the self-test ask it for each way as they do
    CUDA, and its mutation list got the same six mutations). Its kernel is
    compiled for the alphabet's size at runtime anyway, so there was no size
    to compile in. **Measured** the same way, against the code before:

    | Search, with `--whole` | Pruned by the backend | At 5% (shipped) | Never walking them |
    |---|---|---|---|
    | `" -.0-9A-Z_"`, `MUSIC\`, `.WAV`, symbol runs | 0.14% | **+4.3%** | +4.2% |
    | `" ()-.0-9A-Z_"`, `REZ\`, `.WAV`, all rules | 4.9% | +0.5% | -3.6% |
    | `" ()-.0-9A-Z\_"`, `\BWUNIN.EXE`, all rules | 5.1% | +0.6% | -4.8% |
    | the real 49 characters, all rules | 17.4% | +0.2% | -12.3% |

    Walking the lists pays at 5% here even with `.WAV`, so the threshold
    shouldn't be any higher. OpenCL is what the coordinator's busiest
    volunteer runs - whose kernel still hashes each row's candidates right
    after its lookup (see **Verify a chunk's candidates after its rows**),
    worth more than this.
  - [ ] **The same for Metal**, whose list-walking kernel would cost about
    as much: the host's check is shared (`RowPruning::survives`).

## Dictionary searches

A dictionary search (README.md's [Dictionary mode](README.md#dictionary-mode))
hands the backend batches - a leading part's hash states, a run of the word
list's words and the suffix - rather than rows. The GPU kernels
(`dictionaryKernel` in `cuda_backend.cu`, `dictionary.cl`,
`dictionary.metal`) hash each word on from its batch's state - hashA, or with
a basename key the basename hash alone (see **One hash a candidate**) - and
the suffix where a filter lets it through. There's no row trick to be had:
every candidate is a word of its own, about 8 characters. Until **One hash a
candidate**, a search with a key hashed both. Measured on the RTX 3080 Ti Laptop with
`english-1` and the three StarCraft lists (66,276 words), two words a
candidate, four separators, `MUSIC\BG` and `.WAV`, a basename key: 17.6
billion candidates. Kernel times are Nsight Compute's for one launch of 2^28
candidates at its fixed clocks, since the search's own rate followed how hot
the GPU was (between 7.7 and 20 G candidates/s for the same build, in one
afternoon).

- [x] **A first kernel**: a thread per word, a block per segment of a
  batch's words, a word's characters read a byte at a time and looked up
  in the crypt table's two parts in shared memory. 30.66 ms a launch,
  11.7 G candidates/s. The load/store unit was the limit (97%), and a warp's
  lanes, with words of different lengths, were only 22 of 32 busy.
- [x] **Words by length, four characters to a load.** The word table
  (`DictionaryWordTable`) keeps the words in the order of their lengths -
  which a batch of every word, nearly every batch, is searched in; a part
  of the list is searched in the list's order - so a warp's lanes go round
  together (31.4 of 32 busy), and four characters to a uint32, the two
  hashes' keys side by side in shared memory: one 64-bit lookup a
  character, a quarter of a load. 19.97 ms (-35%).
- [x] **The suffix filter.** As the row search's lookup filter: the low
  bits of the state after the suffix depend only on those before it, so a
  table of 2^(2 * 7) bits per hash says whether the suffix can make a
  match, and it's hashed in full for 1 candidate in 128 rather than all
  (the warp still does whenever a lane needs it). With the suffix's keys
  in constant memory and `+ 3` added in once per character: 18.58 ms. At
  4, 5, 6, 7 and 8 bits: 17.60, 14.32, 14.15, 13.88 and 18.18 ms with the
  final kernel - 8 bits' 16 KB of shared memory costs more than it saves.
- [x] **Words per thread.** Every block finds its batch (a binary search,
  by one thread, the others waiting at the barrier - 30% of their stalls)
  and copies its tables. 1, 2, 4, 8, 16 and 32 words per thread: 27.77,
  21.46, 18.58, 17.24, 16.54 and 14.14 ms; 32 shipped.
- [x] **Whole four-character chunks, `__byte_perm`.** A word's full
  chunks without a check per character, then the rest; a character out of
  its chunk with one `PRMT`: 14.54 ms from 16.54 (-12%). The loop is now
  about 10.5 instructions a character for both hashes.
- [ ] **Tried, no better:** queueing the candidates the filter lets
  through in shared memory and hashing their suffixes together at the end
  of the block (14.26 ms against 14.14 - the queue costs what the warps
  saved); the usual step, letting the compiler add `+ 3` per hash (15.40
  against 14.54); walking a word's chunks by pointer (the same).
- [x] **Where it stands:** 13.88 ms a launch, about 190 instructions a
  candidate, issue slots 88% busy - about 20 G candidates/s on the CUDA
  backend and 19.3 on OpenCL's with this list (same GPU, same afternoon),
  against the first kernel's 11.7. `run_dictionary_bench` (up to two
  words of `english-1` alone, 16.3 billion candidates, a basename key):
  17.1 and 16.3 G candidates/s on CUDA and OpenCL with the GPU at 60-64 C,
  10.9 and 10.4 at 85 C - and 0.20 on the CPU backend, 83 s, for which
  the GPU takes under a second.
- [x] **One hash a candidate.** With a basename key, every word was hashed
  twice, hashA and the basename hash - though the file's name has the
  key's basename, so a candidate whose basename doesn't match it can't be
  the file, and its hashA was hashed for nothing. Now the kernels hash the
  basename alone, compared to the key, and the host checks hashA and hashB
  of the few that match (`DictionaryHitVerifier`); a word with a `\` hashes
  only what follows its last one. Without a key, they hash hashA alone, as
  before. The word loop went from 12.4 to 6.8 instructions a character
  (SASS of `dictionaryKernel<4, ...>`: 99 for eight characters, against 109
  for sixteen). **Measured** with `run_dictionary_bench` (`english-1`, up to
  two words, a key), alternating with the code before:
  - Nsight Compute, one launch of 2^28 at its fixed clocks: 12.54-12.61 ms
    against 13.88 (-9%), with 1.21 G warp instructions against 1.64 (-26%).
    Without a key, unchanged: 12.59-12.61 ms against 12.58-12.60. Fewer
    instructions saved less time because the issue slots went from 89% busy
    to 73%: the warps now wait on the word table's loads (long scoreboard,
    L1TEX), which the second hash's arithmetic used to hide.
  - At the GPU's own clocks: CUDA 25.6-27.5 G candidates/s against
    19.7-21.0, +30 to +33% in six pairs of six (about +32%); OpenCL
    25.5-25.8 against 18.6-19.3, +34 to +37% in four of four. Without a
    key, both within 2% of before. This GPU runs at its 80 W power cap,
    and a kernel that does less for each candidate draws less and is
    clocked higher (about 760 MHz against 630 in the timed search, worked
    out from the instructions and Nsight Systems' kernel times) - a GPU
    with power to spare would get nearer the fixed clocks' 10%.
  - The CPU backend (`--words 20000`, on a machine busy with other work):
    0.59-0.63 G candidates/s against 0.34-0.37 with a key (+60 to +70%) -
    it hashed the basename apart from hashA, every character a second
    time - and 0.69-0.72 against 0.67-0.69 without one. Which hash, decided
    for each word, cost 3% without a key; it's decided once for a run of
    words.
  - Metal: tested emulated (its kernel compiled as C++, threadgroups as
    threads), with `dictionary_search_test` in its three builds and the
    key-only path's mutations; not measured.
  - [x] **The key used whenever it's known**, whether or not the basenames
    are recorded (`record_basenames`, a target's `send_basenames`) - and
    `record_hasha_matches` (a `[dictionary]` setting) to hash both again,
    for a search that wants every Hash-A match or doesn't trust its key.
    The kernels take which hashes as a template parameter (a macro in
    OpenCL and Metal). **Measured** at Nsight Compute's fixed clocks: both
    13.88-13.89 ms a launch, as the kernel before (1.644 G warp
    instructions, the same); the basename alone 12.57-12.61 and hashA alone
    12.58-12.62, as before it. At the GPU's own clocks (72-75 C): 21.9 G
    candidates/s with both, 28.2-28.5 with the basename alone, 28.7-29.2
    with hashA alone.
- [x] **Tails** (`tails`, README.md's "Tails"): a word followed by each of
  a list of tails - `digits:1-2, letters:0-1` makes 2,970, `0` to `9Z`.
  A word is hashed once for a run of its tails rather than once for each:
  the GPU kernels take *cells*, a word and a chunk of up to 32 of its tails
  (`kDictionaryTailsPerCell`), a thread hashing its cell's word and then
  each tail, one to eight characters, on from there; a warp's lanes take
  neighbouring words with the same chunk, so they load the same tail and
  go round its loop together. The kernels take tails or not as a template
  parameter (a macro in OpenCL and Metal): without, there's no tail loop,
  and the kernel is as it was but for a cell count (the word count times
  one) and a tail index in each hit. **Measured** on the CPU backend, which
  hashes a word once for all its tails (a key, the machine busy with other
  work): 0.82-0.94 G candidates/s with 300 words of `english-1` and those
  2,970 tails, against 0.40-0.44 with 20,000 words and none. Not measured
  on the GPU yet - it was busy with a search.
- [ ] **Hide the word table's loads.** Since **One hash a candidate**, both
  kernels wait on a word's entry and characters more than they compute
  (73% of issue slots busy at fixed clocks): loading the next word's while
  hashing this one's might win back some of the 10% the fixed clocks
  didn't show.
- [ ] **Overlap the host with the GPU.** The engine builds a call's batches
  (about 4,000 leading parts for a launch of 2^28) while the GPU waits,
  and the GPU then runs while the engine waits. **Measured** with a
  backend that hashes nothing: 85 ns a batch, 22 ms for the 255,561
  batches of two words of `english-1` - under 3% of the GPU's time, and
  the same share with three words, as it's per batch of every word. Would
  take building the next call's batches during the launch, in the engine;
  not worth it yet.
- [ ] **Persistent blocks.** A block per segment pays for its batch's
  binary search and its tables every 8,192 words; blocks that each search
  many segments wouldn't (**estimated**: a few percent, from the
  words-per-thread sweep's trend).
- [ ] **Tune OpenCL and Metal** on their own: both have CUDA's values
  (32 words per thread, 256 threads to a work-group); Metal's kernel has
  only run emulated on the CPU.

## Tooling

- [x] **`search_bench`** counted pruned candidates as searched and pruned
  symbol runs but not brackets, unlike the real configuration. It now
  prunes as the real configuration does (`--prune symbols` gives the old
  behaviour, which every figure here from before 2026-09-30 used), counts
  what it hands the backend, and reports three rates: the range covered,
  the search rate (what the backend searched, per second), and a projection
  for a real search. Its timed range only varies the last two leading
  characters, where symbol runs can't occur, so it prunes 6.3% where a real
  search of that length prunes 20.3% - the projection uses the latter,
  estimated from 4 million random leading values put through the engine's
  own checks (exactly: 20.25% of all 5-character leading values, 16.45% of
  4-character ones - the estimate is within 0.03 points). With nothing
  pruned it checks that the backend was asked for every candidate of the
  range exactly once.
- [ ] **CPU and GPU together.** Run the CPU backend on the cores the GPU
  backend leaves idle. *Estimated* a few percent at best - only worth it
  once the CPU backend has the filter.

## Safeguards against missing a match

The filter is proven, tested densely, checked before every search, and every
backend is self-tested before use (see the README). What's still open:

- [x] **Canaries from the coordinator.** Now and then, the server hands out
  a range with a target planted in it - the hashes of a known candidate in
  that range - indistinguishable from real work, and checks that the client
  reports it. That checks each volunteer's actual hardware, driver and
  build, continuously: bit flips (consumer GPUs have no ECC memory), a
  driver bug, a broken release, a modified client. Nothing on the client
  side can do that. *Done* - no protocol change was needed: 5% of claims, a
  few seconds' worth each, shown per volunteer on the dashboard (see the
  coordinator README's "Canaries").
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
  - [x] **Dictionary searches**: two more checks, `dictionary_search_test`
    built with few of hashA's bits compared ("dictionary") and with small
    GPU launches ("dictionary-small"), and the self-test runner runs the
    dictionary self-test too. CUDA has 18 dictionary mutations and two
    that must be harmless (the word table longest first, an empty segment
    more), OpenCL 8 of its own; both share those of
    `common/dictionary_batch.cpp`. Each caught by the checks it names -
    CUDA's in 50 minutes. Running it also found that the copies lacked
    `data/` (`english-1`), which the build has needed since dictionary mode.
