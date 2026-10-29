# Pruning and inserted text

## `prune_symbol_runs`, `prune_unopened_brackets` and `max_backslash_count` only run on the CPU

All three are cheap heuristics for skipping implausible candidates before
spending a hash chain on them - not correctness rules (a real match could in
principle violate any of them; these just trade a small amount of
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

The upshot: `prune_symbol_runs`/`prune_unopened_brackets`/`max_backslash_count`
only ever examine a candidate's leading characters (everything except the
trailing window). For `prune_unopened_brackets` that is still exact as far as
it goes: once the leading characters have closed a bracket nobody opened,
nothing in the trailing window can change that - it just never looks at a
stray closer that only appears in the trailing window. The same goes for
`prune_adjacent_backslashes`, added later, and `min_backslash_count` asks
of the leading characters only whether the trailing window could still
hold the backslashes they lack.

Those measurements were of the kernel as it was before the lookup filter,
when a check had to be paid on every candidate - and a check in the kernel
still saves nothing unless a whole warp can skip the work together.

## `prune_whole_candidate`: the rules on every character but the last

With `prune_whole_candidate` (a `[search]` key, or a target setting sent
with every claim - see the coordinator's
[How it works](../../coordinator/docs/how-it-works.md)), the three rules apply to
every character of a candidate but its last. The leading characters are
checked on the CPU as before; the rest are left to the backend, which
doesn't check them candidate by candidate but leaves whole rows and row
groups out of what it searches:

- Whether a candidate breaks a rule at a trailing character depends only on
  the trailing characters before it and on the state the leading ones left
  - the symbol run so far, the brackets open, the backslashes used (and
  whether the last character was one). For
  each such state (32 occur in a real search: 470 KB of list each, 15 MB
  in all), the host
  lists the row groups that survive, and for each, which of its rows do:
  the rules tell characters apart only by seven classes (letters, digits
  and space; other symbols; `(`, `)`, `[`, `]` and `\`), so a bit for each
  class the rows' own last character may be of stands for a set of them
  (`backends/common/row_pruning.h`). That covers text inserted after it
  too (`insert_from_end` at 1), which decides which of those characters
  survive. A group none of whose rows survive - one that can't reach
  `min_backslash_count` whatever follows, say - isn't listed. A list
  depends only on the alphabet, the rules, the text inserted into the
  trailing part, the trailing length and that state, so it's built the
  first time a batch needs it, uploaded once, and kept for the next search.
- The GPU kernels walk the batch's list instead of every row group, so a
  pruned group costs nothing. A pruned row, inside a group that isn't,
  still takes its turn - the lanes of a warp step through their rows
  together, so skipping one would save its lane only idle time - but gets
  no candidates. A kernel walking a list measured about 5% slower (CUDA;
  6% OpenCL) than one walking every group, so a search that doesn't prune
  the whole candidate gets the kernel that walks every group, as before -
  the kernels are compiled both ways. So does a CUDA or OpenCL launch whose
  lists leave out less than 5% of its row groups
  (`NAMEBREAK_LIST_MIN_PRUNED_PERCENT` in `backends/cuda/tuning.h`,
  `kListMinPrunedPercent` in the OpenCL backend): with few symbols in the alphabet, the rules
  hardly prune - `" -.0-9A-Z_"` with symbol runs pruned loses about 0.3% of
  its candidates - and the lists cost more than they save. The hits such a
  launch has in rows the lists leave out are dropped on the host
  (`RowPruning::survives`), so it reports what walking the lists would have.
  The CPU backend checks its rows as it walks them, and skips the pruned
  ones.
- A candidate's last character is never checked - for `min_backslash_count`,
  it's assumed to be a `\` if it may be one. Skipping one would save
  nothing - its row is hashed anyway, and at most a few of its candidates
  get past the lookup filter - so every candidate a row has is searched,
  whatever its last character. That the last character is left out is part
  of what the setting means, so that every backend searches exactly the
  same candidates.

That skips about a fifth more of the candidate space at every length than
pruning only the leading characters (worked out exactly for the real
alphabet, both rules on - see [PERFORMANCE.md](../PERFORMANCE.md)): 33.4% of
all 10-character candidates instead of 20.3%. Measured with `search_bench
--whole` on the RTX 3080 Ti Laptop, a real search's projected rate goes up
about 11% on CUDA and 10% on OpenCL: less than the fifth fewer candidates,
because of the list-walking kernel's 5-6% and the pruned rows that still
take their turn. Of the 6,406 file names (without extension) of the
StarCraft listfile in `tests/data`, one breaks a rule before its last
character (`STAREDIT\WAV\STEAL!  THE!  BEACON!!!!.WAV`); none only at it.
It changes which candidates a search covers, so it's a setting - and every
client searching a target has to agree on it, which is why a target has
it: a client too old to know it searches as if it were off, more than it
has to, never less.

## Inserted text

`insert_from_start` and `insert_from_end` add fixed text to every candidate
long enough for it. Where a candidate length puts that text decides what it
costs - the engine works it out once per length:

- In the leading part (`insert_from_start` on all but the shortest
  candidates, `insert_from_end` five or more from the end): hashed on the
  CPU with the leading characters, by `IncrementalPrefixHasher`. Nothing on
  the GPU - within the noise, measured.
- Between a row group's characters, or after them (`insert_from_end` at 2
  to 4): hashed with them, once per thread. A `\` at 2 cost about 4%.
- After the candidate's last character (`insert_from_end` at 0): part of the
  suffix for that length, like a longer suffix - a `\` about 4%.
- After a row's own character, before the candidate's last (`insert_from_end`
  at 1): a hash step for every row and inserted character, in the kernel's
  inner loop - a `\` cost about 16%, two characters about 28%.

(Measured with `search_bench --insert-from-end '\,1'` and so on, CUDA, on
the RTX 3080 Ti Laptop.) A search with text in the trailing part gets a
kernel of its own, compiled with it - CUDA's with the alphabet's size taken
at runtime, OpenCL's and Metal's with the text's place compiled in - so a
search without any runs exactly as fast as before the setting existed.

Text inside the trailing part reaches the backend as part of the search's
constants (`SearchConstants::trailingInsertions`), counted from the end so
that it doesn't depend on the trailing length. A length that puts it
anywhere else in the trailing part, or after the last character, than the
length before (only ever the shortest ones) begins the backend's search
again. The bounds, and the positions the coordinator hands out, stay
without it.
