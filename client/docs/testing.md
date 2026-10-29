# Tests and benchmarks

`ctest --preset default` runs the correctness test suite (see [Building
namebreak](building.md) for building it). The tests that search do so on
every backend in the build; a test whose backend can't run on the machine is
reported as skipped - see [Backends](backends.md#backends) for why a
machine with GPUs should configure with `-DNAMEBREAK_REQUIRE_GPU=ON`.

The search itself is built in several configurations for the tests
(different GPU window, launch sizes and rows per thread, the stress tests'
weaker match, a one-bit filter), each with its own compile of the CUDA
backend - the build runs them in parallel. `-DNAMEBREAK_TEST_VARIANTS=OFF`
leaves those out (the release builds do), which builds much faster but
tests less.

## The search

- **Unit tests, on the CPU**: `prefix_hasher_test` (the incremental hash of
  the leading characters), `lowbits_filter_test` (the [lookup
  filter](lookup-filter.md), also built for other numbers of bits),
  `match_check_test` (what a match of both hashes is checked against - see
  [false matches](configuration.md#false-matches)), `matches_file_test`, `config_test`
  (every section and key of `config.conf`, which GPUs `[hardware]` picks,
  and how `--autotune` picks its values) and `version_test` (release
  versions and the update check, without the network).
- **`integration-*`**: end-to-end tests of every backend that compare real
  search results against an independent brute-force reference - thousands
  of cases: alphabet sizes from 1 to 63, every last-character position,
  ranges starting/ending mid-row and crossing launch boundaries,
  prefix/suffix lengths 0-63 including bytes >= 0x80, the both-hashes-match
  path, and a seeded fuzzer. Also built with tiny launches
  (`-small_launch`) and with GPU windows of 2 and 6.
- **`stress-*`**: the dense-hit stress tests, which make one candidate in
  4096 a hit and compare every one of them with a brute force - see [The
  lookup filter](lookup-filter.md#what-guards-the-implementation). Also
  built with tiny launches, windows of 2 and 6, a one-bit filter
  (`-filter1`), room for a single hit per launch (`-overflow`), and, for
  the CPU backend, its plain loops rather than AVX2 (`stress-cpu-plain`).
- **`overflow-*`**: more hits in a launch than it can record, which must
  still all be checked.
- **`listfile-*`**: a known-answer test, which must find each of the 6,407
  names of a real StarCraft listfile (`tests/data/sc-listfile.txt`) from its
  two hashes, in a small search planted around it - with hashes from the
  test's own copy of the MPQ hash, so it also catches a bug the client's
  hashing shares with every other test.
- **`self_test_test`**: that the self-test every backend runs before it's
  used actually catches backends broken in the ways a GPU kernel, driver or
  compiler could break them.

## Dictionary mode

- `dictionary_test` - numbering, bounds and counting against brute force.
- `wordlist_test` - pins `english-1`.
- `dictionary_batch_test` - the word table, suffix filters, launch plan and
  hit checking the GPU backends share.
- `dictionary-search-*` - the dictionary self-test, then whole searches on
  each backend against brute force: also with one candidate in 256 a hit,
  and one in 256 a basename matching the key by chance, with tails, and
  with few candidates and batches per call, stopped and resumed part way,
  the GPU kernels' thread blocks a few words each and three tails to a
  cell, room for a single hit and a one-bit suffix filter, and cut into
  ranges of numbers as a coordinator hands them out.
- `dictionary-cli-*` - the program itself.

## Meet-in-the-middle mode

- `meet_in_the_middle_test` - the hash run backwards, the tails' numbering,
  the backward walk, the table and the pruning rules split between the two
  halves, all against brute force.
- `meet-in-the-middle-search-*` - the meet-in-the-middle self-test, then
  whole searches on each backend against brute force at every split between
  the table and the walk: also with three bits of each of the meeting
  point's seeds compared, so about one directory in 64 matches it by
  chance, and with a few tails per call, room for a single hit per launch,
  and ranges, stopping and resuming and pausing.
- `meet-in-the-middle-cli-*` - the program itself, recovering `SETUPDAT`
  from the state after it.

## Coordinator mode

- `protocol_test` - the JSON: an alphabet claim as older clients read it,
  and a dictionary claim with its arrays.
- `basename_outbox_test` - the basenames waiting to be sent, and that no
  report says the search got past one it doesn't carry - also while one
  thread adds and another sends.
- `word_lists_test` - a claim's word lists: compiled in, kept, downloaded
  and checked.

When cargo and Python 3 are there, four tests run the real server,
`../coordinator`, against the program in coordinator mode, over HTTP on
localhost:

- `coordinator-dictionary` (`tests/coordinator_dictionary_test.py`):
  quitting part way through a dictionary range, searching dictionary and
  alphabet targets to the end, the server going away and coming back while
  basenames wait to be sent (with `namebreak_heartbeat_1s`, built to
  heartbeat every second), and more basenames than a report carries (with
  `namebreak_one_basename_per_report`, built to carry one, and pairs of
  `english-1` candidates whose basenames share a key): a quit carrying the
  first of two whose range, handed out again, finds the second; a
  completion after the first is sent in a heartbeat; and a backlog sent one
  heartbeat straight after another.
- `coordinator-meet-in-the-middle`: that a client gets no work from a
  target whose table is bigger than the memory it offers; finding a
  directory; a quit part way, resumed from exactly the tail it got to; and
  a meet-in-the-middle canary.
- `coordinator-false-match`: an older client's false match, which the
  server catches itself and hands the rest of the range out again; and this
  client reporting false matches and searching on past them.
- `coordinator-workers`: one client searching as two workers (as with two
  GPUs, `namebreak_two_workers`), sharing a target's ranges, each with its
  own matches file, and quitting together.

## Benchmarks

Not tests; each is run with `cmake --build --preset default --target
<name>`:

- `run_search_bench` times the real search over a fixed range
  (`build/tests/search_bench --scale <n>` for a longer one, `--backend
  <name>` for another backend), pruning as the real configuration does
  (`--prune symbols` or `none` for less, `--whole` to prune the whole
  candidate; `--alphabet`, `--prefix` and `--suffix` time a coordinator
  target's). It reports the range covered per second, the rate at which the
  backend searched what pruning left, and a projection for a real search,
  whose leading characters all vary - the timed range only varies its last
  few, so it prunes far less than a real search would. The projection takes
  the share a real search prunes from random leading values put through the
  engine's own checks; compare GPU windows by it.
- `run_dictionary_bench` times a real dictionary search: two words of
  `english-1`, about 16.3 billion candidates (`build/tests/dictionary_bench
  --words <n>` to time the CPU backend on fewer, `--no-key` to hash hashA
  rather than the basename, `--record-hasha-matches` to hash both).
- `run_prefix_hasher_bench` times the incremental hash of the leading
  characters, CPU only.

`run_mutation_test` (and `run_mutation_test_opencl`) checks that the tests
catch deliberately broken kernels - see [below](#the-mutation-experiment) -
which takes from minutes to hours, so it isn't part of `ctest`.

Everything is built into `build/`; deleting it is the clean build.

## The mutation experiment

To check that the tests and the checks that run on every search (see [The
lookup filter](lookup-filter.md#what-guards-the-implementation)) would
actually catch a broken kernel, broken
ones are built on purpose - each a bug that silently misses (or invents)
candidates - and run against each check on its own (the integration and
stress tests with the self-test switched off). `tests/mutation_test.py`
does it, a backend at a time: `cmake --build --preset default --target
run_mutation_test` for the CUDA kernel (about an hour on the RTX 3080 Ti
Laptop, with the pruning mutations below), `run_mutation_test_opencl` for
the OpenCL one (about four). Run it after any change to a kernel - and when
it says a mutation no longer applies, because the code it breaks has
changed, update the mutation rather than drop it. It also runs two more
checks on every copy, for what the other three rarely or never reach: the
overflow test (more hits in a launch than it can record), and the stress
test built for tiny batches, a few per launch (`stress-small`) - with 16
leading values per launch, nearly every search of the other tests is a
single launch, and only this one has many, where what one launch leaves
behind can break the next. A dictionary search and a meet-in-the-middle
search have their own two checks each (the dictionary and meet-in-the-middle
search tests, built for dense hits and for small launches), which their own
mutations are held to. It first checks that the unchanged code passes every
check, and that a change which mustn't matter (one thread, or one
work-group, too few or too many launched) passes them too, so that it can't
pass by always saying "caught". Against the CUDA kernel as it is now - row
groups split into chunks, 16 batches per launch, and its hits read back with
its hit count - twenty-six mutations: rows' masks cut to 32 bits; seed1 and
seed2 swapped in the lookup; a batch's last row, or first row, one candidate
short; the edges of the range ignored; the loop over a row's flagged
candidates stopping one early; the suffix hashed one character short; the
table built for the wrong target; in the chunking, a chunk's first or last
row skipped; the range cut one row short at its start or end; the first or
last row's cut applied to the wrong row; one character too few of a group
hashed; the wrong character in a row's own step; the last chunk, or the last
group, of a batch never searched; and in the batching, every batch of a
launch searched with the first one's seeds and edges; every hit recorded as,
or checked with, the first batch's; a launch's last batch never searched;
the hits past the first 16 read one short, and the hit count never reset
after a launch with hits; and two in the engine - the batches left over at
the end of a candidate length never searched, and a launch whose hits
overflowed never searched again batch by batch. Every kernel mutation but
the last two was caught by the self-test, by the integration test and by the
stress test - the integration test failed 6-1007 of its 2,266 cases, the
stress test 17-315 of its 346, and the wrong-target table stopped both at
their first search. The rest can only be caught where they can happen: more
than 16 hits in a launch only in the stress test (which caught it), a count
left over from one launch only in a search of many (the self-test, which
searches its grouped batches twice in one search, and `stress-small`); the
engine's two not by the self-test, which tests a backend without the engine
(the first was caught by the integration and stress tests, the overflow one
by the overflow test alone). Since the CUDA kernel hashes a chunk's flagged
candidates after all of its rows, three more: only the first flagged
candidate of a chunk's last flagged row hashed, the flagged rows hashed from
the next row's state, and a row's flag recorded as the next row's - each
caught by all three checks. Against the OpenCL kernel, twenty-one: the same
kinds of bug in its mask, lookup, row edges, loop, suffix and chunking, plus
a lowest-set-bit taken one too high, its own copy of the table index
shifting one bit too far, and the kernel compiled for a table one bit
narrower - with half as many work-items, and a work-group too many, as the
changes that mustn't matter. Every one was caught by all three checks too
(the integration test failed 6-1007 of its 2,266 cases, the stress test
48-394 of its 436), each time the script was run on it. The Metal kernel has
its own list (`--backend metal`, on a Mac); with no Mac here, its kernel
mutations were run through an emulation of Metal on the CPU (see
[Backends](backends.md#backends)), and each was caught by all three checks there too,
while half as many threads, or a threadgroup too many, went unnoticed, as
they must. The CPU backend has its own list too (`--backend cpu`), since it
got the filter: the same kinds of bug in its mask, lookup, row edges, loop
and suffix, the highest flagged bit taken for the lowest, and in its own
code the row walk's characters wrapping one early, the wrong one rehashed
after a change, a work item's first row decoded one off, a row skipped
between two work items, a call's last batch never searched, hits recorded or
checked with the wrong batch, and an overflow hidden by counting a thread's
hits no higher than can be recorded - with larger work items, and everything
on one thread, as the changes that mustn't matter. Each was caught by the
checks it names (the integration test failed 17-1138 of its 2,338 cases, the
stress test 55-426 of its 511).

Each round of this found something. The first one added the self-test's
and the stress test's planted edge cases: before them, a launch whose last
row was one short got past both, and only the integration test caught it.
The second made the self-test plant candidates in the first and last row of
a row group, where it had missed a chunk's last row being skipped; and it
found that launching one thread too few went unnoticed by everything - it
only mattered when the thread count was just past a multiple of 256 - so
the kernel now works out its chunks itself, and how many threads are
launched can no longer change what gets searched. The latest, with 16
batches per launch, found that no test ever reached the engine's fallback
for a launch whose hits overflowed together - where each batch has to be
searched again on its own. Breaking it went unnoticed by every check, so
the overflow test now spreads a pair of colliding candidates over two
batches of one launch (in adjacent leading values, and with a pruned one
between them), and a stress-test build with room for a single hit per
launch (`stress-*-overflow`) overflows launches and batches at random. And
reading hits back with the count found that the default builds' tests had
stopped noticing what one launch leaves for the next - a hit count never
reset got past all three checks, as their searches had become single
launches - so the self-test now searches its grouped batches twice in one
search, and the script runs `stress-small` too. The CPU backend's first
run found that the self-test never stepped from one row to the next on it -
its short ranges were cut into work items a row long, each decoded afresh -
so a row walk wrapping its characters one early got past it; it now also
searches a range three row groups long, and finds a candidate a few rows
into it, in the last row of a group.
Pruning the whole candidate (see
[`prune_whole_candidate`](pruning.md#prune_whole_candidate-the-rules-on-every-character-but-the-last)) added mutations of its own: in the host's
lists of row groups (none left out, a class or a row mask wrong, a batch's
slice one group short at either end, lists cleared without the backend
noticing), in each kernel's walk of them (the wrong offset, count, group
or classes, a row's bit taken from the next row), in each backend's upload
and state, in the CPU backend's row walk, and in the engine (the backend
never told, or every batch started as if there were no leading
characters). Their first run found two gaps, both in how a launch's
batches get their states: pruning every batch with the first batch's
state, and counting one bracket too few of those the leading characters
leave open, got past every check but the self-test (the first) or all of
them (the second). The integration test now searches two leading values
in one launch, one leaving a bracket open and the next not, and a search
whose prefix leaves more brackets open than a row can close; each catches
its mutation. The rules that came after - min_backslash_count, and text
inserted into the candidate - found three more gaps, each a mutation one
check let through: a row kept when its last character can't make up the
backslashes still needed got past the self-test, which never set
min_backslash_count; the text inserted after a row's own character left
unchecked got past the integration test, which only ever inserted text
among a row group's characters with the rules on; and ')' taken for '('
in the rows' classes got past it too, as the alphabet it pruned brackets
with has ')' first, and it never planted a candidate in a row ending in
'('. The self-test now prunes for at least two backslashes, and the
integration test inserts a backslash before the last character, and plants
candidates in the row "BAAA(" beside the pruned "BAAA)"; each catches its
mutation, on CUDA and OpenCL.
`tests/self_test_test.cpp` does the same for the self-test on the CPU
backends, on every `ctest` run.
