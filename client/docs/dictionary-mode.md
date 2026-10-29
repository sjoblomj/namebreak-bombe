# Dictionary mode

Instead of every string of an alphabet, `mode = dictionary` searches
candidates made of words: one word, or several with a separator between each
two, every word from the same list and every separator from the same list.
With the words `CRDT` and `LST` and the separators `""` and `_`, the
two-word candidates include `CRDTLST`, `CRDT_LST` and `LST_CRDT`. A
filename is still prefix + candidate + suffix.

For example, the basename of a credits file whose directory isn't known -
no prefix, so only the basename can match, by its encryption key:

```ini
mode = dictionary

[dictionary]
dictionaries = lists/starcraft-words.txt
max_words = 2
separators = "", "_", "-", " "
prefix = ""
suffix = .txt
hash_a = 0xC4F43358
hash_b = 0x7694C48D
encryption_key = 0x4565C467
matches_name = sc-ptbr-credits
resume_from_last_candidate = true
```

`[dictionary]` keys:

| Key | Required | Meaning |
|---|---|---|
| `builtin_dictionary` | no (default `english-1`) | The dictionary compiled into the program, or `none` to search only `dictionaries`. `english-1` is 63,875 English words - see below. |
| `dictionaries` | no | More word lists, as a list of files (relative to the current directory, unless absolute): one word per line. Spaces and tabs around a word are dropped; blank lines and lines starting with `#` are skipped; a line with anything but printable ASCII is skipped, with a warning. The words of every list and the built-in dictionary are searched together, each once. |
| `min_words` / `max_words` | `max_words` yes (`min_words` default `1`) | How many words a candidate has: every count from `min_words` to `max_words` (at most 8), fewest first. |
| `separators` | no (default `""`) | What goes between two words, as a list: `""` writes them together. `"\"` makes the words before it directories. |
| `tails` | no (default none) | What follows the last word, as a list of elements: every string made of one of each element's, in turn, after every last word - see **Tails.** below. |
| `prefix` / `suffix` | yes | The fixed parts of the filename around the candidate. |
| `lower_bound` / `upper_bound` | no | Whole filenames (inclusive) bounding the search alphabetically - typically an unknown file's neighbours in the archive. Either may be left out. |
| `hash_a` / `hash_b` | yes | The two target MPQ hashes, hex. |
| `encryption_key` | no | The file's raw encryption key, hex: hash type 3 of its basename, before the adjustment for its position and size in the archive (mpqcli's `encryption-key-raw`, the coordinator's `encryption_key_hex`). Every candidate whose basename - what follows the filename's last `\` - hashes to it is recorded (see `record_basenames`), and, unless the suffix has a `\`, only those candidates are compared to `hash_a` and `hash_b` at all: the file's name has that basename, so no other can be it. That makes the search about a third faster (see [Dictionary searches on the GPU](#dictionary-searches-on-the-gpu)) - and a wrong key hides the file (see `record_hasha_matches`). |
| `record_basenames` | no (default `true` with an `encryption_key`, else `false`) | Whether to record the basenames that match the key. `true` needs an `encryption_key`; `false` still uses the key. |
| `record_hasha_matches` | no (default `false`) | With an `encryption_key`, compare every candidate to `hash_a` and `hash_b` all the same, as if the key weren't known: every Hash-A match is recorded in `matches.txt`, and a wrong key can't hide the file - but the search is about a third slower, as before the key was used for that. Without an `encryption_key`, every candidate is compared anyway. |
| `hash_table_size`, `hash_table_first_start_slot`, `hash_table_last_start_slot` | no (all three or none) | As in `[search]`. The `encryption_key` above isn't checked again: it already decides which candidates are compared to the hashes, and with `record_hasha_matches` a match of both hashes is the find whatever its basename. |
| `matches_name` | no | Names the files the search writes, as in `[search]`. |
| `resume_from_last_candidate` | no (default `false`) | Carry on from the progress file (see below). |

As everywhere, letters are made uppercase and `/` made `\` - in the words,
separators, prefix, suffix and bounds - since that's how Storm hashes a name.

A list (`dictionaries`, `separators`) is items separated by commas, each
either `"quoted"` - kept exactly, spaces and commas included - or not, and
then trimmed. `""` is an empty item; an empty unquoted one is an error.

The Windows GUI's setup has a Local Dictionary tab with these keys (a Browse
button adds word lists, relative to the current directory when they're in
it), and checks them as a search would before it writes them. While it
searches, its progress bar is how many of the candidates it has hashed, and
the basenames that match the key are listed with the Hash-A matches.

**Tails.** Names often end with a number, or a number and a letter -
`PROTOSS1`, `KEEP3B` - which `tails` adds after the last word, before the
suffix. Each element of the list stands for some strings:

- `digits:A-B` - every string of A to B digits, leading zeros and all:
  `digits:1-2` is 0-9 and 00-99. `digits:N` is exactly N.
- `letters:A-B` - the same of the letters A-Z.
- `X|Y|...` - one of these literal strings, `""` among them if the element
  ends with `|`: `_|` is an underscore or nothing.

The tails are every string made of one of each element's, in turn, each
once: `tails = digits:1-2, letters:0-1` makes 2,970 of them, `0` to `9Z`,
and `tails = "_|", digits:2` makes `00` to `99` and `_00` to `_99`. A tail
is at most 8 characters, without a `\` (it's part of a basename), and there
are at most 1,048,576 of them. Every candidate is followed by each, so a
search has as many times as many candidates - but a GPU hashes a word once for a
run of its tails, so a candidate with a tail costs less than one without
(see [Dictionary searches on the GPU](#dictionary-searches-on-the-gpu)).

**The numbers.** Every candidate has a number, counting from 0 the way an
odometer counts, with a wheel per word, per separator and for the tail:
all one-word candidates first, then all two-word ones, and so on; within
those, the first word turns slowest, then the separator after it, then the
last word, and the tail fastest. The words are sorted, so the candidates of
each word count come in alphabetical order of their first word. The numbers
depend only on the words, the separators, the word counts and the tails -
not on the bounds, prefix or suffix - and without tails, they're what they
were before there were tails. The search prints how many candidates there
are in all, and how many it will search within the bounds.

**The bounds** compare whole filenames. A word that already puts every
filename starting with it outside the bounds is skipped, with everything
after it; one that puts every such filename inside isn't compared again.
Only the few words a bound cuts through - with `REZ\CRDT_LST.TXT` as the
lower bound, `REZ\CRDT...` is both inside (`REZ\CRDT_MAP.TXT`) and outside
(`REZ\CRDTAARDVARK.TXT`) - have their candidates compared one by one.

**What it writes**, in `matches_dir`, named with `matches_name` if given:

- `matches.txt` - the most recent Hash-A match, as in the other modes (of
  the candidates compared to the hashes at all, see `encryption_key`), and
  `found.txt` for a match of both hashes, where the search stops.
- `basenames.txt` - with `record_basenames`, every basename that matched the
  encryption key, one per line, each once (those already in the file
  included). A basename match is only 32 bits, so about one candidate in
  4,294,967,296 matches by chance: the search prints how many to expect
  before it starts. With two words of `english-1` that's a handful; with
  three, about a million - narrow the search, or turn `record_basenames`
  off, if that's not what you want.
- `wordnumber.txt` - the progress file: every candidate numbered below
  `next` has been searched. Written every 30 seconds, when the search is
  paused, and when it ends. It also has a fingerprint of everything that
  decides which candidates are searched and what's looked for (the words,
  separators, word counts, tails, prefix, suffix, bounds, hashes, encryption key,
  and whether basenames and every Hash-A match are recorded):
  `resume_from_last_candidate = true` refuses to resume from a file another
  search wrote, rather than skip candidates the changed search never
  searched - unless that search did at least as much as this one: compared
  every candidate to the hashes where this one compares only those whose
  basename matches the key (with `record_hasha_matches`, or without an
  `encryption_key`), or recorded the basenames where this one doesn't. So
  turning `record_hasha_matches` off part way through a search carries on,
  and turning it on starts again. A search with an encryption key whose
  basenames are recorded keeps the fingerprint it had before the key was
  used to hash less.

**`english-1`** is every all-lowercase ASCII word of Debian's `wamerican`
word list (from SCOWL) - see `data/english-1.LICENSE.txt` for where it comes
from and its licence. It never changes: a different list would be
`english-2`, so a search numbered over one can't be resumed over another.

The basename alone is often worth searching for: with an encryption key and
the wrong directory - or none, `prefix` empty - the hashes never match, but
the right basename still lands in `basenames.txt`, leaving only the
directory to find.

**Backends.** Every backend searches dictionaries; unless `backend` says
otherwise, the first that can run on the machine is used, as in the other
modes. On the RTX 3080 Ti Laptop, up to two words of `english-1` with four
separators and an encryption key - 16.3 billion candidates,
`run_dictionary_bench` - take about 0.6 s on the GPU, 26-27 G candidates/s
with CUDA and about 25.5 with OpenCL (with the GPU at 77-82 C), and about
28 s on the CPU backend, 0.6 G candidates/s. A short word list with many
words a candidate, where each batch once filled a fraction of a block, is
several times faster now (see [PERFORMANCE.md](../PERFORMANCE.md)'s *Pairs*):
337 words, five a candidate, go from 2.7 to 30 G candidates/s on CUDA. A
dictionary search runs a
self-test of its own first (`selfTestDictionaryBackend`,
`src/backends/self_test.h`): known answers planted among 9,000 words, which
the backend must find before it's used. See [Dictionary searches on the
GPU](#dictionary-searches-on-the-gpu) for how the kernels go about it.

## Dictionary targets in coordinator mode

A coordinator target can be a dictionary target (see the coordinator's
[Dictionary targets](../../coordinator/docs/dictionary-targets.md)): a claim of one hands out a range of candidate
numbers, with the word lists, separators, word counts and bounds to number
them by, and the client searches them as dictionary mode does, on the same
backend - which the first dictionary range runs the dictionary self-test on,
falling back to the cpu backend if it fails it. It needs protocol 1.5 (this
client's, `kProtocolVersion` in `src/net/protocol.h`); older clients never
get one.

- **Word lists.** `english-1` is the one compiled in. Any other list the
  server has is downloaded the first time a claim names it, and kept in
  `word-lists/<name>.txt` in the matches directory; a claim carries each
  list's checksum, and a copy without it is downloaded again. The merged
  words have to have the checksum the claim gives them too - otherwise the
  client hands the range back rather than search other candidates than the
  server numbered (`src/net/word_lists.h`).
- **Progress.** Heartbeats (every minute) and a quit say how far the search
  has got as a candidate number - every candidate below it has been searched -
  so the server checkpoints a dictionary range at every heartbeat, and a quit
  hands back exactly the rest.
- **Basenames.** A target's encryption key is used whenever it has one, as
  `encryption_key` is in dictionary mode. When the target asks for them
  (its `send_basenames`), every candidate whose basename matches the key is
  sent with the next heartbeat, quit or completion, at most 5,000 a
  report. Until a report gets
  through they wait in memory - never in a file - so if the server can't be
  reached, they go with the next report that is. A report never says the
  search got further than a basename it doesn't carry, and a completion
  (the whole range searched) only goes once the rest fit in it - more go in
  heartbeats first, one straight after another
  (`src/net/basename_outbox.h`). So a basename the server never got is
  always in a part of the range it will hand out again, and found again,
  however the client stops: nothing is missed, and nothing is left on the
  user's disk. The server keeps them whatever else it answers (a 409
  included), so only a report that gets no answer, or another error, leaves
  them waiting.

The dictionary engine itself is unchanged by this: a range is a
`DictionaryRequest` with an `endNumber`, without a basenames or progress
file, and with `DictionarySearchHooks` (`src/engine/dictionary_search.h`)
telling the coordinator loop each basename and how far it has got.

## Dictionary searches on the GPU

A dictionary search has nothing
like a row for the GPU to share work along: each candidate is a word of its
own, after a leading part - the prefix and the words and separators before
the last word. The engine (`runDictionarySearch`) walks the leading parts on
the CPU, hashing each once, and hands the backend *batches*: a leading part's
two hash states (hashA's, and the basename hash's - hash type 3 of what
follows the last `\`), followed by a run of the word list's words, each with
a run of the tails, and then the suffix. A batch whose leading part the
bounds put wholly inside stops a word earlier and carries *pairs* - a word
of the list and the separator after it - before its words, so a short
list's many leading parts become one batch's pairs rather than a batch each
(see *The last two words* below). A GPU launch is a call's batches, up to
2^28 candidates, cut into *segments* of up to 8,192 *cells* of one batch,
along its pairs (32 cells to each of 256 threads), one per thread block
(`planDictionaryLaunch`, `src/backends/common/dictionary_batch.h`). A cell
is a word and a chunk of up to 32 of its tails - without tails, a word. A
block finds its batch by a binary search over the batches' first segments,
works out the states after each of its pairs, and its threads take
neighbouring cells: neighbouring words of one pair, with the same chunk of
tails.

What makes it fast (see [PERFORMANCE.md](../PERFORMANCE.md)'s *Dictionary
searches* for the measurements):

- **One hash a candidate.** A search with an encryption key hashes each
  candidate's basename alone, and compares it to the key: the file's name
  has that basename, so a candidate whose basename doesn't match can't be
  it, and its hashA would be hashed for nothing. The host checks hashA and
  hashB of the few that match (`DictionaryHitVerifier`). A search without
  one hashes hashA alone. Hashing both, as the kernels first did, took
  about a third longer - and is what they still do with
  `record_hasha_matches` (unless the basenames aren't recorded: then only
  hashA).
- **A word hashed once for a run of its tails.** With tails, a thread hashes
  its cell's word once, and then each of its chunk's tails - one to eight
  characters each - from there. The lanes of a warp take the same chunk of
  neighbouring words, so they read the same tail, as one load, and go round
  its loop together. A search without tails gets a kernel without the loop
  (each is compiled both ways).
- **The last two words.** A short word list makes short batches - 337 words
  are 337 cells, and a block has 8,192 - so a batch that was a leading part
  and the list's last word now stops a word earlier and holds the list's
  *pairs* (a word and a separator, pair `p` being word `p / S` and
  separator `p % S`) before its last word: 337 words and four separators
  are one batch of 1,348 pairs, not 1,348 batches. A segment is cells of
  one batch along its pairs, so it never mixes a pair's cells with
  another's - a pair of more than 8,192 cells is cut into segments of its
  own, and smaller pairs share a segment, up to `kDictionaryPairSlots` of
  them. A block hashes the state after each of its pairs into shared memory
  first (one `__syncthreads` away), then its threads take runs of 32 cells
  (`dictionaryLanesPerRun`), each run a pair's, hashing on from that pair's
  state. The engine builds whole pairs into the batch and searches a pair a
  call or a bound stops inside as a run of its own, its word and separator
  on the leading part.
- **Words by length.** A warp goes round a word's loop as many times as its
  longest word needs. The word table the kernels read
  (`DictionaryWordTable`) has the words in the order of their lengths, so
  neighbouring threads have words of the same length - which a batch of
  every word, nearly every batch, is searched in. A batch of part of the
  list (at a bound, or where a call ends) is searched in the list's order.
  A hit is reported by the word's index in the list either way, so the
  order is the kernel's business only.
- **Four characters to a load.** The table has four characters to a
  uint32, each word from a uint32 of its own, and the crypt table's part
  for the hash is in shared memory.
- **The suffix filter.** As the row search's [lookup
  filter](lookup-filter.md): the low 7
  bits of seed1 after the suffix depend only on the low 7 bits of seed1 and
  seed2 before it, so a 16,384-bit table per hash, built and checked in full
  for every search, says whether the suffix can make a match - and it's
  hashed in full for 1 candidate in 128.

A word with a `\` starts the basename hash over after its last one (the
table has where), so what comes before it isn't hashed at all, and a suffix
with a `\` makes every candidate's basename the same, its end - checked once
a call on the host, not per candidate (`DictionaryHitVerifier`), while the
kernel hashes hashA. The kernel records hits as (batch, word, tail, pair);
the host rebuilds each filename - the batch's leading part, the pair's word
and separator, the word and the tail - hashes it from scratch and checks
hashB - and, for a basename hit, hashA first - as for the row search. A launch has room for 4,096 hits of each kind - a
search with a 32-bit target has about one hit per 2^32 candidates - and one
that has more is searched again with room for all of them, so none is ever
lost. The OpenCL and Metal kernels are the CUDA one ported, compiled at
runtime once per suffix length, with or without basenames.
