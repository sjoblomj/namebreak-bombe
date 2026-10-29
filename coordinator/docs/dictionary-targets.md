# Dictionary targets

A dictionary target's candidates are made of words: one to `max_words` of
them, with a separator between each two - every word from its word lists,
merged (normalized as Storm hashes names - letters uppercase, `/` as `\` -
sorted, and without duplicates), every separator from its `separators` -
and then, if it has `tails`, a tail. A filename is prefix + candidate +
suffix, so `MUSIC\` + `BATTLE` + `_` + `THEME` + `2` + `.WAV`. It's the client's dictionary mode (see the client's
[Dictionary mode](../../client/docs/dictionary-mode.md)), shared out.

`english-1`, the dictionary compiled into the client, is built into the
server too (`server/data/english-1.txt`, a copy of
`client/data/english-1.txt` - a test checks the two are the same), and
stored as a word list whenever it starts, so a target can use it as it is.
Store any other word list first, one word per line - read as the client
reads them, so blank lines, lines starting with `#` and lines with anything
but printable ASCII are left out (the answer says which lines were):

```sh
curl -X PUT localhost:8080/api/v1/admin/word-lists/sc-units \
  -H 'X-Admin-Token: devsecret' --data-binary @sc-units.txt
# {"name":"sc-units","word_count":212,"checksum":"5e0c...","created_at":...}
```

A word list never changes once stored: clients keep a copy of each one they
download, under its name, so the same name again is only accepted with the
very same file, and a different list needs a different name (409
otherwise). The `english-` names are kept for the dictionaries built into
the server and the client: they can't be uploaded or deleted. A client uses
its own `english-1` rather than download it. A name is 1 to 64 letters,
digits, `.`, `-` and `_`.
`GET /api/v1/admin/word-lists` lists them; `DELETE
/api/v1/admin/word-lists/{name}` deletes one no target uses. Clients
download them from `GET /api/v1/word-lists/{name}`.

Then the target - `dictionary` instead of the alphabet and its settings:

```sh
curl -X POST localhost:8080/api/v1/admin/targets \
  -H 'X-Admin-Token: devsecret' -H 'Content-Type: application/json' \
  -d '{
    "name": "music-words",
    "prefix": "music\\", "suffix": ".wav",
    "hash_a_hex": "0x216A81D3", "hash_b_hex": "0x5A1F2C3B",
    "dictionary": {
      "word_lists": ["english-1", "sc-units"],
      "separators": ["", "_", "-", " "],
      "min_words": 1, "max_words": 2,
      "tails": ["digits:0-1"]
    },
    "lower_bound": "MUSIC\\BG", "upper_bound": "MUSIC\\BH",
    "encryption_key_hex": "0x1D5AD26C", "send_basenames": true
  }'
```

- `word_lists`: stored lists' names. `separators`: at least one, each once,
  printable ASCII - `""` writes words together. `min_words` (default 1) to
  `max_words`: at most 8, and no more candidates than 2^63 - two words of
  `english-1` with four separators are 16.3 billion, three 4.2 * 10^15.
- `tails` (default none): what follows the last word - every string made of
  one of each element's, in turn, each once. An element is `digits:A-B`
  (every string of A to B digits, `digits:N` exactly N), `letters:A-B` (the
  same of A-Z) or `X|Y|...` (one of these literal strings, `""` among them
  if it ends with `|`): `["digits:1-2", "letters:0-1"]` makes 2,970 tails,
  `0` to `9Z`, for names like `PROTOSS1` and `KEEP3B`. At most 8 characters
  a tail, no `\`, and 1,048,576 tails; the candidates are as many times as
  many (see **Tails.** in the client's [Dictionary
  mode](../../client/docs/dictionary-mode.md)).
- `lower_bound`/`upper_bound`: whole filenames this time (prefix, candidate
  and suffix), inclusive, either or both left out for none. Candidates are
  numbered with their first word slowest, so bounds mostly pick out first
  words: the server hands out, for each number of words, the numbers from
  the first first word the bounds let in to the last (exactly, for one-word
  candidates), and the client skips any candidate in between that's outside
  (`AB_` sorts after `ABC`). A target whose bounds leave no candidate is
  refused.
- `prefix`, `suffix`, separators and bounds are stored normalized, as the
  client normalizes them.
- `encryption_key_hex`: with a key, clients compare every candidate's
  basename - what follows its last `\` - to it, and, unless the suffix has
  a `\`, only the candidates whose basename matches to the hashes: the
  file's name has the key's basename. That makes the search about a third
  faster (see the client's [Dictionary searches on the
  GPU](../../client/docs/dictionary-mode.md#dictionary-searches-on-the-gpu)) - and
  means a wrong key hides the file.
- `send_basenames` (default false, needs `encryption_key_hex`): clients
  also send every basename that matches the key, so a file's name is found
  even when the directory searched is the wrong one. About one candidate in
  2^32 matches by chance: some four in two words of `english-1` with four
  separators, about a million in three (see **Basenames** below).
- `name`, `priority`, `description`, `encryption_key_hex` and
  `base_file_name` are as for any target; the alphabet, the pruning rules,
  the insertions, `start_len` and `auto_priority` aren't for a dictionary
  target (400), and nor are priority and skip ranges.

`PATCH` takes `name`, `status`, `priority`, `description`,
`encryption_key_hex`, `base_file_name` and `send_basenames` - the words,
separators, word counts, tails, prefix, suffix and bounds can't be changed.

A dictionary target's ranges are ranges of candidate numbers, sized by each
client's rate at dictionary searches, measured apart from its rate at
alphabet ones (`users.ema_dictionary_rate_per_sec`; a GPU searches some 20
billion a second, a CPU some 0.2 billion) - `DEFAULT_DICTIONARY_RATE_PER_SEC`
until it's measured. A range never spans two numbers of words, which is its
`candidate_len` (the dashboard's Words column). A claim of one carries the
word lists' names and checksums, the separators, the word counts, the
tails' elements and the checksum of the tails they make if the target has
tails, the range's first and end numbers, the bounds, the key if the target
has one, and whether to send basenames (see
`ClaimResponse::dictionary` in `protocol/src/lib.rs`); the client checks its
copy of each word list against its checksum, and the merged words and the
tails it makes against theirs, before it searches - handing the range back
if any differs. Heartbeats and a quit report progress as a
candidate number (`next_candidate_number` - every candidate numbered below it
has been searched) rather than a Hash A match, so a dictionary range is
checkpointed at every heartbeat, and a quit splits it exactly there. Clients
older than protocol 1.5 never get a dictionary target's work, nor a canary
made from one (see [Canaries](how-it-works.md#canaries)).

**Basenames.** A client sends the basenames it finds with its next
heartbeat, quit or completion, at most 5,000 a report
(`MAX_BASENAMES_PER_REPORT`), keeping the ones the server hasn't got yet in
memory until a report gets through. A report never says the search has got
further than a basename it doesn't carry: its `next_candidate_number` is at
most where the first one it leaves waiting was found. A completion says the
whole range was searched, so the client sends more than it can carry in
heartbeats first, one straight after another. So whatever happens to the
client - a crash, a quit, a lost connection - a basename the server never
got lies in a part of the range it will hand out again, and is found again:
none is ever missed, and the client keeps no files of them. The
server keeps each basename once per target, with who sent it first and in
which range, whatever else the report gets - a 409 included, as it's a fact
about the target, like a find - so a client counts them delivered on any
answer but an error. A basename that can't be one (empty, longer than 255
characters, with a `\` or anything but printable ASCII) is left out, and
nothing else is done with them: they're kept for whoever searches them. The
dashboard marks a dictionary target with a Dictionary badge, and gives it a
Basenames tab - all of them, up to the first 100,000 - and a Dictionary tab
with its word lists' words (`english-1` is only mentioned). `GET
/api/v1/targets/{id}/basenames` (public, like the dashboard) lists all of
them, one per line, in the order they came, and `GET
/api/v1/targets/{id}/word-lists/{name}` (public too) a word list the target
uses, as uploaded.

There can be a lot of them: about one candidate in 2^32 matches a key by
chance, so a target of three words of `english-1` with four separators
(4.2 * 10^15 candidates) gets about a million - 31 MB as text. So they're
kept compressed (`server/src/basenames.rs`): each report's new ones as one
gzip member (`basename_reports`), and an 8-byte hash of each
(`basename_hashes`) to keep each once. Measured with a million of them sent
240 a heartbeat, as a GPU client sends them: 33 MB in the database - 17 MB
of reports and 15 MB of hashes - where a row each took 100 MB; the Fly
volume is 1 GB. The server is built for that many: each report adds at most
5,000, in one transaction; a target's count is kept as they come
(`targets.basename_count`) rather than counted; the dashboard reads only a
target's latest reports, for its 20 latest basenames; and the full list is
read and sent 20 reports at a time (`BASENAME_REPORTS_PAGE`), so it's never
all in memory, and the database - one connection, which every claim and
heartbeat needs too - is only held for a page's query, however slowly the
list is downloaded. Two words make only a handful; `send_basenames` is off
by default, so a target only collects them when it's asked to. And should
the disk still run low - under `MIN_FREE_DISK_BYTES` free where the
database is (`server/src/disk.rs`) - the server discards every basename
reported until there's room again, rather than let the database run out of
it: the reports are answered as if they'd been kept, so those basenames are
lost. The dashboard warns in a yellow box at the top before that, from
under `WARN_FREE_DISK_BYTES` free, so there's time to make room, and says
so once they're being discarded.

Every answer is gzipped for a client that says it takes gzip
(`Accept-Encoding` - a browser, `curl --compressed`, the client's word list
downloads): the full list of a million basenames is then 11 MB rather than
31, and the dashboard's JSON and the word lists shrink about as much.
