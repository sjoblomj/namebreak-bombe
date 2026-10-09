use crate::alphabet::{join_pos, Pos};

// Several fields on these row types round-trip DB columns via `SELECT *` /
// sqlx::FromRow without every field being read back in application code -
// they're still needed for the row shape to match the table.
#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub hostname: String,
    pub token: String,
    pub ema_rate_per_sec: Option<f64>,
    pub created_at: i64,
    pub last_seen_at: i64,
    /// This client's declared protocol version (`"X.Y.Z"`), captured at
    /// `/register` time - see `namebreak_protocol::PROTOCOL_VERSION` and
    /// `handlers::register`. Consulted by `ranges::claim_range` so it never
    /// hands this client an alphabet introduced after the version it
    /// declared - see `alphabet::client_alphabet_for`.
    pub protocol_version: String,
    /// The backend this client searches with (e.g. "cuda"), as declared at
    /// `/register` time - informational only.
    pub backend: String,
    /// The client release it last registered with - see `client_release`.
    /// `None` for a client from before protocol 1.1.0.
    pub client_release: Option<String>,
    /// Like `ema_rate_per_sec`, of the dictionary targets' ranges it has
    /// completed - a few hundred times lower on the same GPU, so it's kept
    /// apart (see `ranges::claim_range`).
    pub ema_dictionary_rate_per_sec: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Target {
    pub id: i64,
    pub name: String,
    pub prefix: String,
    pub suffix: String,
    pub hash_a: i64,
    pub hash_b: i64,
    pub lower_bound: String,
    pub upper_bound: String,
    pub prune_symbol_runs: i64,
    pub prune_unopened_brackets: i64,
    pub prune_whole_candidate: i64,
    pub max_backslash_count: i64,
    pub min_backslash_count: i64,
    pub prune_adjacent_backslashes: i64,
    /// See `AdminCreateTargetRequest::insert_from_start` - None: nothing.
    pub insert_from_start_text: Option<String>,
    pub insert_from_start_position: i64,
    pub insert_from_end_text: Option<String>,
    pub insert_from_end_position: i64,
    pub alphabet_name: String,
    pub alphabet: String,
    pub status: String,
    pub found_filename: Option<String>,
    pub found_by_user_id: Option<i64>,
    pub created_at: i64,
    /// Higher claims first - see ranges::claim_range. Defaults to 0.
    pub priority: i64,
    /// Operator note shown on the target's dashboard card, rendered as raw
    /// HTML there - see dashboard.html.
    pub description: Option<String>,
    /// The shortest candidate length the main sweep carves - see
    /// `AdminCreateTargetRequest::start_len` and `ranges::claim_range`.
    pub start_len: i64,
    /// 1 for a canary's virtual target (see `canary.rs`), which takes part in
    /// nothing but its one claim.
    pub is_virtual: i64,
    /// 1 to make priority ranges from likely prefixes automatically - see
    /// `AdminCreateTargetRequest::auto_priority`.
    pub auto_priority: i64,
    /// The length at which the likely prefixes ran out - see
    /// `ranges::create_next_auto_priority_range`.
    pub auto_priority_exhausted_len: Option<i64>,
    /// The file's encryption key, stored like `hash_a` - see
    /// `AdminCreateTargetRequest::encryption_key_hex`. None: not known.
    pub encryption_key: Option<i64>,
    /// The file's name without its directory - see
    /// `AdminCreateTargetRequest::base_file_name`. None: not known.
    pub base_file_name: Option<String>,
    /// `"alphabet"`, or `"dictionary"` for a target whose candidates are made
    /// of words - see `AdminCreateTargetRequest::dictionary` and
    /// `dictionary.rs`. A dictionary target's ranges count candidate numbers
    /// (see `DICTIONARY_ALPHABET_NAME`), and its `lower_bound`/`upper_bound`
    /// are whole filenames, "" for none.
    pub kind: String,
    /// A dictionary target's word lists' names and separators, as JSON
    /// arrays of strings (see `dictionary::parse_string_list`), and its word
    /// counts. None for an alphabet target.
    pub word_lists: Option<String>,
    pub separators: Option<String>,
    pub min_words: Option<i64>,
    pub max_words: Option<i64>,
    /// A dictionary target's tail elements (see `dictionary::expand_tails`),
    /// as a JSON array of strings - None for an alphabet target, or one made
    /// before there were tails, which has none.
    pub tails: Option<String>,
    /// 1 to have clients send the basenames that match the encryption key -
    /// see `AdminCreateTargetRequest::send_basenames`.
    pub send_basenames: i64,
    /// How many basenames clients have sent for it - see `ranges::record_basenames`.
    pub basename_count: i64,
}

/// What a dictionary target's ranges (and carving cursor) have as their
/// alphabet's name, with "" as the alphabet itself: their positions are
/// candidate numbers (see `dictionary.rs`), and `candidate_len` is how many
/// words the candidates have. With the empty alphabet, `alphabet::split_pos`
/// and the rest keep a number whole, in the index - see `alphabet::block_size`.
pub const DICTIONARY_ALPHABET_NAME: &str = "dictionary";

impl Target {
    pub fn is_dictionary(&self) -> bool {
        self.kind == "dictionary"
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct TargetProgress {
    pub target_id: i64,
    pub candidate_len: i64,
    pub next_index: i64,
    /// Which alphabet `candidate_len`/`next_index` are denominated in - not
    /// necessarily `targets.alphabet` if the target's alphabet was patched
    /// since carving last touched this cursor. See
    /// `alphabet::transition_alphabet_cursor` and `ranges::claim_range`,
    /// which lazily reconciles the two the next time it carves.
    pub alphabet_name: String,
    pub alphabet: String,
    /// `next_index`'s block - see `alphabet::split_pos`. Always read the
    /// cursor's position through `next()`, which joins the two.
    pub next_block: i64,
}

impl TargetProgress {
    pub fn next(&self) -> Pos {
        join_pos(&self.alphabet, self.candidate_len, self.next_block, self.next_index)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Range {
    pub id: i64,
    pub target_id: i64,
    pub candidate_len: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub status: String,
    pub assigned_user_id: Option<i64>,
    pub last_assigned_user_id: Option<i64>,
    pub assigned_at: Option<i64>,
    pub lease_seconds: Option<i64>,
    pub lease_expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub created_at: i64,
    pub progress_index: Option<i64>,
    /// The alphabet this specific range was carved with - not necessarily
    /// `targets.alphabet`, since a target's alphabet can be patched after
    /// some of its ranges already exist (see `handlers::admin_patch_target`).
    /// Always used in place of the target's own alphabet when decoding this
    /// range's candidates (dashboard rendering, heartbeat progress
    /// resolution, building a `ClaimResponse`).
    pub alphabet_name: String,
    pub alphabet: String,
    /// Which `PriorityRange` (if any) this range was carved from - `None`
    /// for anything produced by the target's own main sweep. Purely for
    /// dashboard labeling; claim/heartbeat/complete/reclaim logic treats
    /// every range the same regardless of origin.
    pub priority_range_id: Option<i64>,
    /// When this range's progress_index was last confirmed to actually
    /// advance - `None` while it isn't currently claimed. See
    /// `ranges::heartbeat_range` and `STALL_RELEASE_SECONDS`.
    pub last_progress_at: Option<i64>,
    /// The blocks of `start_index`/`end_index`/`progress_index` - see
    /// `alphabet::split_pos`. Always read positions through `start()`,
    /// `end()` and `progress()`, which join the two.
    pub start_block: i64,
    pub end_block: i64,
    pub progress_block: i64,
}

impl Range {
    /// A dictionary target's range, of candidate numbers - see
    /// `DICTIONARY_ALPHABET_NAME`.
    pub fn is_dictionary(&self) -> bool {
        self.alphabet_name == DICTIONARY_ALPHABET_NAME
    }

    pub fn start(&self) -> Pos {
        join_pos(&self.alphabet, self.candidate_len, self.start_block, self.start_index)
    }

    pub fn end(&self) -> Pos {
        join_pos(&self.alphabet, self.candidate_len, self.end_block, self.end_index)
    }

    pub fn progress(&self) -> Option<Pos> {
        self.progress_index.map(|index| join_pos(&self.alphabet, self.candidate_len, self.progress_block, index))
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct PriorityRange {
    pub id: i64,
    pub target_id: i64,
    /// Higher claims first, same convention as `Target::priority`. Ties
    /// break by `created_at`. See `ranges::claim_range`.
    pub priority: i64,
    /// The operator-supplied pattern - see `alphabet::pattern_spans`. The
    /// candidates it matches are stored as this row's segments
    /// (`priority_range_segments`).
    pub pattern: String,
    /// `None` for every row created since
    /// `migrations/0021_skip_ranges_and_pattern_segments.sql`, which an
    /// alphabet patch translates by re-expanding `pattern`. Before that, a
    /// pattern became one row per concrete prefix, and this is that prefix -
    /// or `""` for a row from before `0012_priority_range_prefix.sql`, which
    /// can't be translated at all. See
    /// `ranges::migrate_priority_ranges_to_new_alphabet`.
    pub prefix: Option<String>,
    /// A priority range is always scoped to exactly one candidate length.
    pub candidate_len: i64,
    /// What this row owns is its segments' candidates within
    /// `[start_index, end_index)` - `end_index` can be pulled in below the
    /// last segment's end when the range is removed (see
    /// `ranges::remove_priority_range`).
    pub start_index: i64,
    pub end_index: i64,
    /// This row's own cursor within `[start_index, end_index)`, jumping over
    /// the gaps between segments. Once it reaches `end_index` the row is
    /// permanently exhausted - unlike `TargetProgress`, there's no next
    /// length to bump to.
    pub next_index: i64,
    /// Frozen at creation, but not permanently: a later `admin_patch_target`
    /// alphabet change translates this row onto the new alphabet (or
    /// permanently retires it, if nothing it matches can be expressed in
    /// it) - see `ranges::migrate_priority_ranges_to_new_alphabet`.
    pub alphabet_name: String,
    pub alphabet: String,
    pub created_at: i64,
    /// The blocks of `start_index`/`end_index`/`next_index` - see
    /// `alphabet::split_pos`. Always read positions through `start()`,
    /// `end()` and `next()`, which join the two.
    pub start_block: i64,
    pub end_block: i64,
    pub next_block: i64,
    /// Where the prefix of an automatic priority range came from
    /// (`likely_prefixes::Source::as_str`) - `None` for one an operator made.
    /// See `ranges::create_next_auto_priority_range`.
    pub auto_source: Option<String>,
}

impl PriorityRange {
    pub fn start(&self) -> Pos {
        join_pos(&self.alphabet, self.candidate_len, self.start_block, self.start_index)
    }

    pub fn end(&self) -> Pos {
        join_pos(&self.alphabet, self.candidate_len, self.end_block, self.end_index)
    }

    pub fn next(&self) -> Pos {
        join_pos(&self.alphabet, self.candidate_len, self.next_block, self.next_index)
    }
}

/// Candidates at one length an operator never wants handed out - see
/// `migrations/0021_skip_ranges_and_pattern_segments.sql` and
/// `ranges::claim_range`.
#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SkipRange {
    pub id: i64,
    pub target_id: i64,
    /// See `alphabet::pattern_spans`.
    pub pattern: String,
    /// Plain text explaining why, shown on the dashboard.
    pub reason: String,
    pub candidate_len: i64,
    /// What the segments (`skip_range_segments`) are denominated in.
    pub alphabet_name: String,
    pub alphabet: String,
    pub created_at: i64,
    /// Set once removed after it had already skipped something - see
    /// `ranges::remove_skip_range`. A removed skip range has no segments.
    pub removed_at: Option<i64>,
}

/// One stored stretch of a skip or priority range's candidates - see
/// `alphabet::pattern_spans`. `owner_id` is the skip or priority range it
/// belongs to.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Segment {
    pub owner_id: i64,
    pub start_block: i64,
    pub start_index: i64,
    pub end_block: i64,
    pub end_index: i64,
}

impl Segment {
    /// The `[start, end)` positions this segment covers, at `len` in `alphabet`.
    pub fn span(&self, alphabet: &str, len: i64) -> (Pos, Pos) {
        (join_pos(alphabet, len, self.start_block, self.start_index), join_pos(alphabet, len, self.end_block, self.end_index))
    }
}

/// Stores a `u32` hash in an `i64` column without sign issues (always non-negative,
/// well within i64's range).
impl Target {
    /// The text inserted into the target's candidates - see
    /// `AdminCreateTargetRequest::insert_from_start`.
    pub fn insertions(&self) -> crate::alphabet::Insertions<'_> {
        crate::alphabet::Insertions {
            from_start: self.insert_from_start_text.as_deref().map(|text| (text, self.insert_from_start_position)),
            from_end: self.insert_from_end_text.as_deref().map(|text| (text, self.insert_from_end_position)),
        }
    }
}

pub fn u32_to_i64(v: u32) -> i64 {
    v as i64
}

/// Inverse of `u32_to_i64`. Only meaningful for values this server itself wrote via
/// `u32_to_i64`, which is the only way a hash ever enters the `targets` table.
pub fn i64_to_u32(v: i64) -> u32 {
    v as u32
}

pub fn parse_hash_hex(s: &str) -> Result<u32, std::num::ParseIntError> {
    let s = s.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u32::from_str_radix(s, 16)
}
