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
    /// hands this client a target using an alphabet introduced after the
    /// version it declared - see `alphabet::alphabet_available_to`.
    pub protocol_version: String,
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
    pub max_backslash_count: i64,
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
    /// See `alphabet::compile_skip_regex`/`find_skip_run` - checked only when
    /// carving fresh ranges, never against already-carved ones.
    pub skip_regex: Option<String>,
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
}

#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct PriorityRange {
    pub id: i64,
    pub target_id: i64,
    /// Higher claims first, same convention as `Target::priority`. Ties
    /// break by `created_at`. See `ranges::claim_range`.
    pub priority: i64,
    /// The operator-supplied pattern this row's prefix was expanded from -
    /// see `alphabet::expand_priority_pattern`. Display only.
    pub pattern: String,
    /// The one concrete literal prefix (out of `pattern`'s possibly several
    /// expansions) this specific row covers - `None` only for a row that
    /// predates this column (see `migrations/0012_priority_range_prefix.sql`).
    /// Needed to translate `start_index`/`end_index`/`next_index` onto a new
    /// alphabet if the target's is ever patched - see
    /// `ranges::migrate_priority_ranges_to_new_alphabet`.
    pub prefix: Option<String>,
    /// A priority range is always scoped to exactly one candidate length -
    /// see `alphabet::expand_priority_pattern`'s doc comment for why.
    pub candidate_len: i64,
    pub start_index: i64,
    pub end_index: i64,
    /// This row's own cursor within `[start_index, end_index)`. Once it
    /// reaches `end_index` the row is permanently exhausted - unlike
    /// `TargetProgress`, there's no next length to bump to.
    pub next_index: i64,
    /// Frozen at creation, but not permanently: a later `admin_patch_target`
    /// alphabet change translates `start_index`/`end_index`/`next_index`
    /// onto the new alphabet (or permanently retires this row, if `prefix`
    /// can no longer be expressed in it) - see
    /// `ranges::migrate_priority_ranges_to_new_alphabet`.
    pub alphabet_name: String,
    pub alphabet: String,
    pub created_at: i64,
}

/// Stores a `u32` hash in an `i64` column without sign issues (always non-negative,
/// well within i64's range).
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
