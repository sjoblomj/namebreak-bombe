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
    pub min_len: i64,
    pub max_len: i64,
    pub prune_symbol_runs: i64,
    pub max_backslash_count: i64,
    pub alphabet_name: String,
    pub alphabet: String,
    pub status: String,
    pub found_filename: Option<String>,
    pub found_by_user_id: Option<i64>,
    pub created_at: i64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct TargetProgress {
    pub target_id: i64,
    pub candidate_len: i64,
    pub next_index: i64,
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
    pub assigned_at: Option<i64>,
    pub lease_seconds: Option<i64>,
    pub lease_expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub created_at: i64,
    pub progress_index: Option<i64>,
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
