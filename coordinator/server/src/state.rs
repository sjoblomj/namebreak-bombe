use sqlx::SqlitePool;

use crate::likely_prefixes::LikelyPrefixes;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState(pub Arc<Inner>);

pub struct Inner {
    pub pool: SqlitePool,
    pub admin_token: String,
    pub config: RangeConfig,
}

impl std::ops::Deref for AppState {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.0
    }
}

/// Tunables for how ranges are sized and leased. All overridable via env vars
/// (see `Config::from_env`) - defaults are rough guesses for a modern GPU running
/// the trivial MPQ hash, self-correcting after each user's first completed range
/// updates their `ema_rate_per_sec`.
pub struct RangeConfig {
    pub target_chunk_seconds: f64,
    pub default_rate_per_sec: f64,
    pub min_chunk_candidates: i64,
    pub max_chunk_candidates: i64,
    /// How long a claimed range stays leased after the last sign of life
    /// from its client - the claim itself, then every heartbeat - before
    /// reclaim_expired releases it. Deliberately the same for every range,
    /// whatever its size or its claimant's speed: heartbeats (every 60s)
    /// are what keep a live client's lease going, so this only decides how
    /// long a quit, crashed or disconnected client's range sits unworked.
    pub lease_seconds: i64,
    pub reclaim_interval_secs: u64,
    pub ema_alpha: f64,
    /// How long a range may sit claimed with no real progress (its
    /// progress_index never advancing between heartbeats) before
    /// heartbeat_range releases it back to pending for someone else -
    /// heartbeating alone keeps lease_expires_at renewed indefinitely, so
    /// without this a stalled (e.g. paused) client would hold a range
    /// hostage forever. See `ranges::heartbeat_range`.
    pub stall_release_seconds: i64,
    /// The chance that a claim is handed a canary (see `canary.rs`) instead
    /// of real work. 0 hands out none.
    pub canary_probability: f64,
    /// How long a canary should take the client that's handed it, at its
    /// measured rate - a few seconds, well under the 10 it's meant to stay
    /// within.
    pub canary_seconds: f64,
    /// What automatic priority ranges are made from - see
    /// `likely_prefixes.rs`. Counted once, when the server starts.
    pub likely_prefixes: Arc<LikelyPrefixes>,
}

impl RangeConfig {
    pub fn from_env() -> Self {
        fn env_f64(key: &str, default: f64) -> f64 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_i64(key: &str, default: i64) -> i64 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_u64(key: &str, default: u64) -> u64 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }

        RangeConfig {
            target_chunk_seconds: env_f64("TARGET_CHUNK_SECONDS", 900.0),
            default_rate_per_sec: env_f64("DEFAULT_RATE_PER_SEC", 500_000_000.0),
            min_chunk_candidates: env_i64("MIN_CHUNK_CANDIDATES", 1_000_000),
            max_chunk_candidates: env_i64("MAX_CHUNK_CANDIDATES", 1_000_000_000_000_000),
            lease_seconds: env_i64("LEASE_SECONDS", 6 * 60 * 60),
            reclaim_interval_secs: env_u64("RECLAIM_INTERVAL_SECS", 30),
            ema_alpha: env_f64("EMA_ALPHA", 0.3),
            stall_release_seconds: env_i64("STALL_RELEASE_SECONDS", 24 * 60 * 60),
            canary_probability: env_f64("CANARY_PROBABILITY", 0.33),
            canary_seconds: env_f64("CANARY_SECONDS", 5.0),
            likely_prefixes: Arc::new(LikelyPrefixes::load()),
        }
    }
}

pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs() as i64
}

/// A 256-bit random token, hex-encoded. Not a password (none is asked of users) -
/// just an opaque per-client identity that (a) lets the server attribute/revoke
/// work per client and (b) keeps generic endpoint-scraping bots out, since it's
/// only handed out via /register and required on every other endpoint.
pub fn generate_token() -> String {
    use rand::Rng;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
