use sqlx::SqlitePool;
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
    pub lease_grace_multiplier: f64,
    pub reclaim_interval_secs: u64,
    pub ema_alpha: f64,
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
            max_chunk_candidates: env_i64("MAX_CHUNK_CANDIDATES", 200_000_000_000),
            lease_grace_multiplier: env_f64("LEASE_GRACE_MULTIPLIER", 3.0),
            reclaim_interval_secs: env_u64("RECLAIM_INTERVAL_SECS", 30),
            ema_alpha: env_f64("EMA_ALPHA", 0.3),
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
