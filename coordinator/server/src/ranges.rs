//! Core range life-cycle: carving fresh work off a target's cursor, reassigning
//! timed-out work, and recording completions. Lives separately from `handlers.rs`
//! so the HTTP glue stays thin.

use namebreak_protocol::ClaimResponse;
use sqlx::SqlitePool;

use crate::alphabet::{range_bound_filenames, space_size};
use crate::error::AppError;
use crate::models::{i64_to_u32, Range, Target, TargetProgress, User};
use crate::state::{now_unix, RangeConfig};

fn lease_seconds_for(config: &RangeConfig, range_size: i64, effective_rate: f64) -> i64 {
    let expected = range_size as f64 / effective_rate.max(1.0);
    let leased = (expected * config.lease_grace_multiplier).round() as i64;
    leased.max(60)
}

fn effective_rate(config: &RangeConfig, user: &User) -> f64 {
    user.ema_rate_per_sec.unwrap_or(config.default_rate_per_sec)
}

fn to_claim_response(target: &Target, range_id: i64, candidate_len: i64, start_index: i64, end_index: i64, lease_seconds: i64) -> ClaimResponse {
    let (lower_bound_filename, upper_bound_filename) =
        range_bound_filenames(&target.prefix, &target.suffix, candidate_len, start_index, end_index);
    ClaimResponse {
        range_id,
        target_id: target.id,
        target_name: target.name.clone(),
        prefix: target.prefix.clone(),
        suffix: target.suffix.clone(),
        hash_a_hex: format!("0x{:08X}", i64_to_u32(target.hash_a)),
        hash_b_hex: format!("0x{:08X}", i64_to_u32(target.hash_b)),
        prune_symbol_runs: target.prune_symbol_runs != 0,
        lower_bound_filename,
        upper_bound_filename,
        candidate_count: end_index - start_index,
        lease_seconds,
    }
}

/// Tries to hand `user` a unit of work: first a previously reclaimed range, else a
/// freshly carved slice of the highest-priority active target with room left.
/// Returns `None` when there is nothing available at all.
pub async fn claim_range(pool: &SqlitePool, config: &RangeConfig, user: &User) -> Result<Option<ClaimResponse>, AppError> {
    let mut tx = pool.begin().await?;
    let now = now_unix();
    let rate = effective_rate(config, user);

    // 1) Reassign the oldest reclaimed range, if any.
    let reclaimed = sqlx::query_as::<_, Range>(
        "SELECT ranges.* FROM ranges JOIN targets ON targets.id = ranges.target_id \
         WHERE ranges.status = 'pending' AND targets.status = 'active' \
         ORDER BY ranges.created_at ASC LIMIT 1",
    )
    .fetch_optional(&mut *tx)
    .await?;

    if let Some(range) = reclaimed {
        let lease_seconds = lease_seconds_for(config, range.end_index - range.start_index, rate);
        sqlx::query(
            "UPDATE ranges SET status = 'in_progress', assigned_user_id = ?, assigned_at = ?, \
             lease_seconds = ?, lease_expires_at = ? WHERE id = ?",
        )
        .bind(user.id)
        .bind(now)
        .bind(lease_seconds)
        .bind(now + lease_seconds)
        .bind(range.id)
        .execute(&mut *tx)
        .await?;

        let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?")
            .bind(range.target_id)
            .fetch_one(&mut *tx)
            .await?;
        tx.commit().await?;
        return Ok(Some(to_claim_response(&target, range.id, range.candidate_len, range.start_index, range.end_index, lease_seconds)));
    }

    // 2) Otherwise carve a fresh chunk off the oldest active target that still has room.
    let targets = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE status = 'active' ORDER BY created_at ASC")
        .fetch_all(&mut *tx)
        .await?;

    for target in targets {
        let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
            .bind(target.id)
            .fetch_one(&mut *tx)
            .await?;

        let remaining = space_size(progress.candidate_len) - progress.next_index;
        if remaining <= 0 {
            continue; // this target's search space (up to max_len) is fully carved out
        }

        let desired = (rate * config.target_chunk_seconds).round() as i64;
        let chunk = desired.clamp(config.min_chunk_candidates, config.max_chunk_candidates).min(remaining);

        let start_index = progress.next_index;
        let end_index = start_index + chunk;

        let mut new_len = progress.candidate_len;
        let mut new_next_index = end_index;
        if new_next_index >= space_size(progress.candidate_len) && new_len < target.max_len {
            new_len += 1;
            new_next_index = 0;
        }
        sqlx::query("UPDATE target_progress SET candidate_len = ?, next_index = ? WHERE target_id = ?")
            .bind(new_len)
            .bind(new_next_index)
            .bind(target.id)
            .execute(&mut *tx)
            .await?;

        let lease_seconds = lease_seconds_for(config, chunk, rate);
        let range_id: i64 = sqlx::query_scalar(
            "INSERT INTO ranges (target_id, candidate_len, start_index, end_index, status, \
             assigned_user_id, assigned_at, lease_seconds, lease_expires_at, created_at) \
             VALUES (?, ?, ?, ?, 'in_progress', ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(target.id)
        .bind(progress.candidate_len)
        .bind(start_index)
        .bind(end_index)
        .bind(user.id)
        .bind(now)
        .bind(lease_seconds)
        .bind(now + lease_seconds)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;

        tx.commit().await?;
        return Ok(Some(to_claim_response(&target, range_id, progress.candidate_len, start_index, end_index, lease_seconds)));
    }

    tx.commit().await?;
    Ok(None)
}

pub async fn heartbeat_range(pool: &SqlitePool, user: &User, range_id: i64) -> Result<i64, AppError> {
    let range = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE id = ?")
        .bind(range_id)
        .fetch_optional(pool)
        .await?
        .ok_or(AppError::NotFound)?;

    if range.status != "in_progress" || range.assigned_user_id != Some(user.id) {
        return Err(AppError::Conflict("range is not currently assigned to you".into()));
    }
    let lease_seconds = range.lease_seconds.unwrap_or(300);
    let now = now_unix();
    sqlx::query("UPDATE ranges SET lease_expires_at = ? WHERE id = ?")
        .bind(now + lease_seconds)
        .bind(range_id)
        .execute(pool)
        .await?;
    Ok(lease_seconds)
}

pub struct CompleteOutcome {
    pub target_solved: bool,
}

pub async fn complete_range(
    pool: &SqlitePool,
    config: &RangeConfig,
    user: &User,
    range_id: i64,
    found: bool,
    filename: Option<String>,
    elapsed_seconds: f64,
    candidates_processed: i64,
) -> Result<CompleteOutcome, AppError> {
    let mut tx = pool.begin().await?;

    let range = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE id = ?")
        .bind(range_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(AppError::NotFound)?;

    if range.status != "in_progress" || range.assigned_user_id != Some(user.id) {
        return Err(AppError::Conflict("range is not currently assigned to you".into()));
    }

    let now = now_unix();
    sqlx::query("UPDATE ranges SET status = 'completed', completed_at = ? WHERE id = ?")
        .bind(now)
        .bind(range_id)
        .execute(&mut *tx)
        .await?;

    if elapsed_seconds > 0.001 && candidates_processed > 0 {
        let observed_rate = candidates_processed as f64 / elapsed_seconds;
        let new_ema = match user.ema_rate_per_sec {
            Some(old) => config.ema_alpha * observed_rate + (1.0 - config.ema_alpha) * old,
            None => observed_rate,
        };
        sqlx::query("UPDATE users SET ema_rate_per_sec = ? WHERE id = ?")
            .bind(new_ema)
            .bind(user.id)
            .execute(&mut *tx)
            .await?;
    }

    let mut target_solved = false;
    if found {
        let result = sqlx::query(
            "UPDATE targets SET status = 'solved', found_filename = ?, found_by_user_id = ? \
             WHERE id = ? AND status = 'active'",
        )
        .bind(&filename)
        .bind(user.id)
        .bind(range.target_id)
        .execute(&mut *tx)
        .await?;
        target_solved = result.rows_affected() > 0;
    }

    tx.commit().await?;
    Ok(CompleteOutcome { target_solved })
}

/// Reassigns any range whose lease expired while still `in_progress` back to the
/// pending pool, so the next `/claim` can hand it to a different client.
pub async fn reclaim_expired(pool: &SqlitePool) -> Result<u64, sqlx::Error> {
    let now = now_unix();
    let result = sqlx::query(
        "UPDATE ranges SET status = 'pending', assigned_user_id = NULL, assigned_at = NULL, \
         lease_expires_at = NULL WHERE status = 'in_progress' AND lease_expires_at < ?",
    )
    .bind(now)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{range_bound_filenames, space_size};

    async fn test_pool() -> SqlitePool {
        // In-memory DB behind a single connection (same pattern as db::connect) so
        // it survives across queries for the life of the pool.
        crate::db::connect("sqlite::memory:").await.expect("connect to in-memory sqlite")
    }

    fn test_config(chunk_at_least: i64) -> RangeConfig {
        RangeConfig {
            target_chunk_seconds: 1.0,
            default_rate_per_sec: 1.0,
            // Deliberately larger than either test length's full space, so every
            // claim greedily takes "the rest of the current length" in one chunk -
            // that's what forces a length boundary to actually be crossed between
            // claims instead of taking many small same-length chunks.
            min_chunk_candidates: chunk_at_least,
            max_chunk_candidates: chunk_at_least,
            lease_grace_multiplier: 3.0,
            reclaim_interval_secs: 30,
            ema_alpha: 0.3,
        }
    }

    async fn insert_user(pool: &SqlitePool) -> User {
        let now = now_unix();
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO users (username, hostname, token, created_at, last_seen_at) VALUES (?, ?, ?, ?, ?) RETURNING id",
        )
        .bind("tester")
        .bind("test-host")
        .bind("tok")
        .bind(now)
        .bind(now)
        .fetch_one(pool)
        .await
        .unwrap();
        User { id, username: "tester".into(), hostname: "test-host".into(), token: "tok".into(), ema_rate_per_sec: None, created_at: now, last_seen_at: now }
    }

    /// Creates a target whose candidate length only ever ranges over `min_len..=max_len`,
    /// so its whole space is small enough to carve (and exhaust) in a couple of test claims.
    async fn insert_target(pool: &SqlitePool, min_len: i64, max_len: i64) -> i64 {
        let now = now_unix();
        let target_id: i64 = sqlx::query_scalar(
            "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, min_len, max_len, prune_symbol_runs, status, created_at) \
             VALUES ('t', 'PRE', '.SUF', 0, 0, ?, ?, 0, 'active', ?) RETURNING id",
        )
        .bind(min_len)
        .bind(max_len)
        .bind(now)
        .fetch_one(pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_index) VALUES (?, ?, 0)")
            .bind(target_id)
            .bind(min_len)
            .execute(pool)
            .await
            .unwrap();
        target_id
    }

    /// A target's candidate length can only grow one length at a time (min_len=1,
    /// max_len=2 here), and `namebreak bounded` itself only ever searches a single
    /// fixed length per invocation - so a correct range can never straddle two
    /// lengths. This exercises exactly that boundary: the first claim must exactly
    /// exhaust length 1's whole space (no leftover, nothing skipped), the second
    /// must pick up length 2 starting exactly at index 0 (no gap), and once length
    /// 2's space (== max_len) is exhausted too, claiming must stop entirely rather
    /// than inventing a length 3.
    #[tokio::test]
    async fn range_carving_crosses_a_candidate_length_boundary_cleanly() {
        let pool = test_pool().await;
        let user = insert_user(&pool).await;
        insert_target(&pool, 1, 2).await;

        // Bigger than either length's full space, so each claim greedily takes all
        // of what's left at the current length.
        let config = test_config(space_size(2));

        let claim1 = claim_range(&pool, &config, &user).await.unwrap().expect("length 1 should still have work");
        let (exp_lower1, exp_upper1) = range_bound_filenames("PRE", ".SUF", 1, 0, space_size(1));
        assert_eq!(claim1.lower_bound_filename, exp_lower1);
        assert_eq!(claim1.upper_bound_filename, exp_upper1);
        assert_eq!(claim1.candidate_count, space_size(1), "first range should cover the whole (and only the) length-1 space");

        let claim2 = claim_range(&pool, &config, &user).await.unwrap().expect("length 2 should now be available");
        let (exp_lower2, exp_upper2) = range_bound_filenames("PRE", ".SUF", 2, 0, space_size(2));
        assert_eq!(claim2.lower_bound_filename, exp_lower2, "length-2 work must start at index 0, not skip ahead");
        assert_eq!(claim2.upper_bound_filename, exp_upper2);
        assert_eq!(claim2.candidate_count, space_size(2), "second range should cover the whole (and only the) length-2 space");

        let claim3 = claim_range(&pool, &config, &user).await.unwrap();
        assert!(claim3.is_none(), "max_len's space is now fully carved - there must be no length-3 work invented");
    }
}
