//! Core range life-cycle: carving fresh work off a target's cursor, reassigning
//! timed-out work, and recording completions. Lives separately from `handlers.rs`
//! so the HTTP glue stays thin.

use namebreak_protocol::ClaimResponse;
use sqlx::SqlitePool;

use crate::alphabet::{candidate_to_index, range_bound_filenames, space_size, strip_prefix_suffix};
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
        range_bound_filenames(&target.alphabet, &target.prefix, &target.suffix, candidate_len, start_index, end_index);
    ClaimResponse {
        range_id,
        target_id: target.id,
        target_name: target.name.clone(),
        prefix: target.prefix.clone(),
        suffix: target.suffix.clone(),
        hash_a_hex: format!("0x{:08X}", i64_to_u32(target.hash_a)),
        hash_b_hex: format!("0x{:08X}", i64_to_u32(target.hash_b)),
        prune_symbol_runs: target.prune_symbol_runs != 0,
        max_backslash_count: target.max_backslash_count,
        lower_bound_filename,
        upper_bound_filename,
        alphabet: target.alphabet.clone(),
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

    // 1) Reassign the oldest reclaimed range, if any - resuming past whatever a
    // previous client's heartbeats already confirmed searched, per progress_index.
    loop {
        let reclaimed = sqlx::query_as::<_, Range>(
            "SELECT ranges.* FROM ranges JOIN targets ON targets.id = ranges.target_id \
             WHERE ranges.status = 'pending' AND targets.status = 'active' \
             ORDER BY ranges.created_at ASC LIMIT 1",
        )
        .fetch_optional(&mut *tx)
        .await?;

        let Some(range) = reclaimed else { break };

        let effective_start = match range.progress_index {
            Some(p) if p + 1 > range.start_index => p + 1,
            _ => range.start_index,
        };

        if effective_start >= range.end_index {
            // A previous client's heartbeats show this range was already fully
            // searched (with no full match ever reported) before it was abandoned -
            // nothing left to hand out. Close it out and look at the next reclaimed
            // range instead of returning a nonsensical empty range to a client.
            sqlx::query("UPDATE ranges SET status = 'completed', completed_at = ?, assigned_user_id = NULL WHERE id = ?")
                .bind(now)
                .bind(range.id)
                .execute(&mut *tx)
                .await?;
            continue;
        }

        let lease_seconds = lease_seconds_for(config, range.end_index - effective_start, rate);
        let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?")
            .bind(range.target_id)
            .fetch_one(&mut *tx)
            .await?;

        let new_range_id = if effective_start > range.start_index {
            // Real progress was made by whoever had this range before (still on
            // `range.last_assigned_user_id`, since that's never cleared). Split the
            // row instead of just narrowing it in place: finalize [start,
            // effective_start) as their completed work, and carve a fresh row for
            // [effective_start, end) to hand to `user` - so each finished portion of
            // a range stays correctly credited to whoever actually searched it,
            // rather than the whole thing ending up attributed to the last claimer.
            sqlx::query(
                "UPDATE ranges SET end_index = ?, status = 'completed', completed_at = ?, \
                 progress_index = NULL, assigned_user_id = NULL WHERE id = ?",
            )
            .bind(effective_start)
            .bind(now)
            .bind(range.id)
            .execute(&mut *tx)
            .await?;

            sqlx::query_scalar(
                "INSERT INTO ranges (target_id, candidate_len, start_index, end_index, status, \
                 assigned_user_id, last_assigned_user_id, assigned_at, lease_seconds, lease_expires_at, created_at) \
                 VALUES (?, ?, ?, ?, 'in_progress', ?, ?, ?, ?, ?, ?) RETURNING id",
            )
            .bind(range.target_id)
            .bind(range.candidate_len)
            .bind(effective_start)
            .bind(range.end_index)
            .bind(user.id)
            .bind(user.id)
            .bind(now)
            .bind(lease_seconds)
            .bind(now + lease_seconds)
            .bind(now)
            .fetch_one(&mut *tx)
            .await?
        } else {
            // No progress was ever checkpointed - reassign the same row as-is.
            sqlx::query(
                "UPDATE ranges SET status = 'in_progress', progress_index = NULL, \
                 assigned_user_id = ?, last_assigned_user_id = ?, assigned_at = ?, lease_seconds = ?, lease_expires_at = ? WHERE id = ?",
            )
            .bind(user.id)
            .bind(user.id)
            .bind(now)
            .bind(lease_seconds)
            .bind(now + lease_seconds)
            .bind(range.id)
            .execute(&mut *tx)
            .await?;
            range.id
        };

        tx.commit().await?;
        return Ok(Some(to_claim_response(&target, new_range_id, range.candidate_len, effective_start, range.end_index, lease_seconds)));
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

        let remaining = space_size(&target.alphabet, progress.candidate_len) - progress.next_index;
        if remaining <= 0 {
            continue; // this target's search space (up to max_len) is fully carved out
        }

        let desired = (rate * config.target_chunk_seconds).round() as i64;
        let chunk = desired.clamp(config.min_chunk_candidates, config.max_chunk_candidates).min(remaining);

        let start_index = progress.next_index;
        let end_index = start_index + chunk;

        let mut new_len = progress.candidate_len;
        let mut new_next_index = end_index;
        if new_next_index >= space_size(&target.alphabet, progress.candidate_len) && new_len < target.max_len {
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
             assigned_user_id, last_assigned_user_id, assigned_at, lease_seconds, lease_expires_at, created_at) \
             VALUES (?, ?, ?, ?, 'in_progress', ?, ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(target.id)
        .bind(progress.candidate_len)
        .bind(start_index)
        .bind(end_index)
        .bind(user.id)
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

pub struct HeartbeatOutcome {
    pub lease_seconds: i64,
    pub target_solved: bool,
}

pub async fn heartbeat_range(
    pool: &SqlitePool,
    user: &User,
    range_id: i64,
    last_hash_a_match_filename: Option<String>,
) -> Result<HeartbeatOutcome, AppError> {
    let mut tx = pool.begin().await?;

    let range = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE id = ?")
        .bind(range_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(AppError::NotFound)?;

    if range.status != "in_progress" || range.assigned_user_id != Some(user.id) {
        return Err(AppError::Conflict("range is not currently assigned to you".into()));
    }

    if let Some(filename) = last_hash_a_match_filename {
        if let Some(new_progress) = resolve_progress_index(&mut tx, &range, &filename).await? {
            // Monotonic: never let a late/out-of-order heartbeat move progress backwards.
            let floor = range.progress_index.unwrap_or(range.start_index - 1);
            let new_progress = new_progress.max(floor);
            sqlx::query("UPDATE ranges SET progress_index = ? WHERE id = ?")
                .bind(new_progress)
                .bind(range_id)
                .execute(&mut *tx)
                .await?;
        }
    }

    let target_status: String = sqlx::query_scalar("SELECT status FROM targets WHERE id = ?")
        .bind(range.target_id)
        .fetch_one(&mut *tx)
        .await?;
    let target_solved = target_status == "solved";

    let lease_seconds = range.lease_seconds.unwrap_or(300);
    let now = now_unix();
    if target_solved {
        // The client is about to abort and won't be reporting completion for
        // this range - close it out now instead of leaving it "in_progress"
        // until its lease eventually times out unclaimed (the target's no
        // longer 'active', so nothing would ever reassign it anyway).
        sqlx::query("UPDATE ranges SET status = 'completed', completed_at = ? WHERE id = ?")
            .bind(now)
            .bind(range_id)
            .execute(&mut *tx)
            .await?;
    } else {
        sqlx::query("UPDATE ranges SET lease_expires_at = ? WHERE id = ?")
            .bind(now + lease_seconds)
            .bind(range_id)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;
    Ok(HeartbeatOutcome { lease_seconds, target_solved })
}

/// Turns a client-reported "Hash A matches: <filename>" line into a validated
/// index within `range`. Returns `None` (rather than an error) for anything that
/// doesn't check out - a malformed or stale progress report shouldn't fail the
/// whole heartbeat, since keeping the lease alive matters far more than this
/// secondary optimization.
async fn resolve_progress_index(
    tx: &mut sqlx::SqliteConnection,
    range: &Range,
    filename: &str,
) -> Result<Option<i64>, AppError> {
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?")
        .bind(range.target_id)
        .fetch_one(&mut *tx)
        .await?;

    let Some(candidate) = strip_prefix_suffix(filename, &target.prefix, &target.suffix) else {
        tracing::warn!(range_id = range.id, filename, "heartbeat match filename doesn't match target's prefix/suffix, ignoring");
        return Ok(None);
    };
    if candidate.chars().count() as i64 != range.candidate_len {
        tracing::warn!(range_id = range.id, filename, "heartbeat match candidate length doesn't match range, ignoring");
        return Ok(None);
    }
    let Some(index) = candidate_to_index(&target.alphabet, candidate) else {
        tracing::warn!(range_id = range.id, filename, "heartbeat match candidate has out-of-alphabet characters, ignoring");
        return Ok(None);
    };
    if index < range.start_index || index >= range.end_index {
        tracing::warn!(range_id = range.id, filename, index, "heartbeat match index falls outside the range, ignoring");
        return Ok(None);
    }
    Ok(Some(index))
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
    use crate::alphabet::{index_to_candidate, range_bound_filenames, space_size};

    const DEFAULT: &str = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
    const SIZE42: &str = " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

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

    async fn insert_user(pool: &SqlitePool, name: &str) -> User {
        let now = now_unix();
        let token = format!("{name}-tok");
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO users (username, hostname, token, created_at, last_seen_at) VALUES (?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(name)
        .bind(format!("{name}-host"))
        .bind(&token)
        .bind(now)
        .bind(now)
        .fetch_one(pool)
        .await
        .unwrap();
        User { id, username: name.into(), hostname: format!("{name}-host"), token, ema_rate_per_sec: None, created_at: now, last_seen_at: now }
    }

    /// Creates a target whose candidate length only ever ranges over `min_len..=max_len`,
    /// so its whole space is small enough to carve (and exhaust) in a couple of test claims.
    async fn insert_target(pool: &SqlitePool, min_len: i64, max_len: i64) -> i64 {
        insert_target_with_alphabet(pool, "size49", DEFAULT, min_len, max_len).await
    }

    async fn insert_target_with_alphabet(pool: &SqlitePool, alphabet_name: &str, alphabet: &str, min_len: i64, max_len: i64) -> i64 {
        let now = now_unix();
        let target_id: i64 = sqlx::query_scalar(
            "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, min_len, max_len, prune_symbol_runs, alphabet_name, alphabet, status, created_at) \
             VALUES ('t', 'PRE', '.SUF', 0, 0, ?, ?, 0, ?, ?, 'active', ?) RETURNING id",
        )
        .bind(min_len)
        .bind(max_len)
        .bind(alphabet_name)
        .bind(alphabet)
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
        let user = insert_user(&pool, "tester").await;
        insert_target(&pool, 1, 2).await;

        // Bigger than either length's full space, so each claim greedily takes all
        // of what's left at the current length.
        let config = test_config(space_size(DEFAULT, 2));

        let claim1 = claim_range(&pool, &config, &user).await.unwrap().expect("length 1 should still have work");
        let (exp_lower1, exp_upper1) = range_bound_filenames(DEFAULT, "PRE", ".SUF", 1, 0, space_size(DEFAULT, 1));
        assert_eq!(claim1.lower_bound_filename, exp_lower1);
        assert_eq!(claim1.upper_bound_filename, exp_upper1);
        assert_eq!(claim1.candidate_count, space_size(DEFAULT, 1), "first range should cover the whole (and only the) length-1 space");

        let claim2 = claim_range(&pool, &config, &user).await.unwrap().expect("length 2 should now be available");
        let (exp_lower2, exp_upper2) = range_bound_filenames(DEFAULT, "PRE", ".SUF", 2, 0, space_size(DEFAULT, 2));
        assert_eq!(claim2.lower_bound_filename, exp_lower2, "length-2 work must start at index 0, not skip ahead");
        assert_eq!(claim2.upper_bound_filename, exp_upper2);
        assert_eq!(claim2.candidate_count, space_size(DEFAULT, 2), "second range should cover the whole (and only the) length-2 space");

        let claim3 = claim_range(&pool, &config, &user).await.unwrap();
        assert!(claim3.is_none(), "max_len's space is now fully carved - there must be no length-3 work invented");
    }

    /// The scenario this whole feature exists for: a client heartbeats a partial
    /// (Hash A only) match partway through its range, then disconnects. Once the
    /// server reclaims the abandoned range, whoever claims it next must resume
    /// just past the checkpointed candidate - not redo the whole range from its
    /// original start. And since the first user genuinely finished their portion,
    /// it must stay credited to them as a separate completed range rather than
    /// silently becoming part of whatever the second user ends up owning.
    #[tokio::test]
    async fn heartbeat_progress_splits_the_range_crediting_each_user_with_their_part() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        insert_target(&pool, 3, 3).await; // fixed length, whole space handed out as one range

        let config = test_config(space_size(DEFAULT, 3));
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");
        assert_eq!(claim.candidate_count, space_size(DEFAULT, 3));

        let midpoint_index = space_size(DEFAULT, 3) / 2;
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, midpoint_index, 3));
        heartbeat_range(&pool, &first_user, claim.range_id, Some(midpoint_filename)).await.unwrap();

        // Simulate the client disconnecting: force its lease into the past and run
        // the same sweep the background task runs.
        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?")
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        let second_user = insert_user(&pool, "second").await;
        let resumed = claim_range(&pool, &config, &second_user).await.unwrap().expect("range should be reassignable");
        assert_ne!(resumed.range_id, claim.range_id, "the finished first part and the remaining second part must be distinct rows");

        let expected_start = midpoint_index + 1;
        let (exp_lower, exp_upper) = range_bound_filenames(DEFAULT, "PRE", ".SUF", 3, expected_start, space_size(DEFAULT, 3));
        assert_eq!(resumed.lower_bound_filename, exp_lower, "must resume just past the checkpointed candidate, not from the original start");
        assert_eq!(resumed.upper_bound_filename, exp_upper);
        assert_eq!(resumed.candidate_count, space_size(DEFAULT, 3) - expected_start);

        // The original row: shrunk to exactly the searched portion, completed, and
        // still credited to the first user - not overwritten by the reassignment.
        let (orig_status, orig_start, orig_end, orig_worker): (String, i64, i64, Option<i64>) = sqlx::query_as(
            "SELECT status, start_index, end_index, last_assigned_user_id FROM ranges WHERE id = ?",
        )
        .bind(claim.range_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(orig_status, "completed");
        assert_eq!(orig_start, 0);
        assert_eq!(orig_end, expected_start);
        assert_eq!(orig_worker, Some(first_user.id));

        // The new row: the remainder, credited to the second user.
        let (new_start, new_end, new_worker): (i64, i64, Option<i64>) =
            sqlx::query_as("SELECT start_index, end_index, last_assigned_user_id FROM ranges WHERE id = ?")
                .bind(resumed.range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(new_start, expected_start);
        assert_eq!(new_end, space_size(DEFAULT, 3));
        assert_eq!(new_worker, Some(second_user.id));
    }

    /// If a client's last heartbeat before disconnecting already covered the very
    /// end of its range, the range has in fact been fully searched (with no full
    /// match reported) - there's nothing left to reassign, so it should be closed
    /// out as completed instead of handed to the next claimer as a zero-width range.
    #[tokio::test]
    async fn heartbeat_progress_reaching_the_end_completes_the_range_without_reassigning() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        insert_target(&pool, 2, 2).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");

        let last_index = space_size(DEFAULT, 2) - 1;
        let last_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, last_index, 2));
        heartbeat_range(&pool, &first_user, claim.range_id, Some(last_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?")
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        let second_user = insert_user(&pool, "second").await;
        let claim2 = claim_range(&pool, &config, &second_user).await.unwrap();
        assert!(claim2.is_none(), "range was already fully searched via heartbeats - nothing should be handed out");

        let status: String = sqlx::query_scalar("SELECT status FROM ranges WHERE id = ?")
            .bind(claim.range_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "completed");
    }

    /// Proves the alphabet parameterization actually works end-to-end for a
    /// non-default alphabet, not just for the one everything else in this file
    /// happens to use: a target using size42 should get bounds and a candidate
    /// count computed against a 42-character space, not the default 49.
    #[tokio::test]
    async fn claim_uses_the_targets_own_alphabet_not_the_default() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        insert_target_with_alphabet(&pool, "size42", SIZE42, 3, 3).await;

        let config = test_config(space_size(SIZE42, 3));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        assert_eq!(claim.alphabet, SIZE42);
        assert_eq!(claim.candidate_count, space_size(SIZE42, 3), "should be size42's space (74088), not size49's");
        assert_ne!(claim.candidate_count, space_size(DEFAULT, 3));
        let (exp_lower, exp_upper) = range_bound_filenames(SIZE42, "PRE", ".SUF", 3, 0, space_size(SIZE42, 3));
        assert_eq!(claim.lower_bound_filename, exp_lower);
        assert_eq!(claim.upper_bound_filename, exp_upper);
    }

    /// A target's max_backslash_count must reach the client via ClaimResponse
    /// unchanged, since namebreak itself (not the server) is what enforces it.
    #[tokio::test]
    async fn claim_includes_the_targets_max_backslash_count() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let target_id = insert_target(&pool, 2, 2).await;
        sqlx::query("UPDATE targets SET max_backslash_count = ? WHERE id = ?")
            .bind(3i64)
            .bind(target_id)
            .execute(&pool)
            .await
            .unwrap();

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert_eq!(claim.max_backslash_count, 3);
    }

    /// Once a target is solved (via one range's completion), any other client
    /// still heartbeating a different in-progress range for the same target must
    /// be told to abort - and that range should be closed out immediately rather
    /// than left "in_progress" until its lease eventually times out unclaimed.
    #[tokio::test]
    async fn heartbeat_signals_abort_once_the_target_is_solved_elsewhere() {
        let pool = test_pool().await;
        let finder = insert_user(&pool, "finder").await;
        let other = insert_user(&pool, "other").await;
        insert_target(&pool, 3, 3).await;

        // Small enough that the target's space gets carved into (at least) two ranges.
        let config = test_config(space_size(DEFAULT, 3) / 2);

        let claim_a = claim_range(&pool, &config, &finder).await.unwrap().expect("first range available");
        let claim_b = claim_range(&pool, &config, &other).await.unwrap().expect("second range available");
        assert_ne!(claim_a.range_id, claim_b.range_id);

        // Before anything is found: heartbeat behaves normally.
        let before = heartbeat_range(&pool, &other, claim_b.range_id, None).await.unwrap();
        assert!(!before.target_solved);

        // `finder` reports a match, solving the target.
        let outcome = complete_range(&pool, &config, &finder, claim_a.range_id, true, Some("PRE???.SUF".into()), 1.0, 1)
            .await
            .unwrap();
        assert!(outcome.target_solved);

        // `other`'s next heartbeat must now signal abort, and its range should be
        // closed out rather than left dangling.
        let after = heartbeat_range(&pool, &other, claim_b.range_id, None).await.unwrap();
        assert!(after.target_solved);

        let status: String = sqlx::query_scalar("SELECT status FROM ranges WHERE id = ?")
            .bind(claim_b.range_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "completed");
    }
}
