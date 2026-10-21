//! False matches: names that match both of a target's hashes but aren't its
//! file's name. Two 32-bit hashes match by chance about once in 2^64 names,
//! and a long search hashes that many - REZ\SFZSSP050WSRL.WAV matched a
//! file's Hash A and Hash B and was nothing like its name. What else is known
//! about the file tells them apart: its encryption key (hash type 3 of its
//! basename) and where its entry is in the archive's hash table (see
//! `namebreak_protocol::HashTablePosition`).
//!
//! A client of protocol 1.7 or later is told both with its claim
//! (`add_match_checks`), checks every match of both hashes against them,
//! searches on past one that fails and reports it
//! (`namebreak_protocol::HeartbeatRequest::false_matches`), which
//! `record_reported` keeps. An older client is told neither and takes the
//! first match of both hashes for the file - so the server checks every find
//! itself (`false_match_reason`, see `ranges::complete_range`), whoever
//! reports it.

use namebreak_protocol::{ClaimResponse, HashTablePosition, Version, FALSE_MATCHES_SINCE, MAX_FALSE_MATCHES_PER_REPORT};
use sqlx::SqlitePool;

use crate::error::AppError;
use crate::models::{i64_to_u32, Target, User};
use crate::mpq::{mpq_hash, FILE_KEY_OFFSET, HASH_A_OFFSET, HASH_B_OFFSET, TABLE_INDEX_OFFSET};
use crate::state::now_unix;

/// Why `filename`, which matches both of `target`'s hashes, isn't its file's
/// name after all - `None` if it passes everything known about the file.
/// The client's `falseMatchReason` (client/src/engine/match_check.h), whose
/// wording it shares.
pub fn false_match_reason(target: &Target, filename: &str) -> Option<String> {
    let mut reasons = Vec::new();
    if let Some(key) = target.encryption_key.map(i64_to_u32) {
        let basename = filename.rsplit('\\').next().unwrap_or(filename);
        let actual = mpq_hash(basename, FILE_KEY_OFFSET);
        if actual != key {
            reasons.push(format!("its basename's key is 0x{actual:08X}, not the file's 0x{key:08X}"));
        }
    }
    if let Some(position) = target.hash_table_position() {
        let slot = start_slot(filename, position.table_size);
        if !reaches_entry(&position, slot) {
            reasons.push(format!(
                "its lookup starts at hash table slot {slot}, not at one from {} to {} of {}",
                position.first_start_slot, position.last_start_slot, position.table_size
            ));
        }
    }
    (!reasons.is_empty()).then(|| reasons.join(", and "))
}

/// The slot Storm's lookup of `filename` starts at, in a hash table of
/// `table_size` slots (a power of two).
pub fn start_slot(filename: &str, table_size: i64) -> i64 {
    (mpq_hash(filename, TABLE_INDEX_OFFSET) & (table_size as u32).wrapping_sub(1)) as i64
}

/// Whether a lookup starting at `slot` reaches the entry - see
/// `HashTablePosition`.
fn reaches_entry(position: &HashTablePosition, slot: i64) -> bool {
    let (first, last) = (position.first_start_slot, position.last_start_slot);
    if first <= last {
        (first..=last).contains(&slot)
    } else {
        slot >= first || slot <= last
    }
}

/// Rejects a `HashTablePosition` no archive could have.
pub fn validate_hash_table_position(position: &HashTablePosition) -> Result<(), AppError> {
    let size = position.table_size;
    if !(1..=1 << 31).contains(&size) || size & (size - 1) != 0 {
        return Err(AppError::BadRequest("hash_table_position's table_size must be a power of two, up to 2^31, as an archive's hash table is".into()));
    }
    for (what, slot) in [("first_start_slot", position.first_start_slot), ("last_start_slot", position.last_start_slot)] {
        if slot < 0 || slot >= size {
            return Err(AppError::BadRequest(format!("hash_table_position's {what} must be a slot of the table, 0 to {}", size - 1)));
        }
    }
    Ok(())
}

/// Puts what's known about `target`'s file beyond its hashes into `claim`,
/// for a client that checks a match of both hashes against it - one of
/// protocol 1.7 or later. An older one gets the claim as before: a
/// dictionary target's key, which it searches by from 1.5 on, and nothing
/// else.
pub fn add_match_checks(claim: &mut ClaimResponse, target: &Target, client_version: Version) {
    if client_version < FALSE_MATCHES_SINCE {
        return;
    }
    claim.encryption_key_hex = target.encryption_key.map(|key| format!("0x{:08X}", i64_to_u32(key)));
    if let Some(position) = target.hash_table_position() {
        claim.hash_table_size = Some(position.table_size);
        claim.hash_table_first_start_slot = Some(position.first_start_slot);
        claim.hash_table_last_start_slot = Some(position.last_start_slot);
    }
}

/// Whether `filename` matches both of `target`'s hashes - as everything a
/// client calls a match (a false one included) must.
pub fn matches_hashes(target: &Target, filename: &str) -> bool {
    mpq_hash(filename, HASH_A_OFFSET) == i64_to_u32(target.hash_a) && mpq_hash(filename, HASH_B_OFFSET) == i64_to_u32(target.hash_b)
}

/// Keeps a false match of `target_id`'s, once: a second report of the same
/// filename changes nothing. Returns whether it's new.
pub async fn store(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    filename: &str,
    reason: &str,
    user_id: i64,
    range_id: Option<i64>,
    now: i64,
) -> Result<bool, AppError> {
    let result = sqlx::query(
        "INSERT OR IGNORE INTO false_matches (target_id, filename, reason, user_id, range_id, reported_at) VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(target_id)
    .bind(filename)
    .bind(reason)
    .bind(user_id)
    .bind(range_id)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Keeps the false matches a client reported with a heartbeat, quit or
/// completion of `range_id` (see `HeartbeatRequest::false_matches`), for the
/// range's target - before anything else the report does, and committed on
/// its own, as `ranges::record_basenames` does the basenames, so that they're
/// kept whatever the rest of it gets. From whoever reports them: they're
/// facts about the target. Each is checked here too: one that doesn't match
/// both hashes isn't one at all, and is left out; and one that passes what's
/// known about the file *now* - the target's key or hash table position was
/// changed, or removed, after the range was claimed - is the file's name,
/// and solves the target.
pub async fn record_reported(pool: &SqlitePool, user: &User, range_id: i64, filenames: &[String]) -> Result<(), AppError> {
    if filenames.is_empty() {
        return Ok(());
    }
    if filenames.len() > MAX_FALSE_MATCHES_PER_REPORT {
        return Err(AppError::BadRequest(format!("at most {MAX_FALSE_MATCHES_PER_REPORT} false matches a report")));
    }
    let mut tx = pool.begin().await?;
    let target_id: i64 = sqlx::query_scalar("SELECT target_id FROM ranges WHERE id = ?")
        .bind(range_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(AppError::NotFound)?;
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(&mut *tx).await?;
    // A canary's (see canary.rs): nothing to keep.
    if target.is_virtual != 0 {
        return Ok(());
    }
    let now = now_unix();
    for filename in filenames {
        if !matches_hashes(&target, filename) {
            tracing::warn!(range_id, user_id = user.id, filename, "a reported false match doesn't match the target's hashes, ignoring");
            continue;
        }
        match false_match_reason(&target, filename) {
            Some(reason) => {
                if store(&mut tx, target.id, filename, &reason, user.id, Some(range_id), now).await? {
                    tracing::info!(range_id, user_id = user.id, target_id = target.id, filename, reason, "false match reported");
                }
            }
            None => {
                tracing::info!(range_id, user_id = user.id, target_id = target.id, filename, "a reported false match passes the target's checks as they are now - it's the find");
                crate::ranges::solve_target(&mut tx, target.id, filename, user.id, now).await?;
            }
        }
    }
    tx.commit().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(table_size: i64, first_start_slot: i64, last_start_slot: i64) -> HashTablePosition {
        HashTablePosition { table_size, first_start_slot, last_start_slot }
    }

    #[test]
    fn start_slot_is_hash_type_0_within_the_table() {
        // The name that matched rez-wav-2's hashes by chance - its lookup
        // starts at slot 1786 of bw-ptbr-install.exe's 2048, and the file's
        // entry is at 851 (client/tests/match_check_test.cpp has it too).
        assert_eq!(start_slot("REZ\\SFZSSP050WSRL.WAV", 2048), 1786);
        assert_eq!(start_slot("REZ\\SFZSSP050WSRL.WAV", 1), 0);
    }

    #[test]
    fn reaches_entry_wraps_past_the_tables_end() {
        assert!(reaches_entry(&position(2048, 849, 851), 849));
        assert!(reaches_entry(&position(2048, 849, 851), 851));
        assert!(!reaches_entry(&position(2048, 849, 851), 848));
        assert!(!reaches_entry(&position(2048, 849, 851), 852));
        assert!(reaches_entry(&position(16, 14, 1), 15));
        assert!(reaches_entry(&position(16, 14, 1), 0));
        assert!(!reaches_entry(&position(16, 14, 1), 2));
        assert!(!reaches_entry(&position(16, 14, 1), 13));
    }

    #[test]
    fn validate_hash_table_position_wants_a_power_of_two_and_slots_in_it() {
        assert!(validate_hash_table_position(&position(2048, 849, 851)).is_ok());
        assert!(validate_hash_table_position(&position(16, 14, 1)).is_ok());
        assert!(validate_hash_table_position(&position(1, 0, 0)).is_ok());
        assert!(validate_hash_table_position(&position(1 << 31, 0, (1 << 31) - 1)).is_ok());
        assert!(validate_hash_table_position(&position(0, 0, 0)).is_err());
        assert!(validate_hash_table_position(&position(2000, 1, 2)).is_err());
        assert!(validate_hash_table_position(&position(1 << 32, 1, 2)).is_err());
        assert!(validate_hash_table_position(&position(2048, -1, 2)).is_err());
        assert!(validate_hash_table_position(&position(2048, 1, 2048)).is_err());
    }
}
