//! Canaries: now and then, a claim is handed a small range with a filename
//! planted in it - a virtual target, named `virtual-canary`, whose two hashes
//! are that filename's - instead of real work, and the client must find it.
//! That checks, continuously, each volunteer's actual hardware, driver and
//! build: bit flips (consumer GPUs have no ECC memory), a driver bug, a broken
//! release or backend, a modified client - nothing the client can check about
//! itself.
//!
//! A canary is sized for a few seconds of the client's measured rate, and
//! handed out with probability `RangeConfig::canary_probability` per claim
//! while there's real work, so it costs a small fraction of the real work. It
//! copies a real target's prefix, suffix, alphabet and pruning, so that it
//! searches the way the client's real work does; the planted candidate has
//! only letters and digits, which no pruning rule skips - but for
//! `min_backslash_count`, which a canary leaves at 0. (Text the target
//! inserts, which the canary inserts too, could break a rule whatever is
//! around it - but then every candidate of the target does as well.) Its target takes
//! part in nothing but this one claim - not other claims, the dashboard's
//! targets, `/status`, nor a volunteer's candidates, ranges or names found -
//! and its result is kept in `canaries` (see `complete`).

use std::sync::OnceLock;

use namebreak_protocol::{ClaimResponse, Version};
use rand::RngExt;

use crate::alphabet::{alphabet_size, candidate_to_index, client_alphabet_for, index_to_candidate, max_supported_len, space_size, split_end, Pos};
use crate::error::AppError;
use crate::models::{u32_to_i64, Range, Target, User};
use crate::ranges::{insert_claimed_range, to_claim_response};
use crate::state::RangeConfig;

/// Every canary's target's name. One name for all of them, as a client keeps
/// a matches file per target name.
pub const CANARY_TARGET_NAME: &str = "virtual-canary";

/// The shortest candidate a canary plants.
const MIN_CANARY_LEN: i64 = 6;

/// MPQ's crypt table, as the client's `prepareCryptTable` builds it.
fn crypt_table() -> &'static [u32; 0x500] {
    static TABLE: OnceLock<[u32; 0x500]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 0x500];
        let mut seed: u32 = 0x0010_0001;
        for index1 in 0..0x100 {
            let mut index2 = index1;
            for _ in 0..5 {
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                let high = (seed & 0xFFFF) << 16;
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                table[index2] = high | (seed & 0xFFFF);
                index2 += 0x100;
            }
        }
        table
    })
}

/// A name's MPQ hash with the crypt table at `offset` (0x100 for hash A,
/// 0x200 for hash B), over its bytes as given - as the client hashes it.
pub fn mpq_hash(name: &str, offset: usize) -> u32 {
    let table = crypt_table();
    let (mut seed1, mut seed2) = (0x7FED_7FEDu32, 0xEEEE_EEEEu32);
    for byte in name.bytes() {
        seed1 = table[offset + byte as usize] ^ seed1.wrapping_add(seed2);
        seed2 = (byte as u32).wrapping_add(seed1).wrapping_add(seed2).wrapping_add(seed2 << 5).wrapping_add(3);
    }
    seed1
}

/// A canary for `user`, now and then: with probability
/// `config.canary_probability`, if some real target is active and the
/// client's rate has been measured (a guessed one could make a canary take
/// far longer than `config.canary_seconds`). Its range is
/// claimed for `user` like any other; `None` means the claim goes on to
/// real work.
pub async fn maybe_claim(
    tx: &mut sqlx::SqliteConnection,
    config: &RangeConfig,
    user: &User,
    client_version: Version,
    rate: f64,
    now: i64,
) -> Result<Option<ClaimResponse>, AppError> {
    if user.ema_rate_per_sec.is_none() || config.canary_probability <= 0.0 || rand::rng().random::<f64>() >= config.canary_probability {
        return Ok(None);
    }
    // A real target the client searches as it is, whose prefix, suffix,
    // alphabet and pruning the canary copies - the one it would most likely
    // get work from.
    let templates = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE status = 'active' AND is_virtual = 0 ORDER BY priority DESC, created_at ASC")
        .fetch_all(&mut *tx)
        .await?;
    let Some(template) =
        templates.into_iter().find(|t| {
            client_alphabet_for(&t.alphabet_name, &t.alphabet, client_version) == Some(t.alphabet.as_str()) && t.insertions().searchable_by(client_version)
        })
    else {
        return Ok(None);
    };
    let Some(canary) = plan(&template.alphabet, (rate * config.canary_seconds).max(1.0) as Pos) else {
        return Ok(None);
    };
    let filename = format!("{}{}{}", template.prefix, template.insertions().insert(&canary.planted), template.suffix);

    let first = index_to_candidate(&template.alphabet, canary.start, canary.len);
    let last = index_to_candidate(&template.alphabet, canary.end - 1, canary.len);
    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, prune_unopened_brackets, \
         prune_whole_candidate, max_backslash_count, min_backslash_count, prune_adjacent_backslashes, insert_from_start_text, \
         insert_from_start_position, insert_from_end_text, insert_from_end_position, alphabet_name, alphabet, status, priority, start_len, \
         created_at, is_virtual) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, 'active', 0, ?, ?, 1) RETURNING id",
    )
    .bind(CANARY_TARGET_NAME)
    .bind(&template.prefix)
    .bind(&template.suffix)
    .bind(u32_to_i64(mpq_hash(&filename, 0x100)))
    .bind(u32_to_i64(mpq_hash(&filename, 0x200)))
    .bind(&first)
    .bind(&last)
    .bind(template.prune_symbol_runs)
    .bind(template.prune_unopened_brackets)
    .bind(template.prune_whole_candidate)
    .bind(template.max_backslash_count)
    .bind(template.prune_adjacent_backslashes)
    .bind(&template.insert_from_start_text)
    .bind(template.insert_from_start_position)
    .bind(&template.insert_from_end_text)
    .bind(template.insert_from_end_position)
    .bind(&template.alphabet_name)
    .bind(&template.alphabet)
    .bind(canary.len)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;
    // Every target has a cursor; a canary's is past its one range, so
    // nothing is ever carved from it.
    let (next_block, next_index) = split_end(&template.alphabet, canary.len, canary.end);
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_block, next_index, alphabet_name, alphabet) VALUES (?, ?, ?, ?, ?, ?)")
        .bind(target_id)
        .bind(canary.len)
        .bind(next_block)
        .bind(next_index)
        .bind(&template.alphabet_name)
        .bind(&template.alphabet)
        .execute(&mut *tx)
        .await?;

    let lease_seconds = config.lease_seconds;
    let range_id = insert_claimed_range(
        &mut *tx,
        target_id,
        canary.len,
        canary.start,
        canary.end,
        &template.alphabet_name,
        &template.alphabet,
        None,
        user,
        lease_seconds,
        now,
    )
    .await?;
    sqlx::query("INSERT INTO canaries (target_id, range_id, user_id, filename, created_at) VALUES (?, ?, ?, ?, ?)")
        .bind(target_id)
        .bind(range_id)
        .bind(user.id)
        .bind(&filename)
        .bind(now)
        .execute(&mut *tx)
        .await?;

    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(&mut *tx).await?;
    tracing::info!(range_id, user_id = user.id, candidates = %(canary.end - canary.start), "handed out a canary");
    Ok(Some(to_claim_response(&target, range_id, canary.len, canary.start, canary.end, lease_seconds, &template.alphabet, &template.alphabet)))
}

/// Where a canary goes: its range [start, end) of `len`-character
/// candidates, and the candidate planted in it.
#[derive(Debug)]
struct CanaryPlan {
    len: i64,
    start: Pos,
    end: Pos,
    planted: String,
}

/// A canary of `count` candidates in `alphabet` (fewer, if that's more than
/// a quarter of the shortest length that has room): at the shortest length,
/// from MIN_CANARY_LEN, with at least four times that many, a random candidate
/// of letters and digits planted at a random place in it. `None` if the
/// alphabet has neither letters nor digits.
fn plan(alphabet: &str, count: Pos) -> Option<CanaryPlan> {
    let plantable: Vec<char> = alphabet.chars().filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit()).collect();
    if plantable.is_empty() || alphabet_size(alphabet) < 2 {
        return None;
    }
    let max_len = max_supported_len(alphabet);
    let len = (MIN_CANARY_LEN.min(max_len)..=max_len).find(|&len| space_size(alphabet, len) >= count.saturating_mul(4)).unwrap_or(max_len);
    let space = space_size(alphabet, len);
    let count = count.clamp(1, space);

    let mut rng = rand::rng();
    let planted: String = (0..len).map(|_| plantable[rng.random_range(0..plantable.len())]).collect();
    let index = candidate_to_index(alphabet, &planted).expect("planted from the alphabet's own characters");
    let offset = rng.random_range(0..count as u64) as Pos;
    let start = (index - offset).clamp(0, space - count);
    Some(CanaryPlan { len, start, end: start + count, planted })
}

/// Records a canary's result, when the client completes its range: found if
/// it reported exactly the planted filename. Its target is then done with -
/// solved, or paused. Nothing if `range` isn't a canary's.
pub async fn complete(tx: &mut sqlx::SqliteConnection, range: &Range, found: bool, filename: Option<&str>, now: i64) -> Result<bool, AppError> {
    let Some(planted): Option<String> = sqlx::query_scalar("SELECT filename FROM canaries WHERE target_id = ?")
        .bind(range.target_id)
        .fetch_optional(&mut *tx)
        .await?
    else {
        return Ok(false);
    };
    let passed = found && filename == Some(planted.as_str());
    sqlx::query("UPDATE canaries SET result = ?, resolved_at = ? WHERE target_id = ?")
        .bind(if passed { "found" } else { "missed" })
        .bind(now)
        .bind(range.target_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE targets SET status = ?, found_filename = ?, found_at = ? WHERE id = ?")
        .bind(if passed { "solved" } else { "paused" })
        .bind(if passed { Some(planted.as_str()) } else { None })
        .bind(if passed { Some(now) } else { None })
        .bind(range.target_id)
        .execute(&mut *tx)
        .await?;
    if !passed {
        tracing::warn!(range_id = range.id, user_id = ?range.last_assigned_user_id, planted, reported = ?filename, "a canary was MISSED");
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpq_hash_matches_the_clients() {
        // The client's own (client/src/engine/mpq_hash.cpp) - and (LISTFILE)'s
        // are the well-known ones every MPQ tool has.
        assert_eq!(mpq_hash("(LISTFILE)", 0x100), 0xFD65_7910);
        assert_eq!(mpq_hash("(LISTFILE)", 0x200), 0x4E9B_98A7);
        assert_eq!(mpq_hash("REZ\\ABC123.WAV", 0x100), 0xFCAA_88A2);
        assert_eq!(mpq_hash("REZ\\ABC123.WAV", 0x200), 0xF559_2C3D);
    }

    #[test]
    fn plan_puts_a_letters_and_digits_candidate_inside_a_range_of_the_asked_size() {
        let alphabet = crate::alphabet::lookup_predefined_alphabet("size49").unwrap();
        for count in [1, 1_000, 2_500_000_000, 10_000_000_000_000] {
            for _ in 0..50 {
                let plan = plan(alphabet, count).unwrap();
                assert_eq!(plan.end - plan.start, count);
                assert!(plan.len >= MIN_CANARY_LEN);
                assert!(space_size(alphabet, plan.len) >= count * 4, "{plan:?}");
                assert!(plan.planted.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()), "{plan:?}");
                let index = candidate_to_index(alphabet, &plan.planted).unwrap();
                assert!(plan.start <= index && index < plan.end, "{plan:?}");
            }
        }
        // Letters only, in an alphabet without digits.
        let letters = crate::alphabet::lookup_predefined_alphabet("size29").unwrap();
        assert!(plan(letters, 1_000).unwrap().planted.chars().all(|c| c.is_ascii_uppercase()));
        assert!(plan("!#$%", 10).is_none(), "nothing to plant");
    }
}
