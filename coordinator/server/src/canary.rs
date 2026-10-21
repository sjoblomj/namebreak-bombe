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
//! around it - but then every candidate of the target does as well.) A
//! dictionary target's canary copies its words, separators and tails
//! instead, and its encryption key - made the planted name's (see
//! `dictionary_canary`). Its target takes part in nothing but this one claim - not other claims, the dashboard's
//! targets, `/status`, nor a volunteer's candidates, ranges, names or
//! basenames found -
//! and its result is kept in `canaries` (see `complete`).

use namebreak_protocol::{ClaimResponse, Version, DICTIONARY_SINCE, MEET_IN_THE_MIDDLE_SINCE};
use rand::RngExt;

use crate::alphabet::{alphabet_size, candidate_to_index, client_alphabet_for, index_to_candidate, max_supported_len, space_size, split_end, Pos};
use crate::error::AppError;
use crate::dictionary::DictionarySpace;
use crate::meet_in_the_middle::{self, MeetInTheMiddleSpace, MeetInTheMiddleTarget};
use crate::mpq::{continue_hash, mpq_hash, FILE_KEY_OFFSET, HASH_A_OFFSET, HASH_B_OFFSET, INITIAL_STATE};
use crate::models::{u32_to_i64, Range, Target, User, DICTIONARY_ALPHABET_NAME, MEET_IN_THE_MIDDLE_ALPHABET_NAME};
use crate::ranges::{insert_claimed_range, meet_in_the_middle_table_fits, to_claim_response, to_dictionary_claim_response, to_meet_in_the_middle_claim_response};
use crate::state::RangeConfig;

/// Every canary's target's name. One name for all of them, as a client keeps
/// a matches file per target name.
pub const CANARY_TARGET_NAME: &str = "virtual-canary";

/// The shortest candidate a canary plants.
const MIN_CANARY_LEN: i64 = 6;

/// A canary for `user`, now and then: with probability
/// `config.canary_probability`, copying the real target the client would
/// most likely get work from - the first, by priority, it can search - if
/// the client's rate for that kind of target has been measured (a guessed
/// one could make a canary take far longer than `config.canary_seconds`).
/// Its range is claimed for `user` like any other; `None` means the claim
/// goes on to real work.
pub async fn maybe_claim(
    tx: &mut sqlx::SqliteConnection,
    config: &RangeConfig,
    user: &User,
    client_version: Version,
    now: i64,
) -> Result<Option<ClaimResponse>, AppError> {
    if config.canary_probability <= 0.0 || rand::rng().random::<f64>() >= config.canary_probability {
        return Ok(None);
    }
    let templates = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE status = 'active' AND is_virtual = 0 ORDER BY priority DESC, created_at ASC")
        .fetch_all(&mut *tx)
        .await?;
    let mut template = None;
    for candidate in templates {
        let searchable = if candidate.is_dictionary() {
            client_version >= DICTIONARY_SINCE
        } else if candidate.is_meet_in_the_middle() {
            client_version >= MEET_IN_THE_MIDDLE_SINCE && meet_in_the_middle_table_fits(&candidate, user)?
        } else {
            client_alphabet_for(&candidate.alphabet_name, &candidate.alphabet, client_version) == Some(candidate.alphabet.as_str())
                && candidate.insertions().searchable_by(client_version)
        };
        if searchable {
            template = Some(candidate);
            break;
        }
    }
    let Some(template) = template else {
        return Ok(None);
    };
    if template.is_dictionary() {
        let Some(rate) = user.ema_dictionary_rate_per_sec else { return Ok(None) };
        dictionary_canary(tx, config, user, &template, rate, now).await
    } else if template.is_meet_in_the_middle() {
        let Some(rate) = user.ema_meet_in_the_middle_rate_per_sec else { return Ok(None) };
        meet_in_the_middle_canary(tx, config, user, &template, rate, now).await
    } else {
        let Some(rate) = user.ema_rate_per_sec else { return Ok(None) };
        alphabet_canary(tx, config, user, &template, rate, now).await
    }
}

/// A canary copying alphabet target `template`: its prefix, suffix,
/// alphabet, pruning and insertions.
async fn alphabet_canary(tx: &mut sqlx::SqliteConnection, config: &RangeConfig, user: &User, template: &Target, rate: f64, now: i64) -> Result<Option<ClaimResponse>, AppError> {
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
    .bind(u32_to_i64(mpq_hash(&filename, HASH_A_OFFSET)))
    .bind(u32_to_i64(mpq_hash(&filename, HASH_B_OFFSET)))
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
    let target = record(tx, target_id, range_id, user, &filename, canary.end - canary.start, now).await?;
    Ok(Some(to_claim_response(&target, range_id, canary.len, canary.start, canary.end, lease_seconds, &template.alphabet, &template.alphabet)))
}

/// A canary copying dictionary target `template`: its word lists,
/// separators, word counts, tails, prefix and suffix - at the word count its
/// sweep is at - and its encryption key and `send_basenames`, so the client
/// compares basenames to a key as it does for the real target: the key the
/// planted name's basename has. But not its bounds, which could leave the
/// planted candidate out: a canary has none. Its claim is made with the
/// template's `DictionaryTarget`, the same words, so the canary's target is
/// never loaded into `config.dictionaries` - a canary each time would fill
/// it.
async fn dictionary_canary(
    tx: &mut sqlx::SqliteConnection,
    config: &RangeConfig,
    user: &User,
    template: &Target,
    rate: f64,
    now: i64,
) -> Result<Option<ClaimResponse>, AppError> {
    let dictionary = config.dictionaries.get(&mut *tx, template).await?;
    let space = &dictionary.space;
    let cursor_words: i64 = sqlx::query_scalar("SELECT candidate_len FROM target_progress WHERE target_id = ?").bind(template.id).fetch_one(&mut *tx).await?;
    let words = cursor_words.clamp(space.min_words(), space.max_words());
    let canary = plan_dictionary(space, words, (rate * config.canary_seconds).max(1.0) as i64);
    let filename = format!("{}{}{}", template.prefix, space.text(canary.planted), template.suffix);
    let basename = filename.rsplit('\\').next().expect("rsplit yields at least one piece");
    let encryption_key = template.encryption_key.map(|_| u32_to_i64(mpq_hash(basename, FILE_KEY_OFFSET)));

    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, alphabet_name, alphabet, status, priority, \
         start_len, encryption_key, kind, word_lists, separators, min_words, max_words, tails, send_basenames, created_at, is_virtual) \
         VALUES (?, ?, ?, ?, ?, '', '', ?, '', 'active', 0, 1, ?, 'dictionary', ?, ?, ?, ?, ?, ?, ?, 1) RETURNING id",
    )
    .bind(CANARY_TARGET_NAME)
    .bind(&template.prefix)
    .bind(&template.suffix)
    .bind(u32_to_i64(mpq_hash(&filename, HASH_A_OFFSET)))
    .bind(u32_to_i64(mpq_hash(&filename, HASH_B_OFFSET)))
    .bind(DICTIONARY_ALPHABET_NAME)
    .bind(encryption_key)
    .bind(&template.word_lists)
    .bind(&template.separators)
    .bind(template.min_words)
    .bind(template.max_words)
    .bind(&template.tails)
    .bind(template.send_basenames)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;
    // A cursor past its one range, as an alphabet canary's.
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_block, next_index, alphabet_name, alphabet) VALUES (?, ?, 0, ?, ?, '')")
        .bind(target_id)
        .bind(words)
        .bind(canary.end)
        .bind(DICTIONARY_ALPHABET_NAME)
        .execute(&mut *tx)
        .await?;

    let lease_seconds = config.lease_seconds;
    let range_id =
        insert_claimed_range(&mut *tx, target_id, words, canary.start as Pos, canary.end as Pos, DICTIONARY_ALPHABET_NAME, "", None, user, lease_seconds, now).await?;
    let target = record(tx, target_id, range_id, user, &filename, (canary.end - canary.start) as Pos, now).await?;
    let mut claim = to_dictionary_claim_response(&target, &dictionary, range_id, canary.start, canary.end, lease_seconds);
    claim.filename_lower_bound = None;
    claim.filename_upper_bound = None;
    Ok(Some(claim))
}

/// A canary copying meet-in-the-middle target `template`: its alphabet, prefix,
/// suffix, pruning rules and - so that the client searches exactly the way
/// it does for the real target, within the memory it declared - the same
/// table length. Its meeting point is the state after a directory planted in
/// the range, and its hashes that directory's filename's, so finding it is
/// finding that directory: to the client it's an ordinary meet-in-the-middle
/// claim. One length only, the planted directory's, so the walk has nothing
/// else in it.
async fn meet_in_the_middle_canary(
    tx: &mut sqlx::SqliteConnection,
    config: &RangeConfig,
    user: &User,
    template: &Target,
    rate: f64,
    now: i64,
) -> Result<Option<ClaimResponse>, AppError> {
    let settings = meet_in_the_middle::of_target(template)?;
    let cursor_length: i64 = sqlx::query_scalar("SELECT candidate_len FROM target_progress WHERE target_id = ?").bind(template.id).fetch_one(&mut *tx).await?;
    let length = cursor_length.clamp(settings.space.min_length(), settings.space.max_length());
    let Some(canary) = plan_meet_in_the_middle(&settings.space, length, (rate * config.canary_seconds).max(1.0) as i64) else {
        return Ok(None);
    };
    // The meeting point of the planted directory - what the client searches
    // for - and the hashes of the name it makes, which it checks what it
    // finds against.
    let directory = format!("{}{}", template.prefix, canary.planted);
    let filename = format!("{directory}{}", template.suffix);
    let meeting_point_a = continue_hash(INITIAL_STATE, &directory, HASH_A_OFFSET);
    let meeting_point_b = continue_hash(INITIAL_STATE, &directory, HASH_B_OFFSET);

    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, prune_unopened_brackets, \
         prune_whole_candidate, max_backslash_count, min_backslash_count, prune_adjacent_backslashes, alphabet_name, alphabet, status, \
         priority, start_len, kind, meeting_point_a_seed1, meeting_point_a_seed2, meeting_point_b_seed1, meeting_point_b_seed2, meet_in_the_middle_max_length, \
         meet_in_the_middle_table_length, created_at, is_virtual) \
         VALUES (?, ?, ?, ?, ?, '', '', ?, ?, 0, ?, 0, ?, ?, ?, 'active', 0, ?, 'meet-in-the-middle', ?, ?, ?, ?, ?, ?, ?, 1) RETURNING id",
    )
    .bind(CANARY_TARGET_NAME)
    .bind(&template.prefix)
    .bind(&template.suffix)
    .bind(u32_to_i64(mpq_hash(&filename, HASH_A_OFFSET)))
    .bind(u32_to_i64(mpq_hash(&filename, HASH_B_OFFSET)))
    .bind(template.prune_symbol_runs)
    .bind(template.prune_unopened_brackets)
    .bind(template.max_backslash_count)
    .bind(template.prune_adjacent_backslashes)
    .bind(&template.alphabet_name)
    .bind(&template.alphabet)
    // One length: the planted directory's, so start_len is also the longest.
    .bind(length)
    .bind(u32_to_i64(meeting_point_a.0))
    .bind(u32_to_i64(meeting_point_a.1))
    .bind(u32_to_i64(meeting_point_b.0))
    .bind(u32_to_i64(meeting_point_b.1))
    .bind(length)
    .bind(settings.space.table_length())
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;
    // A cursor past its one range, as the other canaries'.
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_block, next_index, alphabet_name, alphabet) VALUES (?, ?, 0, ?, ?, '')")
        .bind(target_id)
        .bind(length)
        .bind(canary.end)
        .bind(MEET_IN_THE_MIDDLE_ALPHABET_NAME)
        .execute(&mut *tx)
        .await?;

    let lease_seconds = config.lease_seconds;
    let range_id =
        insert_claimed_range(&mut *tx, target_id, length, canary.start as Pos, canary.end as Pos, MEET_IN_THE_MIDDLE_ALPHABET_NAME, "", None, user, lease_seconds, now)
            .await?;
    let target = record(tx, target_id, range_id, user, &filename, (canary.end - canary.start) as Pos, now).await?;
    let meet_in_the_middle = MeetInTheMiddleTarget { space: canary.space, meeting_point_a, meeting_point_b };
    Ok(Some(to_meet_in_the_middle_claim_response(&target, &meet_in_the_middle, range_id, length, canary.start, canary.end, lease_seconds)))
}

/// Keeps a canary's planted `filename` in `canaries`, and reads its target
/// back.
async fn record(tx: &mut sqlx::SqliteConnection, target_id: i64, range_id: i64, user: &User, filename: &str, candidates: Pos, now: i64) -> Result<Target, AppError> {
    sqlx::query("INSERT INTO canaries (target_id, range_id, user_id, filename, created_at) VALUES (?, ?, ?, ?, ?)")
        .bind(target_id)
        .bind(range_id)
        .bind(user.id)
        .bind(filename)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    tracing::info!(range_id, user_id = user.id, %candidates, "handed out a canary");
    Ok(sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(&mut *tx).await?)
}

/// Where a dictionary canary goes: its range [start, end) of candidate
/// numbers, and the number of the candidate planted in it.
#[derive(Debug)]
struct DictionaryCanaryPlan {
    start: i64,
    end: i64,
    planted: i64,
}

/// A dictionary canary of `count` candidates of `words` words (all of
/// them, if there are fewer), with a random one of them planted at a random
/// place in it. Every candidate is one a search compares - a dictionary
/// search prunes nothing.
fn plan_dictionary(space: &DictionarySpace, words: i64, count: i64) -> DictionaryCanaryPlan {
    let (block_start, block_size) = (space.block_start(words), space.block_size(words));
    let count = count.clamp(1, block_size);
    let mut rng = rand::rng();
    let planted = block_start + rng.random_range(0..block_size);
    let offset = rng.random_range(0..count);
    let start = (planted - offset).clamp(block_start, block_start + block_size - count);
    DictionaryCanaryPlan { start, end: start + count, planted }
}

/// Where a meet-in-the-middle canary goes: its range [start, end) of tail
/// indices, the directory planted in it, and the space of its own one-length
/// target.
#[derive(Debug)]
struct MeetInTheMiddleCanaryPlan {
    space: MeetInTheMiddleSpace,
    start: i64,
    end: i64,
    planted: String,
}

/// A meet-in-the-middle canary of `count` tails (fewer, if that's more than
/// the length has) at directory `length`, with a directory of letters and
/// digits planted at a random place in it - no pruning rule skips one of
/// those, as for an alphabet canary, and the canary leaves
/// `min_backslash_count` at 0 since the planted directory has no '\'. `None`
/// if the alphabet has neither letters nor digits, or the length can't make a
/// space of its own.
fn plan_meet_in_the_middle(space: &MeetInTheMiddleSpace, length: i64, count: i64) -> Option<MeetInTheMiddleCanaryPlan> {
    let plantable: Vec<char> = space.alphabet().chars().filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit()).collect();
    if plantable.is_empty() {
        return None;
    }
    // Its own target searches this length alone, with the template's table:
    // the same memory, and a walk of just the tails around the planted one.
    let one_length = MeetInTheMiddleSpace::new(space.alphabet(), length, length, space.table_length()).ok()?;
    let tails = one_length.tails_at(length);
    let count = count.clamp(1, tails);
    let mut rng = rand::rng();
    let planted: String = (0..length).map(|_| plantable[rng.random_range(0..plantable.len())]).collect();
    let index = one_length.tail_index(&planted[space.table_length() as usize..]).expect("planted from the alphabet's own characters");
    let offset = rng.random_range(0..count);
    let start = (index - offset).clamp(0, tails - count);
    Some(MeetInTheMiddleCanaryPlan { space: one_length, start, end: start + count, planted })
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
