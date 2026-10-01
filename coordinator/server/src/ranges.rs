//! Core range life-cycle: carving fresh work off a target's cursor, reassigning
//! timed-out work, and recording completions. Lives separately from `handlers.rs`
//! so the HTTP glue stays thin.

use std::collections::HashMap;

use namebreak_protocol::{ClaimResponse, Version};
use sqlx::SqlitePool;

use crate::alphabet::{
    alphabet_size, bound_indices_at_len, candidate_to_index, ceil_index, client_alphabet_for, floor_index, index_to_candidate, max_supported_len, pattern_spans,
    range_bound_filenames, space_size, split_end, split_pos, strip_prefix_suffix, transition_alphabet_cursor, Pos, PREDEFINED_ALPHABETS,
};
use crate::error::AppError;
use crate::models::{i64_to_u32, PriorityRange, Range, Segment, SkipRange, Target, TargetProgress, User};
use crate::state::{now_unix, RangeConfig};

fn effective_rate(config: &RangeConfig, user: &User) -> f64 {
    user.ema_rate_per_sec.unwrap_or(config.default_rate_per_sec)
}

/// How many of `alphabet`'s candidates at `candidate_len` to carve for a
/// client searching at `rate` in `client_alphabet` (see
/// `alphabet::client_alphabet_for`), so it takes about
/// `target_chunk_seconds`. A client searching in a bigger alphabet searches
/// about `(its size / alphabet's size) ^ candidate_len` candidates per
/// candidate of the range, so it gets that many times fewer.
fn chunk_size(config: &RangeConfig, rate: f64, alphabet: &str, client_alphabet: &str, candidate_len: i64) -> Pos {
    let desired = (rate * config.target_chunk_seconds).round() as i64;
    let chunk = desired.clamp(config.min_chunk_candidates, config.max_chunk_candidates) as f64;
    let ratio = (alphabet_size(alphabet) as f64 / alphabet_size(client_alphabet) as f64).powi(candidate_len as i32);
    ((chunk * ratio).round() as Pos).max(1)
}

/// `alphabet` is the range's own alphabet (`Range::alphabet`), not
/// necessarily `target.alphabet` - a range carved before the target's
/// alphabet was last patched must still be served (and decoded) with
/// whatever alphabet it was actually carved under. `client_alphabet` is what
/// the client searches it with (see `alphabet::client_alphabet_for`): with a
/// bigger one, the same first and last candidate bound every candidate of
/// the range, plus the ones only the bigger alphabet has.
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_claim_response(
    target: &Target,
    range_id: i64,
    candidate_len: i64,
    start_index: Pos,
    end_index: Pos,
    lease_seconds: i64,
    alphabet: &str,
    client_alphabet: &str,
) -> ClaimResponse {
    let (lower_bound_filename, upper_bound_filename) = range_bound_filenames(alphabet, &target.prefix, &target.suffix, candidate_len, start_index, end_index);
    let candidate_count = if client_alphabet == alphabet {
        end_index - start_index
    } else {
        let first = index_to_candidate(alphabet, start_index, candidate_len);
        let last = index_to_candidate(alphabet, end_index - 1, candidate_len);
        let at = |candidate: &str| candidate_to_index(client_alphabet, candidate).expect("client_alphabet contains every character of alphabet");
        at(&last) - at(&first) + 1
    };
    ClaimResponse {
        range_id,
        target_id: target.id,
        target_name: target.name.clone(),
        prefix: target.prefix.clone(),
        suffix: target.suffix.clone(),
        hash_a_hex: format!("0x{:08X}", i64_to_u32(target.hash_a)),
        hash_b_hex: format!("0x{:08X}", i64_to_u32(target.hash_b)),
        prune_symbol_runs: target.prune_symbol_runs != 0,
        prune_unopened_brackets: target.prune_unopened_brackets != 0,
        prune_whole_candidate: target.prune_whole_candidate != 0,
        max_backslash_count: target.max_backslash_count,
        lower_bound_filename,
        upper_bound_filename,
        alphabet: client_alphabet.to_string(),
        // Only ever a chunk carved to at most max_chunk_candidates (or the
        // remainder of one, or that chunk in a slightly bigger alphabet - see
        // chunk_size), so this can't actually saturate.
        candidate_count: i64::try_from(candidate_count).unwrap_or(i64::MAX),
        lease_seconds,
    }
}

/// If `next_index` has run past this length's upper bound, advances the
/// cursor to the start of the next candidate length (mirrors the pre-skip
/// carving logic, just pulled out so both the "skip swallows the whole
/// cursor" and "skip trims a handed-out chunk" branches in `claim_range` can
/// share it). A no-op once `len` has already reached `max_supported_len`.
fn bump_length_if_exhausted(alphabet: &str, lower_bound: &str, upper_bound: &str, len: i64, next_index: Pos, upper_at_len: Pos) -> (i64, Pos) {
    if next_index > upper_at_len && len < max_supported_len(alphabet) {
        let new_len = len + 1;
        let new_next_index = bound_indices_at_len(alphabet, lower_bound, upper_bound, new_len).0;
        (new_len, new_next_index)
    } else {
        (len, next_index)
    }
}

async fn persist_progress(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    next_index: Pos,
    alphabet_name: &str,
    alphabet: &str,
) -> Result<(), AppError> {
    let (next_block, next_index) = split_end(alphabet, candidate_len, next_index);
    sqlx::query("UPDATE target_progress SET candidate_len = ?, next_block = ?, next_index = ?, alphabet_name = ?, alphabet = ? WHERE target_id = ?")
        .bind(candidate_len)
        .bind(next_block)
        .bind(next_index)
        .bind(alphabet_name)
        .bind(alphabet)
        .bind(target_id)
        .execute(tx)
        .await?;
    Ok(())
}

/// Records a skip run as its own permanently unclaimable range, the moment
/// it's actually about to be reached by carving - see `claim_range`. Shown on
/// the dashboard as "Skip" (dashboard.html); never selected by the `pending`
/// reuse query in `claim_range` or the `in_progress` sweep in
/// `reclaim_expired`, so it needs no further handling once inserted.
/// `priority_range_id` is `Some` when this skip run was found while carving
/// a priority range rather than the target's own main sweep - see
/// `claim_priority_range_chunk` - and `skip_range_id` when it's a skip
/// range's rather than what an alphabet change left behind. Both purely for
/// dashboard labeling.
#[allow(clippy::too_many_arguments)]
async fn insert_skipped_range(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    start_index: Pos,
    end_index: Pos,
    alphabet_name: &str,
    alphabet: &str,
    priority_range_id: Option<i64>,
    skip_range_id: Option<i64>,
    now: i64,
) -> Result<(), AppError> {
    let (start_block, start_index) = split_pos(alphabet, candidate_len, start_index);
    let (end_block, end_index) = split_end(alphabet, candidate_len, end_index);
    sqlx::query(
        "INSERT INTO ranges (target_id, candidate_len, start_block, start_index, end_block, end_index, status, created_at, alphabet_name, alphabet, \
         priority_range_id, skip_range_id) \
         VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?, ?, ?)",
    )
    .bind(target_id)
    .bind(candidate_len)
    .bind(start_block)
    .bind(start_index)
    .bind(end_block)
    .bind(end_index)
    .bind(now)
    .bind(alphabet_name)
    .bind(alphabet)
    .bind(priority_range_id)
    .bind(skip_range_id)
    .execute(tx)
    .await?;
    Ok(())
}

/// Inserts an unclaimed range, for `claim_range` to hand out ahead of fresh
/// carving - see `remove_skip_range` and `split_off_chunk`.
#[allow(clippy::too_many_arguments)]
async fn insert_pending_range(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    start_index: Pos,
    end_index: Pos,
    alphabet_name: &str,
    alphabet: &str,
    priority_range_id: Option<i64>,
    created_at: i64,
) -> Result<(), sqlx::Error> {
    let (start_block, start_index) = split_pos(alphabet, candidate_len, start_index);
    let (end_block, end_index) = split_end(alphabet, candidate_len, end_index);
    sqlx::query(
        "INSERT INTO ranges (target_id, candidate_len, start_block, start_index, end_block, end_index, status, created_at, alphabet_name, alphabet, \
         priority_range_id) \
         VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
    )
    .bind(target_id)
    .bind(candidate_len)
    .bind(start_block)
    .bind(start_index)
    .bind(end_block)
    .bind(end_index)
    .bind(created_at)
    .bind(alphabet_name)
    .bind(alphabet)
    .bind(priority_range_id)
    .execute(tx)
    .await?;
    Ok(())
}

/// Inserts a freshly carved range, already claimed by `user`, returning its id.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn insert_claimed_range(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    start_index: Pos,
    end_index: Pos,
    alphabet_name: &str,
    alphabet: &str,
    priority_range_id: Option<i64>,
    user: &User,
    lease_seconds: i64,
    now: i64,
) -> Result<i64, AppError> {
    let (start_block, start_index) = split_pos(alphabet, candidate_len, start_index);
    let (end_block, end_index) = split_end(alphabet, candidate_len, end_index);
    let range_id = sqlx::query_scalar(
        "INSERT INTO ranges (target_id, candidate_len, start_block, start_index, end_block, end_index, status, \
         assigned_user_id, last_assigned_user_id, assigned_at, lease_seconds, lease_expires_at, created_at, alphabet_name, alphabet, priority_range_id, last_progress_at) \
         VALUES (?, ?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(target_id)
    .bind(candidate_len)
    .bind(start_block)
    .bind(start_index)
    .bind(end_block)
    .bind(end_index)
    .bind(user.id)
    .bind(user.id)
    .bind(now)
    .bind(lease_seconds)
    .bind(now + lease_seconds)
    .bind(now)
    .bind(alphabet_name)
    .bind(alphabet)
    .bind(priority_range_id)
    .bind(now)
    .fetch_one(tx)
    .await?;
    Ok(range_id)
}

/// Sets a priority range's own cursor (`next_index` and its block).
async fn set_priority_range_next(tx: &mut sqlx::SqliteConnection, pr: &PriorityRange, next_index: Pos) -> Result<(), AppError> {
    let (next_block, next_index) = split_end(&pr.alphabet, pr.candidate_len, next_index);
    sqlx::query("UPDATE priority_ranges SET next_block = ?, next_index = ? WHERE id = ?").bind(next_block).bind(next_index).bind(pr.id).execute(tx).await?;
    Ok(())
}

/// Which kind of range a segment belongs to - see `insert_segments`.
#[derive(Clone, Copy)]
pub enum SegmentOwner {
    Skip,
    Priority,
}

/// Stores `spans` (see `alphabet::pattern_spans`) as the segments of the
/// skip or priority range `owner_id`.
pub async fn insert_segments(
    tx: &mut sqlx::SqliteConnection,
    owner: SegmentOwner,
    owner_id: i64,
    alphabet: &str,
    candidate_len: i64,
    spans: &[(Pos, Pos)],
) -> Result<(), AppError> {
    let sql = match owner {
        SegmentOwner::Skip => "INSERT INTO skip_range_segments (skip_range_id, start_block, start_index, end_block, end_index) VALUES (?, ?, ?, ?, ?)",
        SegmentOwner::Priority => {
            "INSERT INTO priority_range_segments (priority_range_id, start_block, start_index, end_block, end_index) VALUES (?, ?, ?, ?, ?)"
        }
    };
    for &(start, end) in spans {
        let (start_block, start_index) = split_pos(alphabet, candidate_len, start);
        let (end_block, end_index) = split_end(alphabet, candidate_len, end);
        sqlx::query(sql)
            .bind(owner_id)
            .bind(start_block)
            .bind(start_index)
            .bind(end_block)
            .bind(end_index)
            .execute(&mut *tx)
            .await?;
    }
    Ok(())
}

/// Every segment of every one of `target_id`'s priority ranges, by priority
/// range id - still split into blocks, since only each priority range's own
/// alphabet and length can join them (see `owned_spans`).
async fn priority_segments_by_range(tx: &mut sqlx::SqliteConnection, target_id: i64) -> Result<HashMap<i64, Vec<Segment>>, AppError> {
    let segments = sqlx::query_as::<_, Segment>(
        "SELECT priority_range_id AS owner_id, start_block, start_index, end_block, end_index FROM priority_range_segments \
         WHERE priority_range_id IN (SELECT id FROM priority_ranges WHERE target_id = ?)",
    )
    .bind(target_id)
    .fetch_all(&mut *tx)
    .await?;
    let mut by_range: HashMap<i64, Vec<Segment>> = HashMap::new();
    for segment in segments {
        by_range.entry(segment.owner_id).or_default().push(segment);
    }
    Ok(by_range)
}

/// The candidates `pr` owns: its segments, cut to its own
/// `[start_index, end_index)`, sorted.
pub fn owned_spans(pr: &PriorityRange, segments: Option<&Vec<Segment>>) -> Vec<(Pos, Pos)> {
    let mut spans: Vec<(Pos, Pos)> = segments
        .into_iter()
        .flatten()
        .map(|s| s.span(&pr.alphabet, pr.candidate_len))
        .map(|(start, end)| (start.max(pr.start()), end.min(pr.end())))
        .filter(|&(start, end)| start < end)
        .collect();
    spans.sort();
    spans
}

/// How many candidates of `spans` fall within `[lo, hi)`.
pub fn count_within(spans: &[(Pos, Pos)], lo: Pos, hi: Pos) -> Pos {
    spans.iter().map(|&(start, end)| (end.min(hi) - start.max(lo)).max(0)).sum()
}

/// Whether any span of `a` overlaps any span of `b`.
fn spans_overlap(a: &[(Pos, Pos)], b: &[(Pos, Pos)]) -> bool {
    a.iter().any(|&(a0, a1)| b.iter().any(|&(b0, b1)| a0.max(b0) < a1.min(b1)))
}

/// `spans` sorted, with overlapping and touching ones merged.
fn normalize_spans(spans: &[(Pos, Pos)]) -> Vec<(Pos, Pos)> {
    let mut sorted: Vec<(Pos, Pos)> = spans.iter().copied().filter(|&(start, end)| start < end).collect();
    sorted.sort();
    let mut merged: Vec<(Pos, Pos)> = Vec::with_capacity(sorted.len());
    for (start, end) in sorted {
        match merged.last_mut() {
            Some((_, last_end)) if start <= *last_end => *last_end = (*last_end).max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// The parts of `a` that are also in `b`.
fn intersect_spans(a: &[(Pos, Pos)], b: &[(Pos, Pos)]) -> Vec<(Pos, Pos)> {
    let b = normalize_spans(b);
    let mut out = Vec::new();
    for (a0, a1) in normalize_spans(a) {
        for &(b0, b1) in &b {
            let (start, end) = (a0.max(b0), a1.min(b1));
            if start < end {
                out.push((start, end));
            }
        }
    }
    out
}

/// The parts of `spans` (see `alphabet::pattern_spans`) at `candidate_len`
/// that fall within `[lower_bound, upper_bound]` - a pattern knows nothing of
/// the target it's applied to, so e.g. `"GA"` on a target bounded above by
/// `GAMEMENU.BIN` would otherwise cover all of `GA_________` too. Empty if
/// none of it does.
pub fn clip_spans_to_bounds(alphabet: &str, lower_bound: &str, upper_bound: &str, candidate_len: i64, spans: &[(Pos, Pos)]) -> Vec<(Pos, Pos)> {
    let (lo, hi) = bound_indices_at_len(alphabet, lower_bound, upper_bound, candidate_len);
    intersect_spans(spans, &[(lo, hi + 1)])
}

/// The parts of `a` that aren't in `b`.
fn subtract_spans(a: &[(Pos, Pos)], b: &[(Pos, Pos)]) -> Vec<(Pos, Pos)> {
    let b = normalize_spans(b);
    let mut out = Vec::new();
    for (mut start, end) in normalize_spans(a) {
        for &(b0, b1) in &b {
            if b1 <= start || b0 >= end {
                continue;
            }
            if b0 > start {
                out.push((start, b0));
            }
            start = start.max(b1);
        }
        if start < end {
            out.push((start, end));
        }
    }
    out
}

/// Writes `skipped` rows, tagged with `skip_range_id`, for every part of
/// `spans` (at `candidate_len`, in `alphabet`) within `target`'s bounds that
/// no range covers yet - so a skip range shows on the dashboard as soon as
/// it's created. Carving then jumps straight over its segments (see
/// `claim_range`), so nothing gets skipped twice.
#[allow(clippy::too_many_arguments)]
async fn write_skipped_rows(
    tx: &mut sqlx::SqliteConnection,
    target: &Target,
    skip_range_id: i64,
    candidate_len: i64,
    alphabet_name: &str,
    alphabet: &str,
    spans: &[(Pos, Pos)],
    now: i64,
) -> Result<(), AppError> {
    let (lo, hi) = bound_indices_at_len(alphabet, &target.lower_bound, &target.upper_bound, candidate_len);
    let covered: Vec<(Pos, Pos)> = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE target_id = ? AND candidate_len = ? AND alphabet_name = ?")
        .bind(target.id)
        .bind(candidate_len)
        .bind(alphabet_name)
        .fetch_all(&mut *tx)
        .await?
        .iter()
        .map(|r| (r.start(), r.end()))
        .collect();
    for (start, end) in subtract_spans(&intersect_spans(spans, &[(lo, hi + 1)]), &covered) {
        insert_skipped_range(&mut *tx, target.id, candidate_len, start, end, alphabet_name, alphabet, None, Some(skip_range_id), now).await?;
    }
    Ok(())
}

/// The parts of `spans` (at `candidate_len`, in `alphabet_name`) that
/// carving will still reach if they stop being skipped: ahead of the main
/// sweep and outside every priority range, or inside a priority range and
/// ahead of that priority range's own cursor. Everything else has been
/// passed, and neither the sweep nor a priority range ever goes back. A
/// sweep walking in another alphabet (an alphabet change it hasn't caught
/// up with yet) isn't comparable, so it counts as having passed everything:
/// keeping something skipped is always safe, un-skipping it only when it'll
/// really be searched.
async fn still_to_be_reached(
    tx: &mut sqlx::SqliteConnection,
    target: &Target,
    candidate_len: i64,
    alphabet_name: &str,
    alphabet: &str,
    spans: &[(Pos, Pos)],
) -> Result<Vec<(Pos, Pos)>, AppError> {
    let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
        .bind(target.id)
        .fetch_one(&mut *tx)
        .await?;
    let whole = space_size(alphabet, candidate_len);
    // Mirrors claim_range: a sweep below start_len jumps to the start of start_len.
    let (sweep_len, sweep_at) = if progress.candidate_len < target.start_len {
        (target.start_len, bound_indices_at_len(alphabet, &target.lower_bound, &target.upper_bound, target.start_len).0)
    } else {
        (progress.candidate_len, progress.next())
    };
    let sweep_ahead = if progress.alphabet_name != alphabet_name || sweep_len > candidate_len {
        vec![]
    } else if sweep_len < candidate_len {
        vec![(0, whole)]
    } else {
        vec![(sweep_at, whole)]
    };

    let priority_ranges =
        sqlx::query_as::<_, PriorityRange>("SELECT * FROM priority_ranges WHERE target_id = ? AND candidate_len = ? AND alphabet_name = ?")
            .bind(target.id)
            .bind(candidate_len)
            .bind(alphabet_name)
            .fetch_all(&mut *tx)
            .await?;
    let segments = priority_segments_by_range(tx, target.id).await?;
    let mut owned = Vec::new();
    let mut priority_ahead = Vec::new();
    for pr in &priority_ranges {
        let spans = owned_spans(pr, segments.get(&pr.id));
        priority_ahead.extend(intersect_spans(&spans, &[(pr.next(), whole)]));
        owned.extend(spans);
    }

    let mut reached = subtract_spans(&sweep_ahead, &owned);
    reached.extend(priority_ahead);
    Ok(intersect_spans(spans, &reached))
}

/// Removes the parts of skip range `skip_range_id`'s `skipped` rows (only
/// those in `alphabet_name`, when given) that carving will still reach (see
/// `still_to_be_reached`), so they're searched after all - splitting a row
/// where only part of it is reached. Returns how many of its rows are left.
async fn unskip_unreached_rows(tx: &mut sqlx::SqliteConnection, target: &Target, skip_range_id: i64, alphabet_name: Option<&str>) -> Result<i64, AppError> {
    let rows = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE skip_range_id = ? AND status = 'skipped'")
        .bind(skip_range_id)
        .fetch_all(&mut *tx)
        .await?;
    let mut left = 0;
    for row in rows {
        if alphabet_name.is_some_and(|name| name != row.alphabet_name) {
            left += 1;
            continue;
        }
        let span = [(row.start(), row.end())];
        let reached = still_to_be_reached(tx, target, row.candidate_len, &row.alphabet_name, &row.alphabet, &span).await?;
        if reached.is_empty() {
            left += 1;
            continue;
        }
        sqlx::query("DELETE FROM ranges WHERE id = ?").bind(row.id).execute(&mut *tx).await?;
        for (start, end) in subtract_spans(&span, &reached) {
            insert_skipped_range(&mut *tx, target.id, row.candidate_len, start, end, &row.alphabet_name, &row.alphabet, None, Some(skip_range_id), row.created_at)
                .await?;
            left += 1;
        }
    }
    Ok(left)
}

/// The earliest of `spans` (each tagged with its owner's id) overlapping
/// `[lo, hi)`, cut to it.
fn first_overlap(spans: &[(Pos, Pos, i64)], lo: Pos, hi: Pos) -> Option<(Pos, Pos, i64)> {
    spans
        .iter()
        .map(|&(start, end, id)| (start.max(lo), end.min(hi), id))
        .filter(|&(start, end, _)| start < end)
        .min_by_key(|&(start, _, _)| start)
}

/// Every segment of `target_id`'s skip ranges at `candidate_len`, under
/// `alphabet_name`, tagged with its skip range's id. Only skip ranges in the
/// alphabet being carved in count, the same way as for priority ranges in
/// `priority_spans_at` - an alphabet patch re-expands every skip range
/// right away (see `migrate_skip_ranges_to_new_alphabet`), so this only
/// leaves out the old alphabet's, which nothing carves in any more.
async fn skip_spans_at(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    alphabet_name: &str,
    alphabet: &str,
) -> Result<Vec<(Pos, Pos, i64)>, AppError> {
    let segments = sqlx::query_as::<_, Segment>(
        "SELECT s.skip_range_id AS owner_id, s.start_block, s.start_index, s.end_block, s.end_index \
         FROM skip_range_segments s JOIN skip_ranges r ON r.id = s.skip_range_id \
         WHERE r.target_id = ? AND r.candidate_len = ? AND r.alphabet_name = ?",
    )
    .bind(target_id)
    .bind(candidate_len)
    .bind(alphabet_name)
    .fetch_all(&mut *tx)
    .await?;
    Ok(segments
        .iter()
        .map(|s| {
            let (start, end) = s.span(alphabet, candidate_len);
            (start, end, s.owner_id)
        })
        .collect())
}

/// Everything `target_id`'s priority ranges at `candidate_len`, frozen under
/// `alphabet_name`, own (see `owned_spans`), tagged with the priority range's
/// id - what the target's own main sweep must never cross without jumping
/// straight over it (see `claim_range`), whether or not the priority range
/// is done handing it out. Only priority ranges frozen under the exact
/// alphabet the cursor is currently walking in are considered: one created
/// under an alphabet this target has since moved away from (via
/// `handlers::admin_patch_target`) stores indices that are no longer
/// comparable to the cursor's, so it's left out here rather than risking a
/// wrong comparison - a rare, narrow edge case (a priority range plus a
/// later alphabet patch on the very same target) that's flagged rather than
/// silently mishandled.
async fn priority_spans_at(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    alphabet_name: &str,
) -> Result<Vec<(Pos, Pos, i64)>, AppError> {
    let priority_ranges = sqlx::query_as::<_, PriorityRange>(
        "SELECT * FROM priority_ranges WHERE target_id = ? AND candidate_len = ? AND alphabet_name = ?",
    )
    .bind(target_id)
    .bind(candidate_len)
    .bind(alphabet_name)
    .fetch_all(&mut *tx)
    .await?;
    let segments = priority_segments_by_range(tx, target_id).await?;

    Ok(priority_ranges
        .iter()
        .flat_map(|pr| owned_spans(pr, segments.get(&pr.id)).into_iter().map(|(start, end)| (start, end, pr.id)))
        .collect())
}

/// Tries to carve a fresh chunk from one of `target`'s priority ranges (see
/// `models::PriorityRange`), highest priority first (ties broken by creation
/// order, the same convention `targets.priority` already uses). Returns
/// `None` once none of them has any claimable work left.
///
/// A priority range is always scoped to exactly one candidate_len, so -
/// unlike the target's own cursor - there's no "bump to the next length"
/// once one is exhausted: it just permanently stops being claimable, and the
/// next iteration of the loop below moves on to the next-highest-priority
/// one instead (or falls through to `None`, letting `claim_range` continue
/// with the target's own main sweep). A chunk never runs past the end of
/// the segment it starts in - the gap after it isn't this priority range's.
///
/// Never re-validates a priority range's span against the target's own
/// cursor: that was already done once, and if needed clamped, at creation
/// time (see `handlers::admin_create_priority_range`) - from here on what
/// it owns is a permanent exclusion zone the main sweep never enters
/// (`priority_spans_at`), so there's nothing left to reconcile on every
/// claim.
async fn claim_priority_range_chunk(
    tx: &mut sqlx::SqliteConnection,
    config: &RangeConfig,
    target: &Target,
    user: &User,
    client_version: Version,
    rate: f64,
    now: i64,
) -> Result<Option<ClaimResponse>, AppError> {
    loop {
        let priority_ranges = sqlx::query_as::<_, PriorityRange>("SELECT * FROM priority_ranges WHERE target_id = ? ORDER BY priority DESC, created_at ASC")
            .bind(target.id)
            .fetch_all(&mut *tx)
            .await?;
        let segments = priority_segments_by_range(tx, target.id).await?;

        // The first priority range with anything left that this client can
        // search, where its next chunk starts and where the segment that's
        // in ends. Filtered here rather than in SQL, since a position is
        // only comparable once its block and index are joined.
        let mut found = None;
        for pr in priority_ranges {
            if pr.next() >= pr.end() || client_alphabet_for(&pr.alphabet_name, &pr.alphabet, client_version).is_none() {
                continue;
            }
            match owned_spans(&pr, segments.get(&pr.id)).into_iter().find(|&(_, end)| end > pr.next()) {
                Some((start, end)) => {
                    found = Some((start.max(pr.next()), end, pr));
                    break;
                }
                // Only gaps left before its end - nothing more to hand out.
                None => set_priority_range_next(&mut *tx, &pr, pr.end()).await?,
            }
        }
        let Some((pr_next, pr_end, pr)) = found else {
            return Ok(None);
        };
        let client_alphabet = client_alphabet_for(&pr.alphabet_name, &pr.alphabet, client_version).expect("checked above");

        // Skip ranges still apply within a priority range's own space: a
        // candidate being uninteresting doesn't stop being true just because
        // it's also prioritized. Their `skipped` rows were written when they
        // were created (see `write_skipped_rows`), so they're just jumped over.
        let skip_spans = skip_spans_at(&mut *tx, target.id, pr.candidate_len, &pr.alphabet_name, &pr.alphabet).await?;
        let skip_run = first_overlap(&skip_spans, pr_next, pr_end);

        if let Some((skip_start, skip_end, _)) = skip_run {
            if skip_start == pr_next {
                set_priority_range_next(&mut *tx, &pr, skip_end).await?;
                continue; // this priority range may have room left after the skip, or may now be exhausted - reconsider from the top either way
            }
        }

        let remaining = pr_end - pr_next;
        let natural_chunk = chunk_size(config, rate, &pr.alphabet, client_alphabet, pr.candidate_len).min(remaining);
        let start_index = pr_next;
        let natural_end = start_index + natural_chunk;

        let (end_index, next_index) = match skip_run {
            Some((skip_start, skip_end, _)) if natural_end >= skip_start => (skip_start, skip_end),
            _ => (natural_end, natural_end),
        };
        set_priority_range_next(&mut *tx, &pr, next_index).await?;

        let lease_seconds = config.lease_seconds;
        let range_id = insert_claimed_range(
            &mut *tx,
            target.id,
            pr.candidate_len,
            start_index,
            end_index,
            &pr.alphabet_name,
            &pr.alphabet,
            Some(pr.id),
            user,
            lease_seconds,
            now,
        )
        .await?;

        return Ok(Some(to_claim_response(target, range_id, pr.candidate_len, start_index, end_index, lease_seconds, &pr.alphabet, client_alphabet)));
    }
}

/// Tries to hand `user` a unit of work: for the highest-priority active target
/// that has any, either an already-carved pending range of its own, or else a
/// freshly carved slice of its remaining space. Returns `None` when there is
/// nothing available on any active target.
///
/// Priority is strict, not weighted: targets are walked highest-priority
/// first (ties broken by creation order), and the walk stops at the first one
/// with claimable work - a pending range or room left to carve. A
/// lower-priority target is never touched while a higher-priority one still
/// has either, even if the higher-priority target's own claimable work is
/// just a leftover pending range rather than fresh space; it can starve
/// completely while a higher-priority target is still being worked.
pub async fn claim_range(pool: &SqlitePool, config: &RangeConfig, user: &User) -> Result<Option<ClaimResponse>, AppError> {
    let mut tx = pool.begin().await?;
    let now = now_unix();
    let rate = effective_rate(config, user);
    // Falls back to the oldest possible version on a stored value that
    // somehow doesn't parse (should never happen - handlers::register
    // already validates it before it's ever written) rather than failing
    // the whole claim - the effect is just the most conservative possible
    // gating, never handing this client anything newer than the original
    // protocol version until it re-registers with a well-formed one.
    let client_version: Version = user.protocol_version.parse().unwrap_or(Version::new(1, 0, 0));

    // Now and then a canary instead (see canary.rs).
    if let Some(claim) = crate::canary::maybe_claim(&mut tx, config, user, client_version, rate, now).await? {
        tx.commit().await?;
        return Ok(Some(claim));
    }

    let targets = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE status = 'active' AND is_virtual = 0 ORDER BY priority DESC, created_at ASC")
        .fetch_all(&mut *tx)
        .await?;

    for target in targets {
        // A client that can't search the target's alphabet - it knows
        // neither that alphabet nor any bigger one containing it (see
        // alphabet::client_alphabet_for) - gets no work from this target at
        // all, not even a pending range left over in an older alphabet it
        // could search. An old client simply doesn't get *new* work from
        // this target until it's upgraded, rather than the server trying to
        // thread the needle on every individual leftover range.
        let Some(target_client_alphabet) = client_alphabet_for(&target.alphabet_name, &target.alphabet, client_version) else {
            continue;
        };

        // 1) Reuse this target's own oldest pending range this client can
        // search, if it has one - either a fresh chunk nobody's claimed yet,
        // or the unsearched remainder of a range whose previous claimant's
        // lease expired. A released range keeps its checkpointed
        // progress_index while it sits pending (see `release_range`), so its
        // previous claimant can still re-adopt it as-is - only now that it's
        // actually being handed out is its already-searched portion split
        // off as its own completed row, leaving just the unsearched
        // remainder to assign here.
        let pending = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE target_id = ? AND status = 'pending' ORDER BY created_at ASC")
            .bind(target.id)
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .find_map(|range| client_alphabet_for(&range.alphabet_name, &range.alphabet, client_version).map(str::to_string).map(|a| (range, a)));

        if let Some((range, client_alphabet)) = pending {
            let range = split_off_searched_portion(&mut tx, range, now).await?;
            let chunk = chunk_size(config, rate, &range.alphabet, &client_alphabet, range.candidate_len);
            let range = split_off_chunk(&mut tx, range, chunk).await?;
            let lease_seconds = config.lease_seconds;

            sqlx::query(
                "UPDATE ranges SET status = 'in_progress', assigned_user_id = ?, last_assigned_user_id = ?, \
                 assigned_at = ?, lease_seconds = ?, lease_expires_at = ?, last_progress_at = ? WHERE id = ?",
            )
            .bind(user.id)
            .bind(user.id)
            .bind(now)
            .bind(lease_seconds)
            .bind(now + lease_seconds)
            .bind(now)
            .bind(range.id)
            .execute(&mut *tx)
            .await?;

            tx.commit().await?;
            return Ok(Some(to_claim_response(
                &target,
                range.id,
                range.candidate_len,
                range.start(),
                range.end(),
                lease_seconds,
                &range.alphabet,
                &client_alphabet,
            )));
        }

        // 2) Otherwise, try this target's priority ranges (see
        // `models::PriorityRange`) before its own main sweep, highest
        // priority first - see `claim_priority_range_chunk`.
        if let Some(claim) = claim_priority_range_chunk(&mut tx, config, &target, user, client_version, rate, now).await? {
            tx.commit().await?;
            return Ok(Some(claim));
        }

        // 3) Otherwise carve a fresh chunk off this target's own cursor, if it still has room.
        //
        // Loops rather than a single pass because a skip run can sit right at
        // the cursor (nothing to hand out until it's stepped over), a length
        // can be exhausted (advancing the cursor into the next length, which
        // might itself start inside another skip run), or the cursor's
        // alphabet can still be an old one pending translation onto the
        // target's current alphabet (see the mismatch check below) - any of
        // these needs to reconsider the (now-advanced) cursor from scratch
        // rather than being resolved in one shot.
        'carve: loop {
            let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
                .bind(target.id)
                .fetch_one(&mut *tx)
                .await?;

            // The cursor might still be denominated in a previous alphabet if
            // this target was patched (handlers::admin_patch_target) since
            // carving last touched it. Every computation this iteration must
            // use the cursor's OWN (possibly old) alphabet, not the target's
            // current one, until the mismatch is resolved below.
            let active_alphabet_name = progress.alphabet_name.as_str();
            let active_alphabet = progress.alphabet.as_str();

            // The cursor is below the target's start_len - only possible
            // after a patch raised it (a new target's cursor starts right
            // at it). Jump straight to the start of start_len, leaving
            // everything shorter uncarved: no skipped rows, since priority
            // ranges at those lengths may already have carved parts of them
            // (the dashboard shows the jump as a gap instead). Landing at
            // the very start of a length also settles any pending alphabet
            // change for free - there's no old-alphabet position left to
            // translate, so it lands in the target's current alphabet
            // directly (admin_patch_target checks start_len fits in it).
            if progress.candidate_len < target.start_len {
                let start_index = bound_indices_at_len(&target.alphabet, &target.lower_bound, &target.upper_bound, target.start_len).0;
                persist_progress(&mut tx, target.id, target.start_len, start_index, &target.alphabet_name, &target.alphabet).await?;
                continue 'carve;
            }

            let next_index = progress.next();
            let (_, upper_at_len) = bound_indices_at_len(active_alphabet, &target.lower_bound, &target.upper_bound, progress.candidate_len);
            let remaining = upper_at_len - next_index + 1; // inclusive upper bound
            if remaining <= 0 {
                break 'carve; // this target's bounds are fully carved out at this length (and, if this is its last length, entirely) - try the next target
            }

            // Re-read every iteration rather than hoisted above the loop,
            // since the length and alphabet they're for can change mid-loop.
            let skip_spans = skip_spans_at(&mut tx, target.id, progress.candidate_len, active_alphabet_name, active_alphabet).await?;
            let skip_run = first_overlap(&skip_spans, next_index, upper_at_len + 1);

            // A priority range (see `models::PriorityRange`) permanently
            // excludes what it owns from the main sweep at its exact
            // candidate_len, without ever persisting anything for the gap -
            // see `claim_priority_range_chunk`'s doc comment.
            let priority_spans = priority_spans_at(&mut tx, target.id, progress.candidate_len, active_alphabet_name).await?;
            let priority_boundary = first_overlap(&priority_spans, next_index, upper_at_len + 1);

            // Whichever of the two starts first is this iteration's
            // boundary. Both are jumped straight over without writing
            // anything: a priority range carves its own, and a skip range's
            // `skipped` rows were written when it was created (see
            // `write_skipped_rows`).
            let boundary = [skip_run, priority_boundary].into_iter().flatten().min_by_key(|&(start, _, _)| start).map(|(start, end, _)| (start, end));

            // The cursor itself sits right at the boundary: nothing
            // claimable before it, so just step past it without handing
            // anything out this iteration.
            if let Some((b_start, b_end)) = boundary {
                if b_start == next_index {
                    let (new_len, new_next_index) =
                        bump_length_if_exhausted(active_alphabet, &target.lower_bound, &target.upper_bound, progress.candidate_len, b_end, upper_at_len);
                    persist_progress(&mut tx, target.id, new_len, new_next_index, active_alphabet_name, active_alphabet).await?;
                    continue 'carve;
                }
            }

            // The cursor is clear of any old-alphabet skip run; if the
            // target's alphabet has since been patched, this is the moment
            // to translate the cursor onto the new one - see
            // alphabet::transition_alphabet_cursor. Nothing already carved
            // (pending, in-progress, completed or skipped ranges) is
            // touched; only the cursor moves.
            if active_alphabet_name != target.alphabet_name {
                if progress.candidate_len > max_supported_len(&target.alphabet) {
                    // The new alphabet can't represent a candidate this long
                    // (a smaller alphabet has a *larger* max_supported_len,
                    // not smaller - so this only happens moving to a bigger
                    // alphabet after carving has already gone this deep).
                    // There's no length to bump to and nothing safe to
                    // translate, so this target simply has no more fresh
                    // work to hand out until an operator intervenes; ranges
                    // already carved are entirely unaffected.
                    break 'carve;
                }
                let transition = transition_alphabet_cursor(
                    active_alphabet,
                    &target.alphabet,
                    &target.lower_bound,
                    &target.upper_bound,
                    progress.candidate_len,
                    next_index,
                );
                if let Some((skip_start, skip_end)) = transition.skip {
                    insert_skipped_range(&mut tx, target.id, progress.candidate_len, skip_start, skip_end, active_alphabet_name, active_alphabet, None, None, now)
                        .await?;
                }
                let (_, new_upper_at_len) = bound_indices_at_len(&target.alphabet, &target.lower_bound, &target.upper_bound, progress.candidate_len);
                let (new_len, new_next_index) = bump_length_if_exhausted(
                    &target.alphabet,
                    &target.lower_bound,
                    &target.upper_bound,
                    progress.candidate_len,
                    transition.new_next_index,
                    new_upper_at_len,
                );
                persist_progress(&mut tx, target.id, new_len, new_next_index, &target.alphabet_name, &target.alphabet).await?;
                continue 'carve;
            }

            let natural_chunk = chunk_size(config, rate, &target.alphabet, target_client_alphabet, progress.candidate_len).min(remaining);
            let start_index = next_index;
            let natural_end = start_index + natural_chunk;

            // Only truncate at the boundary if the chunk that would
            // otherwise have been handed out actually reaches it - one
            // further out than what this claim would carve anyway is left
            // for a future claim to jump over.
            let (end_index, next_index_before_bump) = match boundary {
                Some((b_start, b_end)) if natural_end >= b_start => (b_start, b_end),
                _ => (natural_end, natural_end),
            };
            let (new_len, new_next_index) = bump_length_if_exhausted(
                &target.alphabet,
                &target.lower_bound,
                &target.upper_bound,
                progress.candidate_len,
                next_index_before_bump,
                upper_at_len,
            );
            persist_progress(&mut tx, target.id, new_len, new_next_index, &target.alphabet_name, &target.alphabet).await?;

            let lease_seconds = config.lease_seconds;
            let range_id = insert_claimed_range(
                &mut tx,
                target.id,
                progress.candidate_len,
                start_index,
                end_index,
                &target.alphabet_name,
                &target.alphabet,
                None,
                user,
                lease_seconds,
                now,
            )
            .await?;

            tx.commit().await?;
            return Ok(Some(to_claim_response(
                &target,
                range_id,
                progress.candidate_len,
                start_index,
                end_index,
                lease_seconds,
                &target.alphabet,
                target_client_alphabet,
            )));
        }
    }

    tx.commit().await?;
    Ok(None)
}

pub struct HeartbeatOutcome {
    pub lease_seconds: i64,
    pub range_released: bool,
}

pub async fn heartbeat_range(
    pool: &SqlitePool,
    config: &RangeConfig,
    user: &User,
    range_id: i64,
    last_hash_a_match_filename: Option<String>,
) -> Result<HeartbeatOutcome, AppError> {
    let mut tx = pool.begin().await?;

    let mut range = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE id = ?")
        .bind(range_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(AppError::NotFound)?;

    let target_status: String = sqlx::query_scalar("SELECT status FROM targets WHERE id = ?")
        .bind(range.target_id)
        .fetch_one(&mut *tx)
        .await?;
    let target_solved = target_status == "solved";

    let owns_range = range.status == "in_progress" && range.assigned_user_id == Some(user.id);
    // A client that lost its lease (a network outage outlasting it, say) but
    // kept searching regardless: if nobody else has claimed the range in the
    // meantime, it's still sitting pending exactly as this client left it
    // (see `release_range`), so hand it straight back rather than making the
    // client abort and someone else redo the work. Only for an active target
    // - the same work claim_range itself would be willing to hand out.
    let can_readopt = range.status == "pending" && range.last_assigned_user_id == Some(user.id) && target_status == "active";
    if !owns_range && !can_readopt {
        return Err(AppError::Conflict("range is not currently assigned to you".into()));
    }
    if can_readopt {
        let now = now_unix();
        sqlx::query("UPDATE ranges SET status = 'in_progress', assigned_user_id = ?, assigned_at = COALESCE(assigned_at, ?) WHERE id = ?")
            .bind(user.id)
            .bind(now)
            .bind(range_id)
            .execute(&mut *tx)
            .await?;
        range.status = "in_progress".into();
        range.assigned_user_id = Some(user.id);
        tracing::info!(range_id, user_id = user.id, "re-adopted a released range for its returning previous claimant");
    }

    // True only if this heartbeat's reported progress is genuinely new, not
    // just a re-report of the same match(es) already known about - a client
    // with nothing new to say (most notably, one that's paused and has
    // nothing new *to* report) will naturally repeat itself heartbeat after
    // heartbeat. This is the only signal heartbeat_range has for "is this
    // range actually still being worked" - see last_progress_at below.
    let mut made_progress = false;
    if let Some(filename) = last_hash_a_match_filename {
        if let Some(new_progress) = resolve_progress_index(&mut tx, &range, &filename).await? {
            // Monotonic: never let a late/out-of-order heartbeat move progress backwards.
            let floor = range.progress().unwrap_or(range.start() - 1);
            let new_progress = new_progress.max(floor);
            made_progress = new_progress > floor;
            let (progress_block, progress_index) = split_pos(&range.alphabet, range.candidate_len, new_progress);
            sqlx::query("UPDATE ranges SET progress_block = ?, progress_index = ? WHERE id = ?")
                .bind(progress_block)
                .bind(progress_index)
                .bind(range_id)
                .execute(&mut *tx)
                .await?;
        }
    }

    // The current setting, not the range's stored lease_seconds - that's
    // whatever was in force when it was claimed.
    let lease_seconds = config.lease_seconds;
    let now = now_unix();
    let mut range_released = false;
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
        range_released = true;
    } else {
        // Heartbeating alone would otherwise renew lease_expires_at forever,
        // holding this range hostage indefinitely for a client that's
        // stopped actually making progress on it (deliberately paused, or
        // otherwise stuck) - so a stall is capped at stall_release_seconds,
        // measured from the last heartbeat that reported genuinely new
        // progress (defaulting to when the range was claimed, if none ever
        // has).
        let last_progress_at = if made_progress { now } else { range.last_progress_at.unwrap_or(now) };
        if now - last_progress_at >= config.stall_release_seconds {
            release_range(&mut tx, &range, now).await?;
            range_released = true;
        } else {
            sqlx::query("UPDATE ranges SET lease_expires_at = ?, last_progress_at = ? WHERE id = ?")
                .bind(now + lease_seconds)
                .bind(last_progress_at)
                .bind(range_id)
                .execute(&mut *tx)
                .await?;
        }
    }

    tx.commit().await?;
    Ok(HeartbeatOutcome { lease_seconds, range_released })
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
) -> Result<Option<Pos>, AppError> {
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
    // A client searching the range in a bigger alphabet (see
    // alphabet::client_alphabet_for) can report a candidate with characters
    // the range's alphabet doesn't have.
    let known = |c: char| range.alphabet.contains(c) || PREDEFINED_ALPHABETS.iter().any(|(_, chars, _)| chars.contains(c));
    if !candidate.chars().all(known) {
        tracing::warn!(range_id = range.id, filename, "heartbeat match candidate has characters from no known alphabet, ignoring");
        return Ok(None);
    }
    // The range's own alphabet, not the target's current one - see
    // `models::Range::alphabet`. Rounded down, so everything up to it has
    // been searched even when the candidate itself isn't in that alphabet.
    let Some(index) = floor_index(&range.alphabet, candidate) else {
        tracing::warn!(range_id = range.id, filename, "heartbeat match candidate comes before every candidate of the range's alphabet, ignoring");
        return Ok(None);
    };
    if index < range.start() || index >= range.end() {
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

    // A `found: false` report from someone who no longer owns this range is
    // stale - whoever (or whatever) took it over already has the accurate
    // picture, so there's nothing worth keeping from it. A `found: true`
    // report is different: it's a fact about the *target*, not about this
    // range's bookkeeping, and it doesn't stop being true just because a
    // lease expired (a network blip outlasting the heartbeat interval, say)
    // before the client could report in - so it's never rejected just for
    // that. This doesn't meaningfully change what a malicious client could
    // already do: claiming a range and immediately reporting `found: true`
    // with a fabricated filename was always possible, since the server
    // never re-verifies a reported filename's hash - the ownership check
    // was only ever a staleness filter, not real authentication of the claim.
    //
    // A range released back to pending but not yet claimed by anyone else
    // still counts as this client's own, same as for heartbeat_range's
    // re-adoption: a client that lost its lease (a network outage, say) but
    // kept searching and finished offline reports in with /complete first,
    // never getting a heartbeat in to re-adopt it. Unlike re-adoption, the
    // target doesn't need to still be active - the work is already done, so
    // there's nothing left to hand out, only a result to record.
    let owns_range = (range.status == "in_progress" && range.assigned_user_id == Some(user.id))
        || (range.status == "pending" && range.last_assigned_user_id == Some(user.id));
    if !owns_range && !found {
        return Err(AppError::Conflict("range is not currently assigned to you".into()));
    }

    let now = now_unix();
    // A canary's range: its result is recorded, and that's all - its
    // claimant's rate isn't updated (it stops at the planted name), and its
    // virtual target solves nothing else.
    if owns_range && crate::canary::complete(&mut tx, &range, found, filename.as_deref(), now).await? {
        sqlx::query("UPDATE ranges SET status = 'completed', completed_at = ?, assigned_user_id = ? WHERE id = ?")
            .bind(now)
            .bind(user.id)
            .bind(range_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        return Ok(CompleteOutcome { target_solved: false });
    }
    if owns_range {
        // assigned_user_id is only for the pending case, where it was cleared
        // on release - for an in_progress range it's already this user.
        sqlx::query("UPDATE ranges SET status = 'completed', completed_at = ?, assigned_user_id = ? WHERE id = ?")
            .bind(now)
            .bind(user.id)
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
    }

    // Different targets can share the same hash_a/hash_b (e.g. the same
    // underlying file cataloged under more than one naming convention) - a
    // match against one of them is, by construction, a match against all of
    // them, so every other still-active target with the same hash pair is
    // solved right alongside this one, with the same found_filename/finder
    // credited (it's the same real file either way). Each of those targets'
    // own in-progress ranges get closed out lazily, the same way this one's
    // sibling ranges already are - see heartbeat_range's `target_solved` check.
    let mut target_solved = false;
    if found {
        let result = sqlx::query(
            "UPDATE targets SET status = 'solved', found_filename = ?, found_by_user_id = ?, found_at = ? \
             WHERE status = 'active' AND is_virtual = 0 \
             AND hash_a = (SELECT hash_a FROM targets WHERE id = ?) \
             AND hash_b = (SELECT hash_b FROM targets WHERE id = ?)",
        )
        .bind(&filename)
        .bind(user.id)
        .bind(now)
        .bind(range.target_id)
        .bind(range.target_id)
        .execute(&mut *tx)
        .await?;
        target_solved = result.rows_affected() > 0;
    }

    tx.commit().await?;
    Ok(CompleteOutcome { target_solved })
}

/// Releases any range whose lease expired while still `in_progress`, one at a
/// time (each in its own transaction, so a concurrent `/claim` can never see a
/// half-updated row - it only ever sees the fully-resolved end state, exactly
/// as if this had run instantaneously). See `release_range` for what releasing
/// means.
pub async fn reclaim_expired(pool: &SqlitePool) -> Result<u64, sqlx::Error> {
    let now = now_unix();
    let mut resolved = 0u64;

    loop {
        let mut tx = pool.begin().await?;

        let range = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE status = 'in_progress' AND lease_expires_at < ? LIMIT 1")
            .bind(now)
            .fetch_optional(&mut *tx)
            .await?;

        let Some(range) = range else {
            tx.commit().await?;
            break;
        };

        release_range(&mut tx, &range, now).await?;
        tx.commit().await?;
        resolved += 1;
    }

    Ok(resolved)
}

/// Releases `range` back to the pool:
///   - fully searched already (checkpoint reached the end, no match ever
///     reported) -> close it out, nothing left to hand out;
///   - otherwise -> make it `pending` again, as a single row, still carrying
///     its `progress_index` and `last_assigned_user_id`.
///
/// The already-searched portion (if any) is deliberately *not* split off
/// here: until someone else claims the range, its previous claimant may yet
/// come back (a client that lost connectivity but kept searching) and
/// re-adopt it under the very same range id it's still heartbeating - see
/// `heartbeat_range`. `claim_range` does the split instead, once the range
/// is actually handed to someone - see `split_off_searched_portion`.
///
/// Shared by reclaim_expired (lease actually expired) and heartbeat_range
/// (client stalled too long - see STALL_RELEASE_SECONDS) - both mean the same
/// thing to this range: whoever holds it isn't making progress, so hand
/// whatever's left back to the pool.
async fn release_range(tx: &mut sqlx::SqliteConnection, range: &Range, now: i64) -> Result<(), sqlx::Error> {
    let fully_searched = matches!(range.progress(), Some(p) if p + 1 >= range.end());

    if fully_searched {
        sqlx::query(
            "UPDATE ranges SET status = 'completed', completed_at = ?, \
             assigned_user_id = NULL, lease_expires_at = NULL, last_progress_at = NULL WHERE id = ?",
        )
        .bind(now)
        .bind(range.id)
        .execute(&mut *tx)
        .await?;
    } else {
        // assigned_at is kept alongside a checkpoint, since it's still when
        // the searched portion's work began (split_off_searched_portion's
        // completed row keeps it) - but with no checkpoint there's nothing
        // it describes any more.
        sqlx::query(
            "UPDATE ranges SET status = 'pending', assigned_user_id = NULL, \
             assigned_at = CASE WHEN progress_index IS NULL THEN NULL ELSE assigned_at END, \
             lease_expires_at = NULL, last_progress_at = NULL WHERE id = ?",
        )
        .bind(range.id)
        .execute(&mut *tx)
        .await?;
    }

    Ok(())
}

/// Called by `claim_range` on the pending row it's about to hand out. A row
/// released with real progress (see `release_range`) still spans its
/// already-searched portion - this finalizes that portion as its own
/// `completed` row (the original row, shrunk, still credited to whoever
/// actually searched it via `last_assigned_user_id`) and returns a fresh
/// `pending` row for just the unsearched remainder. A row without a
/// checkpoint is returned unchanged.
async fn split_off_searched_portion(tx: &mut sqlx::SqliteConnection, range: Range, now: i64) -> Result<Range, sqlx::Error> {
    let effective_start = match range.progress() {
        Some(p) if p + 1 > range.start() => p + 1,
        _ => return Ok(range),
    };
    let (completed_end_block, completed_end) = split_end(&range.alphabet, range.candidate_len, effective_start);
    let (effective_start_block, effective_start) = split_pos(&range.alphabet, range.candidate_len, effective_start);

    sqlx::query(
        "UPDATE ranges SET end_block = ?, end_index = ?, status = 'completed', completed_at = ?, progress_block = 0, progress_index = NULL WHERE id = ?",
    )
    .bind(completed_end_block)
    .bind(completed_end)
    .bind(now)
    .bind(range.id)
    .execute(&mut *tx)
    .await?;

    // Carries the original range's own alphabet and priority-range origin
    // forward, not the target's current alphabet - the target's alphabet may
    // have been patched since this range was originally carved (see
    // `models::Range::alphabet`), and this remainder is still the same
    // priority range's work if the original was (see
    // `models::Range::priority_range_id`).
    let remainder = sqlx::query_as::<_, Range>(
        "INSERT INTO ranges (target_id, candidate_len, start_block, start_index, end_block, end_index, status, \
         assigned_user_id, last_assigned_user_id, assigned_at, lease_seconds, lease_expires_at, created_at, alphabet_name, alphabet, priority_range_id) \
         VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?) RETURNING *",
    )
    .bind(range.target_id)
    .bind(range.candidate_len)
    .bind(effective_start_block)
    .bind(effective_start)
    .bind(range.end_block)
    .bind(range.end_index)
    .bind(now)
    .bind(&range.alphabet_name)
    .bind(&range.alphabet)
    .bind(range.priority_range_id)
    .fetch_one(&mut *tx)
    .await?;

    Ok(remainder)
}

/// Called by `claim_range` on the pending row it's about to hand out, after
/// `split_off_searched_portion`. A row of more than `chunk` candidates - such
/// as one a removed skip range requeued, see `remove_skip_range` - is shrunk
/// to its first `chunk`, and the rest stays pending as a row of its own,
/// keeping its place in the queue. So a client gets about as much work as a
/// freshly carved chunk, not the whole row at once.
async fn split_off_chunk(tx: &mut sqlx::SqliteConnection, mut range: Range, chunk: Pos) -> Result<Range, sqlx::Error> {
    let split = range.start() + chunk;
    if split >= range.end() {
        return Ok(range);
    }
    insert_pending_range(&mut *tx, range.target_id, range.candidate_len, split, range.end(), &range.alphabet_name, &range.alphabet, range.priority_range_id, range.created_at)
        .await?;
    let (end_block, end_index) = split_end(&range.alphabet, range.candidate_len, split);
    sqlx::query("UPDATE ranges SET end_block = ?, end_index = ? WHERE id = ?").bind(end_block).bind(end_index).bind(range.id).execute(&mut *tx).await?;
    range.end_block = end_block;
    range.end_index = end_index;
    Ok(range)
}

/// Permanently removes a target and everything carved for it (its ranges and
/// carving cursor). SQLite doesn't enforce the `REFERENCES` foreign keys here
/// (no `PRAGMA foreign_keys = ON` is set), so children have to be deleted
/// explicitly rather than relying on a cascade. Returns whether a target with
/// this id actually existed.
pub async fn delete_target(pool: &SqlitePool, target_id: i64) -> Result<bool, AppError> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM ranges WHERE target_id = ?").bind(target_id).execute(&mut *tx).await?;
    sqlx::query("DELETE FROM target_progress WHERE target_id = ?").bind(target_id).execute(&mut *tx).await?;
    sqlx::query("DELETE FROM priority_range_segments WHERE priority_range_id IN (SELECT id FROM priority_ranges WHERE target_id = ?)")
        .bind(target_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM priority_ranges WHERE target_id = ?").bind(target_id).execute(&mut *tx).await?;
    sqlx::query("DELETE FROM skip_range_segments WHERE skip_range_id IN (SELECT id FROM skip_ranges WHERE target_id = ?)")
        .bind(target_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM skip_ranges WHERE target_id = ?").bind(target_id).execute(&mut *tx).await?;
    let result = sqlx::query("DELETE FROM targets WHERE id = ?").bind(target_id).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(result.rows_affected() > 0)
}

/// Where a new priority range owning `spans` (sorted, see
/// `alphabet::pattern_spans`) should start handing out work (its
/// `next_index`), given the target's existing priority ranges at the same
/// length and alphabet. Each existing one permanently owns its spans (see
/// `priority_spans_at`), so overlapping one is normally rejected - except
/// that work a finished priority range already handed out (everything it
/// owns, once `next_index == end_index`) can be skipped when it sits at the
/// new range's start: the new range simply starts right after it, the same
/// way `handlers::admin_create_priority_range` starts one after whatever the
/// main sweep has already searched. `next_index` is where it would start
/// otherwise (its first span's start, or later if the main sweep is already
/// inside it).
///
/// Rejected: overlapping a priority range that's still handing out work, or
/// finished work that isn't at the new range's start (it can't skip a block
/// in its middle or at its end), or a range with nothing left once skipped.
pub async fn priority_range_start(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    candidate_len: i64,
    alphabet_name: &str,
    spans: &[(Pos, Pos)],
    mut next_index: Pos,
) -> Result<Pos, AppError> {
    // Overlap is checked here rather than in SQL, since a position is only
    // comparable once its block and index are joined.
    let priority_ranges =
        sqlx::query_as::<_, PriorityRange>("SELECT * FROM priority_ranges WHERE target_id = ? AND candidate_len = ? AND alphabet_name = ?")
            .bind(target_id)
            .bind(candidate_len)
            .bind(alphabet_name)
            .fetch_all(&mut *tx)
            .await?;
    let segments = priority_segments_by_range(tx, target_id).await?;

    let describe = |pr: &PriorityRange| {
        let first = index_to_candidate(&pr.alphabet, pr.start(), pr.candidate_len);
        let last = index_to_candidate(&pr.alphabet, pr.end() - 1, pr.candidate_len);
        format!("priority range #{} ('{}', '{first}' to '{last}')", pr.id, pr.pattern)
    };

    // What finished priority ranges handed out, tagged with who.
    let mut finished: Vec<(Pos, Pos, &PriorityRange)> = Vec::new();
    for pr in &priority_ranges {
        let owned = owned_spans(pr, segments.get(&pr.id));
        if !spans_overlap(&owned, spans) {
            continue;
        }
        if pr.next() < pr.end() {
            return Err(AppError::BadRequest(format!("overlaps {}, which is still handing out work", describe(pr))));
        }
        finished.extend(owned.into_iter().map(|(start, end)| (start, end, pr)));
    }

    // Step over finished work for as long as it's right where the new
    // range's own remaining work starts.
    loop {
        let Some(at) = spans.iter().find(|&&(_, end)| end > next_index).map(|&(start, _)| start.max(next_index)) else {
            return Err(AppError::BadRequest("already fully handed out by earlier priority ranges - nothing left to prioritize".into()));
        };
        next_index = at;
        match finished.iter().find(|&&(start, end, _)| start <= at && at < end) {
            Some(&(_, end, _)) => next_index = end,
            None => break,
        }
    }

    let remaining: Vec<(Pos, Pos)> =
        spans.iter().map(|&(start, end)| (start.max(next_index), end)).filter(|&(start, end)| start < end).collect();
    if let Some(&(_, _, pr)) = finished.iter().find(|&&(start, end, _)| spans_overlap(&remaining, &[(start, end)])) {
        return Err(AppError::BadRequest(format!(
            "overlaps work {} already handed out - a new priority range can only skip such work at its start",
            describe(pr)
        )));
    }
    Ok(next_index)
}

/// The outcome of removing a priority range - see `remove_priority_range`.
pub struct PriorityRangeRemoval {
    /// The row itself is gone: nothing was ever carved from it, and nothing
    /// of it had to be kept.
    pub deleted: bool,
    /// Candidates handed back to the target's main sweep, which will search
    /// them when it gets there.
    pub returned_to_main_sweep: Pos,
    /// Candidates the main sweep has already jumped past (it never goes
    /// back), so they stay with this priority range - still handed out
    /// ahead of the main sweep - rather than never being searched at all.
    pub kept: Pos,
}

/// How far into `pr`'s span the target's main sweep has already gone past.
/// The sweep jumps straight over a priority range's span (see
/// `priority_spans_at`) and never goes back, so anything of `pr`'s
/// span before the returned index that `pr` hasn't carved yet can only ever
/// be searched by `pr` itself. Never less than `pr.next_index` (what `pr`
/// has already carved is accounted for either way). A cursor walking in a
/// different alphabet isn't index-comparable to `pr`, so it's treated as
/// having passed all of it - keeping work prioritized is always safe,
/// giving it back is only safe when the sweep will really reach it.
fn main_sweep_passed_to(progress: &TargetProgress, pr: &PriorityRange) -> Pos {
    let carved_to = pr.next().min(pr.end());
    if progress.alphabet_name != pr.alphabet_name || progress.candidate_len > pr.candidate_len {
        pr.end()
    } else if progress.candidate_len < pr.candidate_len {
        carved_to
    } else {
        progress.next().clamp(carved_to, pr.end())
    }
}

/// Removes an admin's priority-range hint, as far as that can be done
/// without leaving candidates unsearched:
///   - what it has already carved stays in its span, so the main sweep
///     keeps jumping over (and so doesn't duplicate) that work;
///   - of the rest, whatever the main sweep hasn't reached yet is handed
///     back to it, by shrinking the span;
///   - whatever the main sweep has already jumped past stays in the span
///     and keeps being handed out as priority work - see
///     `main_sweep_passed_to`.
/// A priority range that never carved anything and keeps nothing is deleted
/// outright. Returns `None` if no such row exists.
///
/// Deliberately checks for an actual referencing `ranges` row rather than
/// comparing `next_index` to `start_index` to decide on deletion: a
/// same-length priority range can have its `next_index` clamped ahead of
/// `start_index` at creation time (see `handlers::admin_create_priority_range`)
/// without anything having been carved from it yet.
pub async fn remove_priority_range(pool: &SqlitePool, priority_range_id: i64) -> Result<Option<PriorityRangeRemoval>, AppError> {
    // One transaction, so a concurrent `/claim` can't carve a chunk (moving
    // next_index, and creating the very `ranges` row checked for below)
    // between reading this row and shrinking or deleting it.
    let mut tx = pool.begin().await?;

    let Some(pr) = sqlx::query_as::<_, PriorityRange>("SELECT * FROM priority_ranges WHERE id = ?")
        .bind(priority_range_id)
        .fetch_optional(&mut *tx)
        .await?
    else {
        return Ok(None);
    };
    let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
        .bind(pr.target_id)
        .fetch_one(&mut *tx)
        .await?;

    let new_end = main_sweep_passed_to(&progress, &pr);
    let owned = owned_spans(&pr, priority_segments_by_range(&mut tx, pr.target_id).await?.get(&pr.id));
    let removal = PriorityRangeRemoval {
        deleted: false,
        returned_to_main_sweep: count_within(&owned, new_end, pr.end()),
        kept: count_within(&owned, pr.next().min(pr.end()), new_end),
    };

    let ever_carved: Option<(i64,)> =
        sqlx::query_as("SELECT 1 FROM ranges WHERE priority_range_id = ? LIMIT 1").bind(priority_range_id).fetch_optional(&mut *tx).await?;

    let removal = if ever_carved.is_none() && removal.kept == 0 {
        sqlx::query("DELETE FROM priority_range_segments WHERE priority_range_id = ?").bind(priority_range_id).execute(&mut *tx).await?;
        sqlx::query("DELETE FROM priority_ranges WHERE id = ?").bind(priority_range_id).execute(&mut *tx).await?;
        PriorityRangeRemoval { deleted: true, ..removal }
    } else {
        let (end_block, end_index) = split_end(&pr.alphabet, pr.candidate_len, new_end);
        sqlx::query("UPDATE priority_ranges SET end_block = ?, end_index = ? WHERE id = ?")
            .bind(end_block)
            .bind(end_index)
            .bind(priority_range_id)
            .execute(&mut *tx)
            .await?;
        removal
    };

    tx.commit().await?;
    Ok(Some(removal))
}

/// Translates every one of `target_id`'s priority ranges still frozen under
/// a stale alphabet onto `new_alphabet_name`/`new_alphabet` - called eagerly,
/// as part of the same transaction `handlers::admin_patch_target` uses to
/// apply an alphabet change. Unlike the target's own main cursor (see
/// `transition_alphabet_cursor`, applied lazily the next time carving
/// reaches it), this can't wait: `priority_spans_at` re-checks every
/// priority range on *every* claim, so leaving one stale would leave the
/// main sweep unable to exclude it for as long as it stays untouched -
/// possibly indefinitely, if a higher-priority range keeps winning the
/// queue ahead of it. Doing all of them up front is cheap given how few of
/// these exist per target in practice.
///
/// Its pattern is expanded afresh under the new alphabet - or, for a row
/// from before `migrations/0021_skip_ranges_and_pattern_segments.sql`, its
/// one literal prefix. A row from before `0012_priority_range_prefix.sql`
/// (`prefix` is `""`) is left exactly as it was - there's nothing to safely
/// translate it with, so it keeps the same limitation this function
/// otherwise closes.
pub async fn migrate_priority_ranges_to_new_alphabet(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    new_alphabet_name: &str,
    new_alphabet: &str,
    now: i64,
) -> Result<(), AppError> {
    let stale = sqlx::query_as::<_, PriorityRange>("SELECT * FROM priority_ranges WHERE target_id = ? AND alphabet_name != ?")
        .bind(target_id)
        .bind(new_alphabet_name)
        .fetch_all(&mut *tx)
        .await?;
    let segments = priority_segments_by_range(tx, target_id).await?;
    let (lower_bound, upper_bound): (String, String) =
        sqlx::query_as("SELECT lower_bound, upper_bound FROM targets WHERE id = ?").bind(target_id).fetch_one(&mut *tx).await?;

    for pr in stale {
        if pr.prefix.as_deref() == Some("") {
            continue;
        }

        let new_spans = if pr.candidate_len > max_supported_len(new_alphabet) {
            None
        } else if let Some(prefix) = pr.prefix.as_deref() {
            prefix.chars().all(|c| candidate_to_index(new_alphabet, &c.to_string()).is_some()).then(|| {
                let (start, end_inclusive) = bound_indices_at_len(new_alphabet, prefix, prefix, pr.candidate_len);
                vec![(start, end_inclusive + 1)]
            })
        } else {
            pattern_spans(new_alphabet, &pr.pattern, pr.candidate_len).ok()
        };
        let new_spans = new_spans
            .map(|spans| clip_spans_to_bounds(new_alphabet, &lower_bound, &upper_bound, pr.candidate_len, &spans))
            .filter(|spans| !spans.is_empty());

        // Nothing it matches can ever be produced by the new alphabet's own
        // walk - the length itself no longer fits, or the pattern needs a
        // character the new alphabet dropped - so there's no exclusion
        // needed for it going forward either; just stop offering it as
        // fresh work and leave its bounds as a historical record under the
        // alphabet it actually holds candidates in. (Also when the pattern
        // would now take too many spans - the main sweep then searches
        // those candidates itself - and when none of what it matches lies
        // within the target's bounds.)
        let Some(new_spans) = new_spans else {
            sqlx::query("UPDATE priority_ranges SET next_block = end_block, next_index = end_index WHERE id = ?").bind(pr.id).execute(&mut *tx).await?;
            continue;
        };

        // Candidate strings bounding this row's own span, which
        // transition_alphabet_cursor clips a skip to.
        let first = index_to_candidate(&pr.alphabet, pr.start(), pr.candidate_len);
        let last = index_to_candidate(&pr.alphabet, pr.end() - 1, pr.candidate_len);

        let new_start = new_spans[0].0;
        let spans_end = new_spans[new_spans.len() - 1].1;
        let old_spans_end = segments.get(&pr.id).into_iter().flatten().map(|s| s.span(&pr.alphabet, pr.candidate_len).1).max().unwrap_or(pr.end());
        let new_end = if pr.end() >= old_spans_end {
            spans_end
        } else {
            // Pulled in by remove_priority_range: keep it at the same place,
            // the first candidate at or after it that the new alphabet has.
            transition_alphabet_cursor(&pr.alphabet, new_alphabet, &first, &last, pr.candidate_len, pr.end()).new_next_index.clamp(new_start, spans_end)
        };

        let new_next_index = if pr.next() >= pr.end() {
            // Already fully exhausted under the old alphabet - nothing to
            // resume, just carry the "done" state over to the new bounds.
            new_end
        } else {
            // Scoped to this priority range's own span rather than the
            // target's overall bounds, so the skip this can produce is
            // clipped to this row's own block, not the whole target's space -
            // see transition_alphabet_cursor's own doc comment for the
            // general algorithm.
            let transition = transition_alphabet_cursor(&pr.alphabet, new_alphabet, &first, &last, pr.candidate_len, pr.next());
            if let Some((skip_start, skip_end)) = transition.skip {
                insert_skipped_range(&mut *tx, target_id, pr.candidate_len, skip_start, skip_end, &pr.alphabet_name, &pr.alphabet, Some(pr.id), None, now)
                    .await?;
            }
            // A priority range has no next length to bump to (unlike the
            // main cursor) - if transition_alphabet_cursor reports nothing
            // representable left at all, clamp down to new_end so next_index
            // stays within its usual [start_index, end_index] invariant
            // instead of storing transition_alphabet_cursor's raw sentinel.
            transition.new_next_index.min(new_end)
        };

        let (start_block, new_start) = split_pos(new_alphabet, pr.candidate_len, new_start);
        let (end_block, new_end) = split_end(new_alphabet, pr.candidate_len, new_end);
        let (next_block, new_next_index) = split_end(new_alphabet, pr.candidate_len, new_next_index);
        sqlx::query(
            "UPDATE priority_ranges SET start_block = ?, start_index = ?, end_block = ?, end_index = ?, next_block = ?, next_index = ?, \
             alphabet_name = ?, alphabet = ? WHERE id = ?",
        )
            .bind(start_block)
            .bind(new_start)
            .bind(end_block)
            .bind(new_end)
            .bind(next_block)
            .bind(new_next_index)
            .bind(new_alphabet_name)
            .bind(new_alphabet)
            .bind(pr.id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM priority_range_segments WHERE priority_range_id = ?").bind(pr.id).execute(&mut *tx).await?;
        insert_segments(tx, SegmentOwner::Priority, pr.id, new_alphabet, pr.candidate_len, &new_spans).await?;
    }

    Ok(())
}

/// Creates a skip range on `target`: every candidate of length
/// `candidate_len` matching `pattern` (see `alphabet::pattern_spans`) is
/// left out of the search, recorded right away as `skipped` ranges for
/// every part no range covers yet (see `write_skipped_rows`). Anything
/// already carved stays as it is. Returns the new skip range's id.
pub async fn create_skip_range(
    tx: &mut sqlx::SqliteConnection,
    target: &Target,
    pattern: &str,
    candidate_len: i64,
    reason: &str,
    now: i64,
) -> Result<i64, AppError> {
    let cap = max_supported_len(&target.alphabet);
    if candidate_len <= 0 || candidate_len > cap {
        return Err(AppError::BadRequest(format!("length must be between 1 and this target's alphabet's max supported length ({cap})")));
    }
    if reason.trim().is_empty() {
        return Err(AppError::BadRequest("reason must not be empty".into()));
    }
    let spans = pattern_spans(&target.alphabet, pattern, candidate_len).map_err(AppError::BadRequest)?;

    let id: i64 = sqlx::query_scalar(
        "INSERT INTO skip_ranges (target_id, pattern, reason, candidate_len, alphabet_name, alphabet, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(target.id)
    .bind(pattern)
    .bind(reason)
    .bind(candidate_len)
    .bind(&target.alphabet_name)
    .bind(&target.alphabet)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;
    insert_segments(tx, SegmentOwner::Skip, id, &target.alphabet, candidate_len, &spans).await?;
    write_skipped_rows(tx, target, id, candidate_len, &target.alphabet_name, &target.alphabet, &spans, now).await?;
    Ok(id)
}

/// Removes a skip range, so nothing it matches stays skipped. Its `skipped`
/// rows that carving will still reach are removed, so carving searches
/// them. The ones carving has already passed (see `still_to_be_reached`),
/// which it never goes back to, become `pending` rows instead, which
/// `claim_range` hands out ahead of fresh carving - in chunks, see
/// `split_off_chunk`. The skip range itself is deleted - also one removed
/// before this worked this way, whose passed rows then get searched too.
/// Returns how many candidates were requeued as pending, or `None` if
/// there's no such skip range.
pub async fn remove_skip_range(pool: &SqlitePool, skip_range_id: i64) -> Result<Option<Pos>, AppError> {
    let mut tx = pool.begin().await?;
    let Some(skip_range) = sqlx::query_as::<_, SkipRange>("SELECT * FROM skip_ranges WHERE id = ?").bind(skip_range_id).fetch_optional(&mut *tx).await?
    else {
        return Ok(None);
    };
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(skip_range.target_id).fetch_one(&mut *tx).await?;
    sqlx::query("DELETE FROM skip_range_segments WHERE skip_range_id = ?").bind(skip_range_id).execute(&mut *tx).await?;

    let rows = sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE skip_range_id = ? AND status = 'skipped'")
        .bind(skip_range_id)
        .fetch_all(&mut *tx)
        .await?;
    let now = now_unix();
    let mut requeued: Pos = 0;
    for row in rows {
        let span = [(row.start(), row.end())];
        let reached = still_to_be_reached(&mut tx, &target, row.candidate_len, &row.alphabet_name, &row.alphabet, &span).await?;
        sqlx::query("DELETE FROM ranges WHERE id = ?").bind(row.id).execute(&mut *tx).await?;
        for (start, end) in subtract_spans(&span, &reached) {
            insert_pending_range(&mut tx, target.id, row.candidate_len, start, end, &row.alphabet_name, &row.alphabet, row.priority_range_id, now).await?;
            requeued += end - start;
        }
    }

    sqlx::query("DELETE FROM skip_ranges WHERE id = ?").bind(skip_range_id).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(Some(requeued))
}

/// Where the main sweep stands, as a position in `alphabet` - which needn't
/// be the one its cursor is in (an alphabet change it hasn't caught up with
/// yet): it has passed every shorter length, and everything before the
/// returned position at the returned length. Its cursor carries over to a
/// new alphabet as the first of its candidates at or after the cursor's
/// own (see `alphabet::transition_alphabet_cursor`), which is what
/// `ceil_index` gives.
async fn sweep_position_in(tx: &mut sqlx::SqliteConnection, target: &Target, alphabet: &str) -> Result<(i64, Pos), AppError> {
    let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
        .bind(target.id)
        .fetch_one(&mut *tx)
        .await?;
    // Mirrors claim_range: a sweep below start_len jumps to the start of start_len.
    if progress.candidate_len < target.start_len {
        return Ok((target.start_len, bound_indices_at_len(alphabet, &target.lower_bound, &target.upper_bound, target.start_len).0));
    }
    let len = progress.candidate_len;
    if len > max_supported_len(alphabet) {
        return Ok((len, 0));
    }
    if progress.next() >= space_size(&progress.alphabet, len) {
        return Ok((len, space_size(alphabet, len)));
    }
    let cursor = index_to_candidate(&progress.alphabet, progress.next(), len);
    Ok((len, ceil_index(alphabet, &cursor).unwrap_or_else(|| space_size(alphabet, len))))
}

/// Re-expands every one of `target_id`'s skip ranges onto a new alphabet -
/// called by `handlers::admin_patch_target` in the same transaction as the
/// alphabet change, for the same reason as
/// `migrate_priority_ranges_to_new_alphabet`. A skip range has no cursor of
/// its own, so its pattern is simply matched afresh: its old-alphabet
/// `skipped` rows carving won't pass any more are removed (the ones already
/// passed stay, see `still_to_be_reached`), and new ones are written in the
/// new alphabet - only where the sweep hasn't been yet, since what it has
/// passed was already searched or skipped in the old one. One whose pattern
/// matches nothing in the new alphabet (or whose length it can't reach) is
/// left with no segments.
///
/// Must run before `migrate_priority_ranges_to_new_alphabet`, while the
/// priority ranges are still in the old alphabet the rows are compared
/// against.
pub async fn migrate_skip_ranges_to_new_alphabet(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    new_alphabet_name: &str,
    new_alphabet: &str,
    now: i64,
) -> Result<(), AppError> {
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(&mut *tx).await?;
    let stale = sqlx::query_as::<_, SkipRange>("SELECT * FROM skip_ranges WHERE target_id = ? AND alphabet_name != ? AND removed_at IS NULL")
        .bind(target_id)
        .bind(new_alphabet_name)
        .fetch_all(&mut *tx)
        .await?;
    let (sweep_len, sweep_at) = sweep_position_in(tx, &target, new_alphabet).await?;
    for skip_range in stale {
        sqlx::query("DELETE FROM skip_range_segments WHERE skip_range_id = ?").bind(skip_range.id).execute(&mut *tx).await?;
        unskip_unreached_rows(tx, &target, skip_range.id, Some(&skip_range.alphabet_name)).await?;
        let len = skip_range.candidate_len;
        if len <= max_supported_len(new_alphabet) {
            if let Ok(spans) = pattern_spans(new_alphabet, &skip_range.pattern, len) {
                insert_segments(tx, SegmentOwner::Skip, skip_range.id, new_alphabet, len, &spans).await?;
                let ahead = match len.cmp(&sweep_len) {
                    std::cmp::Ordering::Less => vec![],
                    std::cmp::Ordering::Equal => intersect_spans(&spans, &[(sweep_at, space_size(new_alphabet, len))]),
                    std::cmp::Ordering::Greater => spans,
                };
                write_skipped_rows(tx, &target, skip_range.id, len, new_alphabet_name, new_alphabet, &ahead, now).await?;
            }
        }
        sqlx::query("UPDATE skip_ranges SET alphabet_name = ?, alphabet = ? WHERE id = ?")
            .bind(new_alphabet_name)
            .bind(new_alphabet)
            .bind(skip_range.id)
            .execute(&mut *tx)
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{index_width, max_supported_len, Pos};

    // `i64` versions of the alphabet's position functions, which is what the
    // tests below work in: they stay at lengths up to `index_width`, where a
    // position is block 0 and so equal to its stored index. Tests of longer
    // lengths use `crate::alphabet`'s own functions directly.
    fn space_size(alphabet: &str, len: i64) -> i64 {
        i64::try_from(crate::alphabet::space_size(alphabet, len)).expect("tests here stay within index_width")
    }

    fn candidate_to_index(alphabet: &str, candidate: &str) -> Option<i64> {
        crate::alphabet::candidate_to_index(alphabet, candidate).map(|p| i64::try_from(p).expect("tests here stay within index_width"))
    }

    fn index_to_candidate(alphabet: &str, index: i64, len: i64) -> String {
        crate::alphabet::index_to_candidate(alphabet, index as Pos, len)
    }

    fn range_bound_filenames(alphabet: &str, prefix: &str, suffix: &str, len: i64, start_index: i64, end_index: i64) -> (String, String) {
        crate::alphabet::range_bound_filenames(alphabet, prefix, suffix, len, start_index as Pos, end_index as Pos)
    }

    fn bound_indices_at_len(alphabet: &str, lower_bound: &str, upper_bound: &str, len: i64) -> (i64, i64) {
        let (lo, hi) = crate::alphabet::bound_indices_at_len(alphabet, lower_bound, upper_bound, len);
        (i64::try_from(lo).expect("tests here stay within index_width"), i64::try_from(hi).expect("tests here stay within index_width"))
    }

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
            lease_seconds: 6 * 60 * 60,
            reclaim_interval_secs: 30,
            ema_alpha: 0.3,
            canary_probability: 0.0,
            canary_seconds: 5.0,
            stall_release_seconds: 24 * 60 * 60,
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
        User {
            id,
            username: name.into(),
            hostname: format!("{name}-host"),
            token,
            ema_rate_per_sec: None,
            created_at: now,
            last_seen_at: now,
            protocol_version: "1.0.0".to_string(),
            backend: String::new(),
            client_release: None,
        }
    }

    /// Like `insert_user`, but with an explicit `protocol_version` instead
    /// of the default `"1.0.0"` - for tests about `claim_range`'s
    /// alphabet-version gating.
    async fn insert_user_with_protocol_version(pool: &SqlitePool, name: &str, protocol_version: &str) -> User {
        let now = now_unix();
        let token = format!("{name}-tok");
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO users (username, hostname, token, created_at, last_seen_at, protocol_version) VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(name)
        .bind(format!("{name}-host"))
        .bind(&token)
        .bind(now)
        .bind(now)
        .bind(protocol_version)
        .fetch_one(pool)
        .await
        .unwrap();
        User {
            id,
            username: name.into(),
            hostname: format!("{name}-host"),
            token,
            ema_rate_per_sec: None,
            created_at: now,
            last_seen_at: now,
            protocol_version: protocol_version.to_string(),
            backend: String::new(),
            client_release: None,
        }
    }

    /// Bounds spanning the *entire* space at a single fixed length - the
    /// simplest possible bounds, for tests that just want "the whole space at
    /// length N" with no interest in the bound-tightening behavior itself.
    fn full_bounds(alphabet: &str, len: i64) -> (String, String) {
        let min_char = alphabet.chars().next().unwrap();
        let max_char = alphabet.chars().last().unwrap();
        (min_char.to_string().repeat(len as usize), max_char.to_string().repeat(len as usize))
    }

    /// Creates a target bounded by `[lower_bound, upper_bound]` (candidate
    /// strings, already stripped of prefix/suffix).
    async fn insert_target(pool: &SqlitePool, lower_bound: &str, upper_bound: &str) -> i64 {
        insert_target_with_alphabet(pool, "size49", DEFAULT, lower_bound, upper_bound).await
    }

    async fn insert_target_with_alphabet(pool: &SqlitePool, alphabet_name: &str, alphabet: &str, lower_bound: &str, upper_bound: &str) -> i64 {
        let now = now_unix();
        let target_id: i64 = sqlx::query_scalar(
            "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, alphabet_name, alphabet, status, created_at) \
             VALUES ('t', 'PRE', '.SUF', 0, 0, ?, ?, 0, ?, ?, 'active', ?) RETURNING id",
        )
        .bind(lower_bound)
        .bind(upper_bound)
        .bind(alphabet_name)
        .bind(alphabet)
        .bind(now)
        .fetch_one(pool)
        .await
        .unwrap();
        // Through crate::alphabet directly (not this module's i64 shims),
        // so a bound longer than index_width works here too.
        let start_len = lower_bound.chars().count() as i64;
        let start = crate::alphabet::candidate_to_index(alphabet, lower_bound).unwrap();
        let (start_block, start_index) = crate::alphabet::split_pos(alphabet, start_len, start);
        sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_block, next_index, alphabet_name, alphabet) VALUES (?, ?, ?, ?, ?, ?)")
            .bind(target_id)
            .bind(start_len)
            .bind(start_block)
            .bind(start_index)
            .bind(alphabet_name)
            .bind(alphabet)
            .execute(pool)
            .await
            .unwrap();
        target_id
    }

    /// Like `insert_target`, but with explicit `hash_a`/`hash_b` instead of
    /// the fixed `0, 0` every other fixture here uses - for tests about
    /// multiple targets sharing (or not sharing) the same hash pair.
    async fn insert_target_with_hash(pool: &SqlitePool, hash_a: i64, hash_b: i64, lower_bound: &str, upper_bound: &str) -> i64 {
        let now = now_unix();
        let target_id: i64 = sqlx::query_scalar(
            "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, alphabet_name, alphabet, status, created_at) \
             VALUES ('t', 'PRE', '.SUF', ?, ?, ?, ?, 0, 'size49', ?, 'active', ?) RETURNING id",
        )
        .bind(hash_a)
        .bind(hash_b)
        .bind(lower_bound)
        .bind(upper_bound)
        .bind(DEFAULT)
        .bind(now)
        .fetch_one(pool)
        .await
        .unwrap();
        let start_len = lower_bound.chars().count() as i64;
        let start_index = candidate_to_index(DEFAULT, lower_bound).unwrap();
        sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_index, alphabet_name, alphabet) VALUES (?, ?, ?, 'size49', ?)")
            .bind(target_id)
            .bind(start_len)
            .bind(start_index)
            .bind(DEFAULT)
            .execute(pool)
            .await
            .unwrap();
        target_id
    }

    /// Creates a skip range on `target_id` the way the admin API does.
    async fn add_skip_range(pool: &SqlitePool, target_id: i64, pattern: &str, candidate_len: i64) -> i64 {
        let target: Target = sqlx::query_as("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(pool).await.unwrap();
        let mut conn = pool.acquire().await.unwrap();
        create_skip_range(&mut conn, &target, pattern, candidate_len, "testing", now_unix()).await.unwrap()
    }

    /// A letters-alphabet ("custom") target from `lower_bound` to `upper_bound`
    /// with a skip range of `pattern`, at `lower_bound`'s length.
    async fn insert_target_with_skip_range(pool: &SqlitePool, alphabet: &str, lower_bound: &str, upper_bound: &str, pattern: &str) -> i64 {
        let target_id = insert_target_with_alphabet(pool, "custom", alphabet, lower_bound, upper_bound).await;
        add_skip_range(pool, target_id, pattern, lower_bound.chars().count() as i64).await;
        target_id
    }

    /// Adds a segment `[start, end)` to priority range `priority_range_id`.
    async fn add_priority_segment(pool: &SqlitePool, priority_range_id: i64, alphabet: &str, candidate_len: i64, start: Pos, end: Pos) {
        let mut conn = pool.acquire().await.unwrap();
        insert_segments(&mut conn, SegmentOwner::Priority, priority_range_id, alphabet, candidate_len, &[(start, end)]).await.unwrap();
    }

    /// Inserts a `priority_ranges` row directly (bypassing
    /// `handlers::admin_create_priority_range`'s validation, same as the
    /// other `insert_*` fixtures here bypass their own admin handler) for
    /// `pattern`, with its segments. Passing `""` covers the entire
    /// `candidate_len` space, which is handy for tests that don't care about
    /// narrowing it.
    async fn insert_priority_range(pool: &SqlitePool, target_id: i64, alphabet: &str, alphabet_name: &str, priority: i64, pattern: &str, candidate_len: i64) -> i64 {
        let pattern = if pattern.is_empty() { "." } else { pattern };
        let spans = pattern_spans(alphabet, pattern, candidate_len).unwrap();
        let (start_index, end_index) = (spans[0].0 as i64, spans[spans.len() - 1].1 as i64);
        let now = now_unix();
        let id = sqlx::query_scalar(
            "INSERT INTO priority_ranges (target_id, priority, pattern, candidate_len, start_index, end_index, next_index, alphabet_name, alphabet, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(target_id)
        .bind(priority)
        .bind(pattern)
        .bind(candidate_len)
        .bind(start_index)
        .bind(end_index)
        .bind(start_index)
        .bind(alphabet_name)
        .bind(alphabet)
        .bind(now)
        .fetch_one(pool)
        .await
        .unwrap();
        let mut conn = pool.acquire().await.unwrap();
        insert_segments(&mut conn, SegmentOwner::Priority, id, alphabet, candidate_len, &spans).await.unwrap();
        id
    }

    /// The scenario skip ranges exist for: a target that would
    /// normally hand out its whole A-Z space in one claim instead splits
    /// around a skipped middle chunk - A-L handed out, M-Q recorded as a
    /// "skipped" range immediately (not in advance, and not waiting for a
    /// worker to reach it), and R-Z left for the *next* claim rather than
    /// bundled into this one.
    #[tokio::test]
    async fn claim_range_splits_around_a_skip_range_and_defers_the_remainder() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_skip_range(&pool, letters, "A", "Z", "[M-Q]").await;

        // Bigger than the whole 26-candidate space, so the first claim would
        // otherwise greedily take all of A-Z in one go.
        let config = test_config(30);

        let claim1 = claim_range(&pool, &config, &user).await.unwrap().expect("A-L should be claimable");
        assert_eq!(claim1.target_id, target_id);
        assert_eq!(claim1.lower_bound_filename, "PREA.SUF");
        assert_eq!(claim1.upper_bound_filename, "PREL.SUF");
        assert_eq!(claim1.candidate_count, 12);

        let (skip_status, skip_start, skip_end): (String, i64, i64) =
            sqlx::query_as("SELECT status, start_index, end_index FROM ranges WHERE target_id = ? AND status = 'skipped'")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .expect("the M-Q skip range must already exist, before anyone claims past it");
        assert_eq!(skip_status, "skipped");
        assert_eq!(skip_start, 12); // 'M'
        assert_eq!(skip_end, 17); // one past 'Q'

        let claim2 = claim_range(&pool, &config, &user).await.unwrap().expect("R-Z should be claimable next");
        assert_eq!(claim2.lower_bound_filename, "PRER.SUF");
        assert_eq!(claim2.upper_bound_filename, "PREZ.SUF");
        assert_eq!(claim2.candidate_count, 9);

        let skip_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE target_id = ? AND status = 'skipped'")
            .bind(target_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(skip_count, 1, "no extra skip ranges should appear once R-Z is claimed");
    }

    /// When the cursor starts *inside* a skip run (here: right at the very
    /// beginning, because the target's own lower_bound falls in the skipped
    /// prefix), `claim_range` must record the skip and keep going within the
    /// same call rather than returning nothing - and running the whole
    /// skipped length out to its end must advance the cursor into the next
    /// candidate length, exactly like exhausting a length normally does.
    #[tokio::test]
    async fn claim_range_steps_over_a_skip_run_starting_at_the_cursor_within_one_call() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_skip_range(&pool, letters, "A", "Z", "[A-L]").await;

        let config = test_config(30);

        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("M-Z should be claimable, past the skipped A-L prefix");
        assert_eq!(claim.lower_bound_filename, "PREM.SUF");
        assert_eq!(claim.upper_bound_filename, "PREZ.SUF");
        assert_eq!(claim.candidate_count, 14);

        let (skip_start, skip_end): (i64, i64) =
            sqlx::query_as("SELECT start_index, end_index FROM ranges WHERE target_id = ? AND status = 'skipped'")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .expect("the A-L skip range must exist even though it produced no claimable work of its own");
        assert_eq!((skip_start, skip_end), (0, 12));

        // The cursor must have advanced into length 2 (there's no length-1
        // work left: 0..12 was skipped, 12..26 was just claimed above).
        let (candidate_len, next_index): (i64, i64) =
            sqlx::query_as("SELECT candidate_len, next_index FROM target_progress WHERE target_id = ?")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(candidate_len, 2);
        assert_eq!(next_index, 0);
    }

    /// `namebreak bounded` itself only ever searches a single fixed length per
    /// invocation, so a correct range can never straddle two lengths. This
    /// exercises exactly that boundary: the first claim must exactly exhaust
    /// length 1's whole space (no leftover, nothing skipped), and the second
    /// must pick up length 2 starting exactly at index 0 (no gap) - using
    /// bounds (a single min-char to a single max-char) that span the *entire*
    /// space at every length, so what's being tested is purely the crossing
    /// itself, not any particular stopping point.
    #[tokio::test]
    async fn range_carving_crosses_a_candidate_length_boundary_cleanly() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower1, upper1) = full_bounds(DEFAULT, 1);
        insert_target(&pool, &lower1, &upper1).await;

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
    }

    /// There's no admin-supplied max_len ("always go as long as possible") -
    /// carving goes past index_width (where a position no longer fits one
    /// stored i64) and only stops at max_supported_len, the client's own
    /// limit. Long bounds that differ only near the end keep lengths 15 and
    /// 16 tiny, so two claims reach the ceiling.
    #[tokio::test]
    async fn range_carving_goes_past_index_width_and_stops_at_max_supported_len() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let cap = max_supported_len(DEFAULT);
        assert!(cap > index_width(DEFAULT) + 1);
        // Length 15: just these two. Length 16: each followed by any of the 49 characters.
        let lower = format!("Y{}", "_".repeat(cap as usize - 2));
        let upper = format!("Z{}", " ".repeat(cap as usize - 2));
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(1_000);
        let claim1 = claim_range(&pool, &config, &user).await.unwrap().expect(&format!("length {} should have work", cap - 1));
        assert_eq!(claim1.candidate_count, 2);
        assert_eq!(claim1.lower_bound_filename, format!("PRE{lower}.SUF"));
        assert_eq!(claim1.upper_bound_filename, format!("PRE{upper}.SUF"));

        let claim2 = claim_range(&pool, &config, &user).await.unwrap().expect(&format!("length {cap} (the ceiling) should have work"));
        assert_eq!(claim2.candidate_count, 2 * 49);
        assert_eq!(claim2.lower_bound_filename, format!("PRE{lower} .SUF"));
        assert_eq!(claim2.upper_bound_filename, format!("PRE{upper}_.SUF"));

        // Stored per leading prefix: its start is the last candidate under
        // one prefix ("Y____"), its end just past the first under the next ("Z    ").
        let range: Range = sqlx::query_as("SELECT * FROM ranges WHERE id = ?").bind(claim2.range_id).fetch_one(&pool).await.unwrap();
        assert!(range.start_block > 0 && range.end_block == range.start_block + 1);
        assert_eq!(range.start(), crate::alphabet::candidate_to_index(DEFAULT, &format!("{lower} ")).unwrap());
        assert_eq!(range.end(), crate::alphabet::candidate_to_index(DEFAULT, &format!("{upper}_")).unwrap() + 1);

        let claim3 = claim_range(&pool, &config, &user).await.unwrap();
        assert!(claim3.is_none(), "max_supported_len is now fully carved - there must be no work beyond the true ceiling");
    }

    /// A heartbeat checkpoint and the split on reassignment work across a
    /// leading-prefix boundary too, not just within one block.
    #[tokio::test]
    async fn heartbeat_progress_splits_a_range_straddling_two_leading_prefixes() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        let second_user = insert_user(&pool, "second").await;
        // Length 12: the last candidate under leading "Y", then the first under "Z".
        let (lower, upper) = (format!("Y{}", "_".repeat(11)), format!("Z{}", " ".repeat(11)));
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(1_000);
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");
        assert_eq!(claim.candidate_count, 2);
        heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(format!("PRE{lower}.SUF"))).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?").bind(claim.range_id).execute(&pool).await.unwrap();
        reclaim_expired(&pool).await.unwrap();
        let resumed = claim_range(&pool, &config, &second_user).await.unwrap().expect("the unsearched remainder");
        assert_eq!(resumed.candidate_count, 1);
        assert_eq!(resumed.lower_bound_filename, format!("PRE{upper}.SUF"));

        let done: Range = sqlx::query_as("SELECT * FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(done.status, "completed");
        assert_eq!(done.end(), crate::alphabet::candidate_to_index(DEFAULT, &upper).unwrap());
    }

    /// A priority range at a long length can span many leading prefixes -
    /// here a whole one ("A" at length 12) - and is carved across them.
    #[tokio::test]
    async fn priority_range_spanning_leading_prefixes_is_carved_from_its_stored_blocks() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let target_id = insert_target(&pool, &format!(" {}", " ".repeat(11)), "_").await;
        let (start, end_inclusive) = crate::alphabet::bound_indices_at_len(DEFAULT, "A", "A", 12);
        let end = end_inclusive + 1;
        assert!(end > i64::MAX as Pos, "must not fit a single i64");
        let (start_block, start_index) = crate::alphabet::split_pos(DEFAULT, 12, start);
        let (end_block, end_index) = crate::alphabet::split_end(DEFAULT, 12, end);
        let id: i64 = sqlx::query_scalar(
            "INSERT INTO priority_ranges (target_id, priority, pattern, candidate_len, start_block, start_index, end_block, end_index, \
             next_block, next_index, alphabet_name, alphabet, created_at) VALUES (?, 1, 'A', 12, ?, ?, ?, ?, ?, ?, 'size49', ?, 0) RETURNING id",
        )
        .bind(target_id)
        .bind(start_block)
        .bind(start_index)
        .bind(end_block)
        .bind(end_index)
        .bind(start_block)
        .bind(start_index)
        .bind(DEFAULT)
        .fetch_one(&pool)
        .await
        .unwrap();
        add_priority_segment(&pool, id, DEFAULT, 12, start, end).await;

        let claim = claim_range(&pool, &test_config(1_000), &user).await.unwrap().expect("priority work");
        assert_eq!(claim.lower_bound_filename, format!("PREA{}.SUF", " ".repeat(11)));
        assert_eq!(claim.candidate_count, 1_000);
        let pr: PriorityRange = sqlx::query_as("SELECT * FROM priority_ranges WHERE target_id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        assert_eq!((pr.start(), pr.next(), pr.end()), (start, start + 1_000, end));
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
        let (lower, upper) = full_bounds(DEFAULT, 3); // fixed length, whole space handed out as one range
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 3));
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");
        assert_eq!(claim.candidate_count, space_size(DEFAULT, 3));

        let midpoint_index = space_size(DEFAULT, 3) / 2;
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, midpoint_index, 3));
        heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(midpoint_filename)).await.unwrap();

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

    /// Releasing a range with real progress must *not* split it straight
    /// away: until someone else claims it, it stays a single pending row
    /// still carrying its checkpoint and previous claimant, so that claimant
    /// can still come back and re-adopt it under the same range id (see
    /// heartbeat_range). Checks DB state right after reclaim_expired, with no
    /// claim_range call in between.
    #[tokio::test]
    async fn reclaim_expired_defers_the_split_until_someone_claims_it() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 3));
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");

        let midpoint_index = space_size(DEFAULT, 3) / 2;
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, midpoint_index, 3));
        heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(midpoint_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?")
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        let row_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE target_id = ?")
            .bind(claim.target_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(row_count, 1, "no split yet - nobody else has claimed the range");

        let (status, start, end, progress, assigned, worker): (String, i64, i64, Option<i64>, Option<i64>, Option<i64>) = sqlx::query_as(
            "SELECT status, start_index, end_index, progress_index, assigned_user_id, last_assigned_user_id FROM ranges WHERE id = ?",
        )
        .bind(claim.range_id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(status, "pending");
        assert_eq!((start, end), (0, space_size(DEFAULT, 3)), "still the whole original range");
        assert_eq!(progress, Some(midpoint_index), "the checkpoint must survive the release");
        assert!(assigned.is_none());
        assert_eq!(worker, Some(first_user.id), "still remembers who it belonged to");
    }

    /// The scenario deferring the split exists for: a client loses
    /// connectivity long enough for its lease to lapse, keeps searching
    /// regardless, and heartbeats again once it's back. Nobody else claimed
    /// the range in the meantime, so it's handed straight back - same range
    /// id, checkpoint intact, and the newly reported progress applied.
    #[tokio::test]
    async fn heartbeat_readopts_a_released_range_nobody_else_claimed() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "flaky").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 3));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        let quarter_index = space_size(DEFAULT, 3) / 4;
        let quarter_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, quarter_index, 3));
        heartbeat_range(&pool, &config, &user, claim.range_id, Some(quarter_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?")
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        // Back online, having kept searching past the midpoint meanwhile.
        let midpoint_index = space_size(DEFAULT, 3) / 2;
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, midpoint_index, 3));
        let outcome = heartbeat_range(&pool, &config, &user, claim.range_id, Some(midpoint_filename)).await.unwrap();
        assert!(!outcome.range_released);

        let (status, start, end, progress, assigned, lease_expires_at, last_progress_at): (String, i64, i64, Option<i64>, Option<i64>, Option<i64>, Option<i64>) =
            sqlx::query_as(
                "SELECT status, start_index, end_index, progress_index, assigned_user_id, lease_expires_at, last_progress_at FROM ranges WHERE id = ?",
            )
            .bind(claim.range_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "in_progress");
        assert_eq!((start, end), (0, space_size(DEFAULT, 3)));
        assert_eq!(progress, Some(midpoint_index), "the progress reported on return must be applied");
        assert_eq!(assigned, Some(user.id));
        assert!(lease_expires_at.unwrap() > now_unix(), "a fresh lease");
        assert!(last_progress_at.is_some(), "the stall clock must be running again");

        // And it completes normally under the same range id.
        complete_range(&pool, &config, &user, claim.range_id, false, None, 1.0, 1).await.unwrap();
        let status: String = sqlx::query_scalar("SELECT status FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(status, "completed");
    }

    /// Re-adoption is only for the range's own previous claimant, and only
    /// while it's still unclaimed: once someone else has claimed it (splitting
    /// off the searched portion), a returning claimant's heartbeat is stale
    /// and its reported progress must be discarded, exactly as before.
    #[tokio::test]
    async fn heartbeat_does_not_readopt_a_range_claimed_by_someone_else() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        let second_user = insert_user(&pool, "second").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 3));
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");

        let quarter_index = space_size(DEFAULT, 3) / 4;
        let quarter_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, quarter_index, 3));
        heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(quarter_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?")
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        // Some other user's heartbeat against the pending row mustn't adopt it either.
        let result = heartbeat_range(&pool, &config, &second_user, claim.range_id, None).await;
        assert!(matches!(result, Err(AppError::Conflict(_))));

        let resumed = claim_range(&pool, &config, &second_user).await.unwrap().expect("reassignable");
        assert_ne!(resumed.range_id, claim.range_id);

        let midpoint_index = space_size(DEFAULT, 3) / 2;
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, midpoint_index, 3));
        let result = heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(midpoint_filename)).await;
        assert!(matches!(result, Err(AppError::Conflict(_))));

        let (status, end, progress): (String, i64, Option<i64>) =
            sqlx::query_as("SELECT status, end_index, progress_index FROM ranges WHERE id = ?")
                .bind(claim.range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "completed", "the searched portion, split off by the second user's claim");
        assert_eq!(end, quarter_index + 1);
        assert!(progress.is_none(), "the stale heartbeat's progress must not have been recorded");

        let (resumed_progress,): (Option<i64>,) =
            sqlx::query_as("SELECT progress_index FROM ranges WHERE id = ?").bind(resumed.range_id).fetch_one(&pool).await.unwrap();
        assert!(resumed_progress.is_none(), "nor leaked onto the second user's remainder");
    }

    /// If a client's last heartbeat before disconnecting already covered the very
    /// end of its range, the range has in fact been fully searched (with no full
    /// match reported) - it should be closed out as completed on its own, rather
    /// than handed to the next claimer as a zero-width range (or, worse, silently
    /// reassigned and redone).
    #[tokio::test]
    async fn heartbeat_progress_reaching_the_end_completes_the_range_without_reassigning() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("work available");

        let last_index = space_size(DEFAULT, 2) - 1;
        let last_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, last_index, 2));
        heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(last_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?")
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        // The reclaimed range itself must be closed out, not handed back out again -
        // even though the target still has longer lengths to search under "always
        // go as long as possible", so a subsequent claim legitimately returns fresh
        // work there rather than nothing at all.
        let second_user = insert_user(&pool, "second").await;
        let claim2 = claim_range(&pool, &config, &second_user).await.unwrap().expect("target still has longer lengths to search");
        assert_ne!(claim2.range_id, claim.range_id, "the already-fully-searched range must not be reassigned");

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
        let (lower, upper) = full_bounds(SIZE42, 3);
        insert_target_with_alphabet(&pool, "size42", SIZE42, &lower, &upper).await;

        let config = test_config(space_size(SIZE42, 3));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        assert_eq!(claim.alphabet, SIZE42);
        assert_eq!(claim.candidate_count, space_size(SIZE42, 3), "should be size42's space (74088), not size49's");
        assert_ne!(claim.candidate_count, space_size(DEFAULT, 3));
        let (exp_lower, exp_upper) = range_bound_filenames(SIZE42, "PRE", ".SUF", 3, 0, space_size(SIZE42, 3));
        assert_eq!(claim.lower_bound_filename, exp_lower);
        assert_eq!(claim.upper_bound_filename, exp_upper);
    }

    /// The core scenario protocol versioning exists for: a target using an
    /// alphabet the calling client's declared version predates is entirely
    /// invisible to it - claim_range falls through to a different,
    /// compatible target instead, rather than handing out work the client
    /// has no idea how to search. `size42` is tagged `(1, 0)` in
    /// `PREDEFINED_ALPHABETS`, so a hypothetical pre-1.0 client (not
    /// something `handlers::register` would ever actually let through, but
    /// perfectly well-formed semver, and this test only needs a stored
    /// value lower than `(1, 0)` to exercise the comparison) can't see it.
    #[tokio::test]
    async fn claim_range_hides_a_target_whose_alphabet_predates_the_clients_declared_version() {
        let pool = test_pool().await;
        let old_client = insert_user_with_protocol_version(&pool, "old-client", "0.9.0").await;
        let (lower, upper) = full_bounds(SIZE42, 2);
        insert_target_with_alphabet(&pool, "size42", SIZE42, &lower, &upper).await;

        let config = test_config(space_size(SIZE42, 2));
        assert!(claim_range(&pool, &config, &old_client).await.unwrap().is_none(), "the only target uses an alphabet this client predates");
    }

    /// A client searching a range in a bigger alphabet gets the same first
    /// and last candidate as bounds, the bigger alphabet, and the number of
    /// candidates it will actually search.
    #[tokio::test]
    async fn to_claim_response_hands_out_a_range_in_a_bigger_alphabet() {
        let pool = test_pool().await;
        let (lower, upper) = full_bounds(SIZE42, 2);
        let target_id = insert_target_with_alphabet(&pool, "size42", SIZE42, &lower, &upper).await;
        let target: Target = sqlx::query_as("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        let (start, end) = (candidate_to_index(SIZE42, "A ").unwrap(), candidate_to_index(SIZE42, "B ").unwrap());

        let claim = to_claim_response(&target, 1, 2, start.into(), end.into(), 60, SIZE42, DEFAULT);
        assert_eq!((claim.lower_bound_filename.as_str(), claim.upper_bound_filename.as_str()), ("PREA .SUF", "PREA_.SUF"));
        assert_eq!(claim.alphabet, DEFAULT);
        assert_eq!(claim.candidate_count, 49, "every size49 candidate from 'A ' to 'A_'");

        let own = to_claim_response(&target, 1, 2, start.into(), end.into(), 60, SIZE42, SIZE42);
        assert_eq!((own.alphabet.as_str(), own.candidate_count), (SIZE42, 42));
    }

    /// A chunk for a client searching in a bigger alphabet shrinks by how
    /// many more candidates it searches per candidate of the range.
    #[test]
    fn clip_spans_to_bounds_drops_what_lies_outside_the_targets_bounds() {
        let (lower, upper) = ("FINZ09BX.TXT", "GAMEMENU.BIN");
        let at = |candidate: &str| crate::alphabet::candidate_to_index(DEFAULT, candidate).unwrap();
        let clip = |pattern: &str| clip_spans_to_bounds(DEFAULT, lower, upper, 12, &pattern_spans(DEFAULT, pattern, 12).unwrap());

        // Straddles the upper bound: cut off right after it.
        assert_eq!(clip("GA"), vec![(at("GA          "), at(upper) + 1)]);
        // Straddles the lower bound: starts right at it.
        assert_eq!(clip("FI"), vec![(at(lower), at("FI__________") + 1)]);
        // Entirely inside: untouched.
        assert_eq!(clip("FL"), pattern_spans(DEFAULT, "FL", 12).unwrap());
        // Entirely outside: nothing left.
        assert!(clip("GB").is_empty());
    }

    #[test]
    fn chunk_size_shrinks_for_a_bigger_client_alphabet() {
        let config = test_config(49 * 49);
        assert_eq!(chunk_size(&config, 1.0, SIZE42, SIZE42, 2), 49 * 49);
        assert_eq!(chunk_size(&config, 1.0, SIZE42, DEFAULT, 2), 42 * 42);
    }

    /// Progress reported by a client searching in a bigger alphabet can be a
    /// candidate the range's alphabet doesn't have: it's rounded down to the
    /// last one the range has at or before it, so nothing is skipped.
    #[tokio::test]
    async fn heartbeat_rounds_progress_from_a_bigger_alphabet_down() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(SIZE42, 2);
        insert_target_with_alphabet(&pool, "size42", SIZE42, &lower, &upper).await;
        let config = test_config(42); // the whole ' ?' block: "  " to " _"
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work");

        let progress = |pool: SqlitePool| async move {
            let range: Range = sqlx::query_as("SELECT * FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
            range.progress().map(|p| crate::alphabet::index_to_candidate(SIZE42, p, 2))
        };
        // '!' isn't in size42 and comes right after ' ': only "  " is certainly searched.
        heartbeat_range(&pool, &config, &user, claim.range_id, Some("PRE !.SUF".into())).await.unwrap();
        assert_eq!(progress(pool.clone()).await.as_deref(), Some("  "));
        // '[' comes between 'Z' and '_'.
        heartbeat_range(&pool, &config, &user, claim.range_id, Some("PRE [.SUF".into())).await.unwrap();
        assert_eq!(progress(pool.clone()).await.as_deref(), Some(" Z"));
        // '~' is in no alphabet at all: ignored.
        heartbeat_range(&pool, &config, &user, claim.range_id, Some("PRE ~.SUF".into())).await.unwrap();
        assert_eq!(progress(pool.clone()).await.as_deref(), Some(" Z"));
    }

    /// The other side of the same gate: once a target's alphabet is one the
    /// client's declared version does understand, it's claimable as normal.
    #[tokio::test]
    async fn claim_range_serves_a_target_whose_alphabet_the_client_understands() {
        let pool = test_pool().await;
        let new_client = insert_user_with_protocol_version(&pool, "new-client", "1.0.0").await;
        let (lower, upper) = full_bounds(SIZE42, 2);
        insert_target_with_alphabet(&pool, "size42", SIZE42, &lower, &upper).await;

        let config = test_config(space_size(SIZE42, 2));
        let claim = claim_range(&pool, &config, &new_client).await.unwrap().expect("this client's version understands size42");
        assert_eq!(claim.alphabet, SIZE42);
    }

    /// A target with a custom alphabet: a client of protocol 1.3 searches it as
    /// it is; an older one, which can only search the predefined alphabets'
    /// sizes, the smallest predefined alphabet that has all of its characters
    /// - or, with none, gets no work from it.
    #[tokio::test]
    async fn claim_range_gives_a_custom_alphabet_only_to_clients_that_take_any() {
        let pool = test_pool().await;
        let (name, chars) = crate::alphabet::custom_alphabet("ABC").unwrap();
        let (lower, upper) = full_bounds(&chars, 2);
        insert_target_with_alphabet(&pool, &name, &chars, &lower, &upper).await;
        let config = test_config(space_size(&chars, 2));

        let new_client = insert_user_with_protocol_version(&pool, "new-client", "1.3.0").await;
        let claim = claim_range(&pool, &config, &new_client).await.unwrap().expect("a 1.3 client takes a custom alphabet");
        assert_eq!(claim.alphabet, chars);
        let old_client = insert_user_with_protocol_version(&pool, "old-client", "1.2.0").await;
        let claim = claim_range(&pool, &config, &old_client).await.unwrap().expect("size29 has A, B and C");
        assert_eq!(claim.alphabet, crate::alphabet::lookup_predefined_alphabet("size29").unwrap());

        let pool = test_pool().await;
        let (name, chars) = crate::alphabet::custom_alphabet("AB~").unwrap();
        let (lower, upper) = full_bounds(&chars, 2);
        insert_target_with_alphabet(&pool, &name, &chars, &lower, &upper).await;
        let old_client = insert_user_with_protocol_version(&pool, "old-client", "1.2.0").await;
        assert!(claim_range(&pool, &config, &old_client).await.unwrap().is_none(), "no predefined alphabet has '~'");
    }

    /// With two targets, one alphabet-incompatible and one not, an old
    /// client is transparently steered to the one it can handle - the whole
    /// point of gating per-target rather than just refusing the client
    /// outright. Every *real* predefined alphabet happens to be tagged
    /// `(1, 0)` as of this writing, so there's no actual alphabet a 0.9.0
    /// client could understand - the "compatible" target here uses a
    /// synthetic alphabet name instead, relying on `client_alphabet_for`'s
    /// fail-open behavior for unrecognized names (see its own doc comment).
    /// That's still a faithful test of the *mechanism* (skip one target,
    /// fall through to the next), just not of a same-day real-world alphabet.
    #[tokio::test]
    async fn claim_range_falls_through_to_a_compatible_target_when_another_is_hidden() {
        let pool = test_pool().await;
        let old_client = insert_user_with_protocol_version(&pool, "old-client", "0.9.0").await;
        let (lower42, upper42) = full_bounds(SIZE42, 2);
        insert_target_with_alphabet(&pool, "size42", SIZE42, &lower42, &upper42).await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let compatible_target = insert_target_with_alphabet(&pool, "synthetic-compatible", DEFAULT, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2).max(space_size(SIZE42, 2)));
        let claim = claim_range(&pool, &config, &old_client).await.unwrap().expect("the synthetic-alphabet target is still visible");
        assert_eq!(claim.target_id, compatible_target);
    }

    /// The scenario admin_patch_target's alphabet change exists for: a target
    /// already carved partway through a candidate length under one alphabet
    /// gets patched to a smaller one, and the very next claim (not the patch
    /// itself - see transition_alphabet_cursor's doc comment) must: leave
    /// every already-carved range alone, record a `skipped` range under the
    /// OLD alphabet for the portion of the current block that can no longer
    /// be expressed, and hand out fresh work under the NEW alphabet starting
    /// from the first candidate that actually is.
    #[tokio::test]
    async fn claim_range_transitions_the_cursor_lazily_after_an_alphabet_patch() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        // Bounds must stay expressible in whatever alphabet is active - since
        // they're immutable, they have to work under both the old alphabet
        // and the new (letters-only) one this test patches to.
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AAAAAA", "ZZZZZZ").await;

        // Simulate carving having already reached "ABC001" under the old alphabet.
        let cursor_index = candidate_to_index(old_alphabet, "ABC001").unwrap();
        sqlx::query("UPDATE target_progress SET candidate_len = 6, next_index = ? WHERE target_id = ?")
            .bind(cursor_index)
            .bind(target_id)
            .execute(&pool)
            .await
            .unwrap();

        // Simulate an admin PATCH: only the target's own alphabet flips - the
        // cursor stays denominated in the old alphabet until claim_range
        // itself reaches it (see target_progress.alphabet).
        sqlx::query("UPDATE targets SET alphabet_name = 'letters_only', alphabet = ? WHERE id = ?")
            .bind(new_alphabet)
            .bind(target_id)
            .execute(&pool)
            .await
            .unwrap();

        let config = test_config(1_000_000);
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available under the new alphabet");

        assert_eq!(claim.alphabet, new_alphabet, "the freshly carved range must use the new alphabet");
        assert_eq!(claim.lower_bound_filename, "PREABCAAA.SUF", "must resume at the first candidate expressible in the new alphabet");

        let (skip_status, skip_start, skip_end, skip_alphabet): (String, i64, i64, String) =
            sqlx::query_as("SELECT status, start_index, end_index, alphabet FROM ranges WHERE target_id = ? AND status = 'skipped'")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .expect("the old-alphabet remainder of the ABC block must be persisted as a skipped range");
        assert_eq!(skip_status, "skipped");
        assert_eq!(skip_start, cursor_index);
        assert_eq!(skip_end, candidate_to_index(old_alphabet, "ABCAAA").unwrap(), "skip runs exactly up to the resume point, not the whole shared prefix block");
        assert_eq!(skip_alphabet, old_alphabet, "the skip range is denominated in the OLD alphabet, not the target's current one");

        let (progress_alphabet_name, progress_alphabet): (String, String) =
            sqlx::query_as("SELECT alphabet_name, alphabet FROM target_progress WHERE target_id = ?")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(progress_alphabet_name, "letters_only", "the cursor itself must now be stamped with the new alphabet");
        assert_eq!(progress_alphabet, new_alphabet);
    }

    /// Fetches the (candidate_len, start_index) of a carved range.
    async fn range_position(pool: &SqlitePool, range_id: i64) -> (i64, i64) {
        sqlx::query_as("SELECT candidate_len, start_index FROM ranges WHERE id = ?").bind(range_id).fetch_one(pool).await.unwrap()
    }

    /// Raising start_len past the cursor makes the next carve jump straight
    /// to the start of start_len, persisting nothing for what it jumps over;
    /// lowering it again never moves the cursor back.
    #[tokio::test]
    async fn claim_range_jumps_to_a_raised_start_len_and_never_back() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let target_id = insert_target(&pool, "A", "Z").await;
        sqlx::query("UPDATE targets SET start_len = 3 WHERE id = ?").bind(target_id).execute(&pool).await.unwrap();

        // Larger than all of length 3 within these bounds, so each claim takes a whole length.
        let config = test_config(1_000_000);
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available at start_len");
        assert_eq!(range_position(&pool, claim.range_id).await, (3, bound_indices_at_len(DEFAULT, "A", "Z", 3).0));
        let range_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE target_id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        assert_eq!(range_count, 1, "nothing is persisted for the jumped-over lengths");

        sqlx::query("UPDATE targets SET start_len = 1 WHERE id = ?").bind(target_id).execute(&pool).await.unwrap();
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available at length 4");
        assert_eq!(range_position(&pool, claim.range_id).await, (4, bound_indices_at_len(DEFAULT, "A", "Z", 4).0));
    }

    /// A start_len jump lands at the very start of a length, so a pending
    /// alphabet patch needs no translation: the cursor goes straight to the
    /// target's current alphabet, with no skipped range for the old one.
    #[tokio::test]
    async fn claim_range_start_len_jump_settles_a_pending_alphabet_patch() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AAAAAA", "ZZZZZZ").await;
        let cursor_index = candidate_to_index(old_alphabet, "ABC001").unwrap();
        sqlx::query("UPDATE target_progress SET next_index = ? WHERE target_id = ?").bind(cursor_index).bind(target_id).execute(&pool).await.unwrap();
        sqlx::query("UPDATE targets SET alphabet_name = 'letters_only', alphabet = ?, start_len = 7 WHERE id = ?")
            .bind(new_alphabet)
            .bind(target_id)
            .execute(&pool)
            .await
            .unwrap();

        let claim = claim_range(&pool, &test_config(1_000), &user).await.unwrap().expect("work available at start_len");
        assert_eq!(claim.alphabet, new_alphabet);
        assert_eq!(claim.lower_bound_filename, "PREAAAAAAA.SUF");
        let skipped: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE target_id = ? AND status = 'skipped'").bind(target_id).fetch_one(&pool).await.unwrap();
        assert_eq!(skipped, 0);
    }

    /// The core scheduling behavior a priority range exists for: it's
    /// claimed before the target's own main sweep even though both would
    /// otherwise compete for the very same length-1 space, and once it's
    /// exhausted the main sweep's own chunk is truncated to stop just short
    /// of it - jumping straight over the single candidate it owns without
    /// leaving any row behind for that gap.
    #[tokio::test]
    async fn claim_range_serves_a_priority_range_before_the_targets_own_main_sweep() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;
        let priority_range_id = insert_priority_range(&pool, target_id, letters, "letters", 10, "Z", 1).await;

        let config = test_config(100); // bigger than the whole 26-candidate length-1 space

        let claim1 = claim_range(&pool, &config, &user).await.unwrap().expect("the priority range should be claimed first");
        assert_eq!(claim1.lower_bound_filename, "PREZ.SUF");
        assert_eq!(claim1.upper_bound_filename, "PREZ.SUF");
        assert_eq!(claim1.candidate_count, 1);

        let (priority_range_ref,): (Option<i64>,) =
            sqlx::query_as("SELECT priority_range_id FROM ranges WHERE id = ?").bind(claim1.range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(priority_range_ref, Some(priority_range_id), "the claimed range should carry its priority-range provenance");

        let (pr_next_index,): (i64,) =
            sqlx::query_as("SELECT next_index FROM priority_ranges WHERE id = ?").bind(priority_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(pr_next_index, 26, "the priority range's own cursor should now be exhausted");

        let claim2 = claim_range(&pool, &config, &user).await.unwrap().expect("the main sweep should now be claimable");
        assert_eq!(claim2.lower_bound_filename, "PREA.SUF");
        assert_eq!(claim2.upper_bound_filename, "PREY.SUF", "must stop just short of 'Z', which the priority range already owns");
        assert_eq!(claim2.candidate_count, 25);

        let skip_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE target_id = ? AND status = 'skipped'").bind(target_id).fetch_one(&pool).await.unwrap();
        assert_eq!(skip_count, 0, "the priority range's gap must not be persisted as its own row");
    }

    /// Several priority ranges on the same target are walked highest
    /// priority first, ties broken by creation order - the same convention
    /// `targets.priority` already uses.
    #[tokio::test]
    async fn claim_priority_range_chunk_prefers_higher_priority_then_creation_order() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;

        let low_id = insert_priority_range(&pool, target_id, letters, "letters", 1, "A", 1).await;
        let high_id = insert_priority_range(&pool, target_id, letters, "letters", 10, "M", 1).await;
        let tie_id = insert_priority_range(&pool, target_id, letters, "letters", 10, "N", 1).await;

        let config = test_config(1); // exactly one candidate per claim

        for expected in [high_id, tie_id, low_id] {
            let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
            let (priority_range_ref,): (Option<i64>,) =
                sqlx::query_as("SELECT priority_range_id FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
            assert_eq!(priority_range_ref, Some(expected));
        }
    }

    /// Skip ranges still exclude matching candidates *within* a priority
    /// range's own space - being prioritized doesn't stop a candidate from
    /// also being uninteresting.
    #[tokio::test]
    async fn claim_priority_range_chunk_still_applies_the_targets_skip_ranges() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_skip_range(&pool, letters, "A", "Z", "[M-Q]").await;
        // Covers the entire length-1 space, so it overlaps the skip zone.
        insert_priority_range(&pool, target_id, letters, "custom", 10, "", 1).await;

        let config = test_config(30); // bigger than the whole 26-candidate space

        let claim1 = claim_range(&pool, &config, &user).await.unwrap().expect("A-L should be claimable from the priority range");
        assert_eq!(claim1.lower_bound_filename, "PREA.SUF");
        assert_eq!(claim1.upper_bound_filename, "PREL.SUF");

        let (skip_status, skip_start, skip_end, skip_range_id): (String, i64, i64, Option<i64>) =
            sqlx::query_as("SELECT status, start_index, end_index, skip_range_id FROM ranges WHERE target_id = ? AND status = 'skipped'")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .expect("the M-Q skip must be recorded, once, even though it's inside a priority range");
        assert_eq!(skip_status, "skipped");
        assert_eq!((skip_start, skip_end), (12, 17));
        assert!(skip_range_id.is_some(), "tagged with the skip range it came from");

        let claim2 = claim_range(&pool, &config, &user).await.unwrap().expect("R-Z should be claimable next, still from the priority range");
        assert_eq!(claim2.lower_bound_filename, "PRER.SUF");
        assert_eq!(claim2.upper_bound_filename, "PREZ.SUF");
    }

    /// A reclaimed, split-off remainder must still point back at whichever
    /// priority range the original (now-shrunk, completed) row came from -
    /// otherwise re-claiming it would silently look like ordinary main-sweep
    /// work on the dashboard.
    #[tokio::test]
    async fn reclaim_expired_carries_the_priority_range_id_forward_on_a_split_remainder() {
        let pool = test_pool().await;
        let first_user = insert_user(&pool, "first").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "AAA", "ZZZ").await;
        let priority_range_id = insert_priority_range(&pool, target_id, letters, "letters", 10, "A", 3).await;

        let config = test_config(676); // the whole "A??" block at length 3
        let claim = claim_range(&pool, &config, &first_user).await.unwrap().expect("priority range should be claimable");
        assert_eq!(claim.candidate_count, 676);

        let (start_index,): (i64,) = sqlx::query_as("SELECT start_index FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(letters, start_index + 338, 3));
        heartbeat_range(&pool, &config, &first_user, claim.range_id, Some(midpoint_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?").bind(claim.range_id).execute(&pool).await.unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        // The split happens once someone else claims it.
        let second_user = insert_user(&pool, "second").await;
        let resumed = claim_range(&pool, &config, &second_user).await.unwrap().expect("remainder should be claimable");
        assert_ne!(resumed.range_id, claim.range_id);

        let (new_priority_range_id,): (Option<i64>,) =
            sqlx::query_as("SELECT priority_range_id FROM ranges WHERE id = ?").bind(resumed.range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(new_priority_range_id, Some(priority_range_id));
    }

    /// An untouched priority range (nothing ever carved from it) is removed
    /// outright; one that has carved everything it had is kept as a record
    /// of that work, so the main sweep keeps jumping over (and not
    /// duplicating) it.
    #[tokio::test]
    async fn remove_priority_range_deletes_when_untouched_and_keeps_what_was_carved() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;

        let untouched_id = insert_priority_range(&pool, target_id, letters, "letters", 5, "A", 1).await;
        let touched_id = insert_priority_range(&pool, target_id, letters, "letters", 10, "M", 1).await;

        let config = test_config(1);
        claim_range(&pool, &config, &user).await.unwrap().expect("should claim the higher-priority one");

        let removal = remove_priority_range(&pool, untouched_id).await.unwrap().expect("exists");
        assert!(removal.deleted);
        assert_eq!((removal.returned_to_main_sweep, removal.kept), (1, 0));
        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM priority_ranges WHERE id = ?").bind(untouched_id).fetch_one(&pool).await.unwrap();
        assert_eq!(remaining, 0);

        let removal = remove_priority_range(&pool, touched_id).await.unwrap().expect("exists");
        assert!(!removal.deleted, "work was carved from it, so it stays as the record of that");
        assert_eq!((removal.returned_to_main_sweep, removal.kept), (0, 0));

        assert!(remove_priority_range(&pool, 999_999).await.unwrap().is_none());
    }

    /// Letters alphabet, a target at length 2, and a priority range "M"
    /// (MA-MZ, indices 312..338) with its first 5 candidates (MA-ME) carved.
    async fn partly_carved_priority_range(pool: &SqlitePool, user: &User, config: &RangeConfig) -> (i64, i64, i64) {
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(pool, "letters", letters, "AA", "ZZ").await;
        sqlx::query("UPDATE target_progress SET candidate_len = 2, next_index = 0 WHERE target_id = ?").bind(target_id).execute(pool).await.unwrap();
        let priority_range_id = insert_priority_range(pool, target_id, letters, "letters", 10, "M", 2).await;
        let claim = claim_range(pool, config, user).await.unwrap().expect("the priority range should be claimed first");
        assert_eq!(claim.candidate_count, 5);
        (target_id, priority_range_id, 312)
    }

    /// The scenario that used to lose candidates: a priority range is removed
    /// after carving some of its span. The rest must go back to the main
    /// sweep (which hasn't reached it yet) - and then actually get searched,
    /// with every length-2 candidate handed out exactly once.
    #[tokio::test]
    async fn remove_priority_range_hands_the_uncarved_rest_back_to_the_main_sweep() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let config = test_config(5);
        let (target_id, priority_range_id, start) = partly_carved_priority_range(&pool, &user, &config).await;

        let removal = remove_priority_range(&pool, priority_range_id).await.unwrap().expect("exists");
        assert!(!removal.deleted);
        assert_eq!((removal.returned_to_main_sweep, removal.kept), (21, 0));
        let (next_index, end_index): (i64, i64) =
            sqlx::query_as("SELECT next_index, end_index FROM priority_ranges WHERE id = ?").bind(priority_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!((next_index, end_index), (start + 5, start + 5), "the span shrinks to just what was carved");

        // Let the main sweep work through all of length 2.
        for _ in 0..1000 {
            let len: i64 = sqlx::query_scalar("SELECT candidate_len FROM target_progress WHERE target_id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
            if len > 2 {
                break;
            }
            claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        }
        let mut spans: Vec<(i64, i64)> = sqlx::query_as("SELECT start_index, end_index FROM ranges WHERE target_id = ? AND candidate_len = 2")
            .bind(target_id)
            .fetch_all(&pool)
            .await
            .unwrap();
        spans.sort();
        let mut covered_to = 0;
        for (lo, hi) in spans {
            assert_eq!(lo, covered_to, "length-2 candidates must be handed out with no gap or overlap");
            covered_to = hi;
        }
        assert_eq!(covered_to, 26 * 26);
    }

    /// Once the main sweep has jumped past a priority range, it never comes
    /// back for it - so removing the priority range must keep its uncarved
    /// rest as priority work instead of dropping it.
    #[tokio::test]
    async fn remove_priority_range_keeps_what_the_main_sweep_has_passed() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let config = test_config(5);
        let (target_id, priority_range_id, start) = partly_carved_priority_range(&pool, &user, &config).await;
        sqlx::query("UPDATE target_progress SET candidate_len = 3, next_index = 0 WHERE target_id = ?").bind(target_id).execute(&pool).await.unwrap();

        let removal = remove_priority_range(&pool, priority_range_id).await.unwrap().expect("exists");
        assert!(!removal.deleted);
        assert_eq!((removal.returned_to_main_sweep, removal.kept), (0, 21));

        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        let (claimed_from, claimed_start): (Option<i64>, i64) =
            sqlx::query_as("SELECT priority_range_id, start_index FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        assert_eq!((claimed_from, claimed_start), (Some(priority_range_id), start + 5), "the kept rest is still handed out");
    }

    /// The main sweep partway through a priority range's span (at the same
    /// length): what it has passed is kept, the rest goes back to it.
    #[tokio::test]
    async fn remove_priority_range_splits_around_the_main_sweep_cursor() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let config = test_config(5);
        let (target_id, priority_range_id, start) = partly_carved_priority_range(&pool, &user, &config).await;
        sqlx::query("UPDATE target_progress SET next_index = ? WHERE target_id = ?").bind(start + 10).bind(target_id).execute(&pool).await.unwrap();

        let removal = remove_priority_range(&pool, priority_range_id).await.unwrap().expect("exists");
        assert_eq!((removal.returned_to_main_sweep, removal.kept), (16, 5));
        let (next_index, end_index): (i64, i64) =
            sqlx::query_as("SELECT next_index, end_index FROM priority_ranges WHERE id = ?").bind(priority_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!((next_index, end_index), (start + 5, start + 10));
    }

    /// A priority range row with an explicit span and carving position, on a
    /// letters-alphabet target at length 2 - for priority_range_start's tests.
    async fn insert_raw_priority_range(pool: &SqlitePool, target_id: i64, start_index: i64, end_index: i64, next_index: i64) -> i64 {
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let id = sqlx::query_scalar(
            "INSERT INTO priority_ranges (target_id, priority, pattern, candidate_len, start_index, end_index, next_index, alphabet_name, alphabet, created_at) \
             VALUES (?, 1, 'x', 2, ?, ?, ?, 'letters', ?, 0) RETURNING id",
        )
        .bind(target_id)
        .bind(start_index)
        .bind(end_index)
        .bind(next_index)
        .bind(letters)
        .fetch_one(pool)
        .await
        .unwrap();
        add_priority_segment(pool, id, letters, 2, start_index.into(), end_index.into()).await;
        id
    }

    async fn start_for_spans(pool: &SqlitePool, target_id: i64, spans: &[(i64, i64)], next_index: i64) -> Result<i64, AppError> {
        let spans: Vec<(Pos, Pos)> = spans.iter().map(|&(start, end)| (start.into(), end.into())).collect();
        let mut conn = pool.acquire().await.unwrap();
        priority_range_start(&mut conn, target_id, 2, "letters", &spans, next_index.into()).await.map(|p| p as i64)
    }

    async fn start_for(pool: &SqlitePool, target_id: i64, start_index: i64, end_index: i64, next_index: i64) -> Result<i64, AppError> {
        start_for_spans(pool, target_id, &[(start_index, end_index)], next_index).await
    }

    /// The case this exists for: a finished priority range (here "M", MA-MZ
    /// = 312..338, removed after handing out MA-ME) already covers the start
    /// of a new one, which then starts right after that work instead of
    /// being rejected.
    #[tokio::test]
    async fn priority_range_start_skips_finished_work_at_its_start() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let config = test_config(5);
        let (target_id, priority_range_id, start) = partly_carved_priority_range(&pool, &user, &config).await;
        remove_priority_range(&pool, priority_range_id).await.unwrap().expect("exists");

        assert_eq!(start_for(&pool, target_id, start, start + 26, start).await.unwrap(), start + 5);
        // Also when the main sweep has already moved the starting point into that work.
        assert_eq!(start_for(&pool, target_id, start, start + 26, start + 2).await.unwrap(), start + 5);
        // And when the finished work lies entirely behind the starting point.
        assert_eq!(start_for(&pool, target_id, start, start + 26, start + 7).await.unwrap(), start + 7);
    }

    #[tokio::test]
    async fn priority_range_start_rejects_a_priority_range_still_handing_out_work() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let config = test_config(5);
        let (target_id, _, start) = partly_carved_priority_range(&pool, &user, &config).await;

        let err = start_for(&pool, target_id, start, start + 26, start).await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(msg) if msg.contains("still handing out work")));
    }

    /// Finished work can only be skipped at the start: a block in the middle
    /// would need the new range to jump over it partway through.
    #[tokio::test]
    async fn priority_range_start_rejects_finished_work_past_its_start() {
        let pool = test_pool().await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "AA", "ZZ").await;
        insert_raw_priority_range(&pool, target_id, 320, 325, 325).await;

        let err = start_for(&pool, target_id, 312, 338, 312).await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(msg) if msg.contains("can only skip such work at its start")));
    }

    /// Several finished blocks back to back at the start are all skipped;
    /// one covering everything leaves nothing to create.
    #[tokio::test]
    async fn priority_range_start_skips_consecutive_finished_work_and_rejects_nothing_left() {
        let pool = test_pool().await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "AA", "ZZ").await;
        insert_raw_priority_range(&pool, target_id, 312, 317, 317).await;
        insert_raw_priority_range(&pool, target_id, 317, 330, 330).await;

        assert_eq!(start_for(&pool, target_id, 312, 338, 312).await.unwrap(), 330);

        insert_raw_priority_range(&pool, target_id, 330, 338, 338).await;
        let err = start_for(&pool, target_id, 312, 338, 312).await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(msg) if msg.contains("nothing left")));
    }

    /// With several segments, finished work covering the whole first one
    /// moves the start on to the next segment; finished work lying in the
    /// gap between two segments doesn't overlap anything.
    #[tokio::test]
    async fn priority_range_start_steps_over_finished_work_into_its_next_segment() {
        let pool = test_pool().await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "AA", "ZZ").await;
        insert_raw_priority_range(&pool, target_id, 312, 338, 338).await; // MA-MZ, done
        insert_raw_priority_range(&pool, target_id, 350, 360, 360).await; // in the gap below, done

        assert_eq!(start_for_spans(&pool, target_id, &[(312, 338), (364, 390)], 312).await.unwrap(), 364);
    }

    /// Migration 0018 repairs priority ranges retired the old way (next_index
    /// forced to end_index, full span kept): one the main sweep hasn't
    /// reached is cut back so the sweep searches the rest, one it has passed
    /// gets its rest handed out as priority work again.
    #[tokio::test]
    async fn repair_migration_fixes_priority_ranges_retired_the_old_way() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "AA", "ZZ").await;
        sqlx::query("UPDATE target_progress SET candidate_len = 2, next_index = 0 WHERE target_id = ?").bind(target_id).execute(&pool).await.unwrap();
        let config = test_config(5);

        let m = insert_priority_range(&pool, target_id, letters, "letters", 10, "M", 2).await; // 312..338
        claim_range(&pool, &config, &user).await.unwrap().expect("carves MA-ME");
        let q = insert_priority_range(&pool, target_id, letters, "letters", 20, "Q", 2).await; // 416..442
        claim_range(&pool, &config, &user).await.unwrap().expect("carves QA-QE");
        let untouched = insert_priority_range(&pool, target_id, letters, "letters", 5, "T", 2).await; // 494..520

        // Retire M and Q the old way, and put the main sweep between them.
        sqlx::query("UPDATE priority_ranges SET next_index = end_index WHERE id IN (?, ?)").bind(m).bind(q).execute(&pool).await.unwrap();
        sqlx::query("UPDATE target_progress SET next_index = 364 WHERE target_id = ?").bind(target_id).execute(&pool).await.unwrap();

        sqlx::raw_sql(include_str!("../migrations/0018_repair_retired_priority_ranges.sql")).execute(&pool).await.unwrap();

        let span = |id: i64| {
            let pool = pool.clone();
            async move {
                sqlx::query_as::<_, (i64, i64, i64)>("SELECT start_index, next_index, end_index FROM priority_ranges WHERE id = ?")
                    .bind(id)
                    .fetch_one(&pool)
                    .await
                    .unwrap()
            }
        };
        assert_eq!(span(m).await, (312, 317, 338), "the sweep is past M: its rest is priority work again");
        assert_eq!(span(q).await, (416, 421, 421), "the sweep hasn't reached Q: cut back so the sweep searches the rest");
        assert_eq!(span(untouched).await, (494, 494, 520), "a priority range that wasn't retired is left alone");
    }

    /// Regression test: a same-length priority range can have `next_index`
    /// clamped ahead of `start_index` at creation time (see
    /// `handlers::admin_create_priority_range`, when the main sweep's cursor
    /// is already partway through that exact length) without the priority
    /// mechanism ever actually carving anything from it. That must still be
    /// treated as "untouched" and deleted outright - it deliberately checks
    /// for a referencing `ranges` row, not comparing `next_index` to
    /// `start_index`, precisely so this case works.
    #[tokio::test]
    async fn remove_priority_range_deletes_a_clamped_but_never_carved_range() {
        let pool = test_pool().await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;

        // Mimics admin_create_priority_range's same-length clamp: start_index
        // is behind next_index, but nothing has been carved from this row.
        // The cursor is 5 candidates into length 2, and this row's span (AA-AZ)
        // starts there - so its first 5 are already searched by the sweep.
        sqlx::query("UPDATE target_progress SET candidate_len = 2, next_index = 5 WHERE target_id = ?").bind(target_id).execute(&pool).await.unwrap();
        let (start_index, end_index_inclusive) = bound_indices_at_len(letters, "A", "A", 2);
        let clamped_next_index = start_index + 5;
        let priority_range_id: i64 = sqlx::query_scalar(
            "INSERT INTO priority_ranges (target_id, priority, pattern, candidate_len, start_index, end_index, next_index, alphabet_name, alphabet, created_at) \
             VALUES (?, 10, 'A', 2, ?, ?, ?, 'letters', ?, ?) RETURNING id",
        )
        .bind(target_id)
        .bind(start_index)
        .bind(end_index_inclusive + 1)
        .bind(clamped_next_index)
        .bind(letters)
        .bind(now_unix())
        .fetch_one(&pool)
        .await
        .unwrap();
        add_priority_segment(&pool, priority_range_id, letters, 2, start_index.into(), (end_index_inclusive + 1).into()).await;

        assert!(remove_priority_range(&pool, priority_range_id).await.unwrap().expect("exists").deleted);
        let remaining: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM priority_ranges WHERE id = ?").bind(priority_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(remaining, 0);
    }

    /// The core scenario migrate_priority_ranges_to_new_alphabet exists for:
    /// a priority range partway through its own carving, under an alphabet
    /// the target has since moved away from, gets its bounds and cursor
    /// translated onto the new alphabet - including recording a skip range
    /// (under the OLD alphabet) for whatever part of its own block can no
    /// longer be expressed, exactly mirroring transition_alphabet_cursor's
    /// behavior for the target's own main cursor, just scoped to this one
    /// priority range's own prefix instead of the whole target.
    #[tokio::test]
    async fn migrate_priority_ranges_to_new_alphabet_translates_a_still_open_range() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AAAAAA", "ZZZZZZ").await;
        let priority_range_id = insert_priority_range(&pool, target_id, old_alphabet, "digits_and_letters", 10, "ABC", 6).await;

        // Simulate this priority range having already carved partway through its own block.
        let cursor_index = candidate_to_index(old_alphabet, "ABC001").unwrap();
        sqlx::query("UPDATE priority_ranges SET next_index = ? WHERE id = ?").bind(cursor_index).bind(priority_range_id).execute(&pool).await.unwrap();

        let mut tx = pool.begin().await.unwrap();
        migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        let (start_index, end_index, next_index, alphabet_name, alphabet): (i64, i64, i64, String, String) =
            sqlx::query_as("SELECT start_index, end_index, next_index, alphabet_name, alphabet FROM priority_ranges WHERE id = ?")
                .bind(priority_range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(alphabet_name, "letters_only");
        assert_eq!(alphabet, new_alphabet);
        let (exp_start, exp_end_inclusive) = bound_indices_at_len(new_alphabet, "ABC", "ABC", 6);
        assert_eq!(start_index, exp_start, "bounds must be recomputed for the same prefix under the new alphabet");
        assert_eq!(end_index, exp_end_inclusive + 1);
        assert_eq!(next_index, candidate_to_index(new_alphabet, "ABCAAA").unwrap(), "resumes at the first candidate expressible in the new alphabet");

        let (skip_status, skip_alphabet, skip_priority_range_id): (String, String, Option<i64>) =
            sqlx::query_as("SELECT status, alphabet, priority_range_id FROM ranges WHERE target_id = ? AND status = 'skipped'")
                .bind(target_id)
                .fetch_one(&pool)
                .await
                .expect("the old-alphabet remainder of this priority range's own block must be persisted as a skipped range");
        assert_eq!(skip_status, "skipped");
        assert_eq!(skip_alphabet, old_alphabet, "the skip is denominated in the OLD alphabet, not the target's current one");
        assert_eq!(skip_priority_range_id, Some(priority_range_id));
    }

    /// A priority range whose prefix uses a character the new alphabet
    /// simply doesn't have at all can't be translated to anything - it must
    /// be permanently retired (no more fresh work offered) rather than
    /// panicking or silently producing nonsense bounds.
    #[tokio::test]
    async fn migrate_priority_ranges_to_new_alphabet_retires_a_prefix_the_new_alphabet_cant_express() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // no digits at all
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "A0", "ZZ").await;
        let priority_range_id = insert_priority_range(&pool, target_id, old_alphabet, "digits_and_letters", 10, "A0", 2).await;

        let mut tx = pool.begin().await.unwrap();
        migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        let (next_index, end_index, alphabet_name): (i64, i64, String) =
            sqlx::query_as("SELECT next_index, end_index, alphabet_name FROM priority_ranges WHERE id = ?")
                .bind(priority_range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(next_index, end_index, "must be permanently retired, not translated");
        assert_eq!(alphabet_name, "digits_and_letters", "left as a historical record under the alphabet it actually holds candidates in");
    }

    /// A priority range that was already fully exhausted before the patch
    /// still needs its start/end translated (even though it has no cursor
    /// left to resume) - otherwise the main sweep's exclusion would forget
    /// about the space it already covered, once it looks under the new
    /// alphabet's name.
    #[tokio::test]
    async fn migrate_priority_ranges_to_new_alphabet_translates_bounds_of_an_already_exhausted_range() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AAA", "ZZZ").await;
        let priority_range_id = insert_priority_range(&pool, target_id, old_alphabet, "digits_and_letters", 10, "ABC", 3).await;
        sqlx::query("UPDATE priority_ranges SET next_index = end_index WHERE id = ?").bind(priority_range_id).execute(&pool).await.unwrap();

        let mut tx = pool.begin().await.unwrap();
        migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        let (start_index, end_index, next_index, alphabet_name): (i64, i64, i64, String) =
            sqlx::query_as("SELECT start_index, end_index, next_index, alphabet_name FROM priority_ranges WHERE id = ?")
                .bind(priority_range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(alphabet_name, "letters_only");
        let (exp_start, exp_end_inclusive) = bound_indices_at_len(new_alphabet, "ABC", "ABC", 3);
        assert_eq!(start_index, exp_start);
        assert_eq!(end_index, exp_end_inclusive + 1);
        assert_eq!(next_index, end_index, "stays exhausted under the new bounds too");
    }

    /// A row from before prefixes were stored (`prefix` is `""` since
    /// migration 0021) is left entirely untouched rather than guessed at.
    #[tokio::test]
    async fn migrate_priority_ranges_to_new_alphabet_leaves_a_prefix_less_row_alone() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "A", "Z").await;
        let priority_range_id = insert_priority_range(&pool, target_id, old_alphabet, "digits_and_letters", 10, "A", 1).await;
        sqlx::query("UPDATE priority_ranges SET prefix = '' WHERE id = ?").bind(priority_range_id).execute(&pool).await.unwrap();

        let mut tx = pool.begin().await.unwrap();
        migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        let alphabet_name: String =
            sqlx::query_scalar("SELECT alphabet_name FROM priority_ranges WHERE id = ?").bind(priority_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(alphabet_name, "digits_and_letters", "left completely untouched with no prefix to translate it by");
    }

    /// The segments of `priority_range_id`, as (start, end) pairs.
    async fn priority_segments(pool: &SqlitePool, priority_range_id: i64) -> Vec<(i64, i64)> {
        sqlx::query_as("SELECT start_index, end_index FROM priority_range_segments WHERE priority_range_id = ? ORDER BY start_index")
            .bind(priority_range_id)
            .fetch_all(pool)
            .await
            .unwrap()
    }

    /// A row from before migration 0021 is one of several its pattern was
    /// expanded into: it's translated by its own prefix, not the pattern,
    /// which would make it take over its siblings' candidates too.
    #[tokio::test]
    async fn migrate_priority_ranges_to_new_alphabet_translates_a_pre_0021_row_by_its_prefix() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AA", "ZZ").await;
        let priority_range_id = insert_priority_range(&pool, target_id, old_alphabet, "digits_and_letters", 10, "B", 2).await;
        sqlx::query("UPDATE priority_ranges SET pattern = '[A-C]', prefix = 'B' WHERE id = ?").bind(priority_range_id).execute(&pool).await.unwrap();

        let mut tx = pool.begin().await.unwrap();
        migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        let (start, end_inclusive) = bound_indices_at_len(new_alphabet, "B", "B", 2);
        assert_eq!(priority_segments(&pool, priority_range_id).await, vec![(start, end_inclusive + 1)]);
    }

    /// Re-expanded under the new alphabet, a pattern can merge into fewer
    /// segments: "0" and "A" aren't next to each other with digits in the
    /// alphabet, but without them only "A" is left.
    #[tokio::test]
    async fn migrate_priority_ranges_to_new_alphabet_re_expands_the_pattern() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AA", "ZZ").await;
        let priority_range_id = insert_priority_range(&pool, target_id, old_alphabet, "digits_and_letters", 10, "[0A-B]", 2).await;
        assert_eq!(priority_segments(&pool, priority_range_id).await.len(), 2);

        let mut tx = pool.begin().await.unwrap();
        migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        assert_eq!(priority_segments(&pool, priority_range_id).await, vec![(0, 2 * 26)]);
        let (start_index, next_index, end_index): (i64, i64, i64) =
            sqlx::query_as("SELECT start_index, next_index, end_index FROM priority_ranges WHERE id = ?").bind(priority_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!((start_index, next_index, end_index), (0, 0, 2 * 26));
    }

    /// Claims until a claim comes back at a length past `len` (carving goes
    /// on through every length up to the maximum, so it never simply runs
    /// out), returning the claims up to there.
    async fn claim_through_len(pool: &SqlitePool, config: &RangeConfig, user: &User, len: i64) -> Vec<ClaimResponse> {
        let mut claims = Vec::new();
        while let Some(claim) = claim_range(pool, config, user).await.unwrap() {
            let claim_len: i64 = sqlx::query_scalar("SELECT candidate_len FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(pool).await.unwrap();
            if claim_len > len {
                break;
            }
            claims.push(claim);
        }
        claims
    }

    /// A pattern covering two separate stretches is one priority range: each
    /// chunk stays within one of them, and the main sweep afterwards gets
    /// exactly what's between and around them.
    #[tokio::test]
    async fn claim_range_carves_a_multi_segment_priority_range_one_segment_at_a_time() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;
        let priority_range_id = insert_priority_range(&pool, target_id, letters, "letters", 10, "[A-BX]", 1).await;

        let mut claimed = Vec::new();
        for claim in claim_through_len(&pool, &test_config(30), &user, 1).await {
            let (from,): (Option<i64>,) =
                sqlx::query_as("SELECT priority_range_id FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
            claimed.push((claim.lower_bound_filename, claim.upper_bound_filename, from));
        }
        let expected = [
            ("PREA.SUF", "PREB.SUF", Some(priority_range_id)),
            ("PREX.SUF", "PREX.SUF", Some(priority_range_id)),
            ("PREC.SUF", "PREW.SUF", None),
            ("PREY.SUF", "PREZ.SUF", None),
        ];
        let expected: Vec<(String, String, Option<i64>)> = expected.iter().map(|&(a, b, c)| (a.to_string(), b.to_string(), c)).collect();
        assert_eq!(claimed, expected);
    }

    /// `(start, end)` of every `skipped` row of `target_id` at `len`, sorted.
    async fn skipped_rows(pool: &SqlitePool, target_id: i64, len: i64) -> Vec<(i64, i64)> {
        let mut rows: Vec<(i64, i64)> =
            sqlx::query_as("SELECT start_index, end_index FROM ranges WHERE target_id = ? AND candidate_len = ? AND status = 'skipped'")
                .bind(target_id)
                .bind(len)
                .fetch_all(pool)
                .await
                .unwrap();
        rows.sort();
        rows
    }

    /// A skip range lying partly inside a priority range: its row is
    /// written once, when it's created, and neither the main sweep nor the
    /// priority range records any of it again - nor hands any of it out.
    #[tokio::test]
    async fn a_skip_range_reaching_into_a_priority_range_is_recorded_once() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_skip_range(&pool, letters, "A", "Z", "[C-F]").await;
        insert_priority_range(&pool, target_id, letters, "custom", 10, "[E-H]", 1).await;
        assert_eq!(skipped_rows(&pool, target_id, 1).await, vec![(2, 6)], "written right away");

        claim_through_len(&pool, &test_config(30), &user, 1).await;
        assert_eq!(skipped_rows(&pool, target_id, 1).await, vec![(2, 6)], "and never again");

        let mut spans: Vec<(i64, i64)> = sqlx::query_as("SELECT start_index, end_index FROM ranges WHERE target_id = ? AND candidate_len = 1")
            .bind(target_id)
            .fetch_all(&pool)
            .await
            .unwrap();
        spans.sort();
        let mut covered_to = 0;
        for (lo, hi) in spans {
            assert_eq!(lo, covered_to, "every candidate handed out or skipped exactly once");
            covered_to = hi;
        }
        assert_eq!(covered_to, 26);
    }

    /// A new skip range's rows cover only what no range covers yet, within
    /// the target's bounds - here the pattern is C-H, C-D is already handed
    /// out, and the target stops at G.
    #[tokio::test]
    async fn create_skip_range_writes_rows_only_for_uncovered_candidates_within_bounds() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "G").await;
        claim_range(&pool, &test_config(4), &user).await.unwrap().expect("A-D");

        add_skip_range(&pool, target_id, "[C-H]", 1).await;
        assert_eq!(skipped_rows(&pool, target_id, 1).await, vec![(4, 7)], "E-G");

        let claims = claim_through_len(&pool, &test_config(30), &user, 1).await;
        assert!(claims.is_empty(), "nothing left at length 1: {:?}", claims.iter().map(|c| &c.lower_bound_filename).collect::<Vec<_>>());
    }

    /// Removing a skip range withdraws the part of its rows the main sweep
    /// hasn't reached yet - so the sweep searches it - and requeues the
    /// part it has passed as pending, splitting a row where the sweep is
    /// inside it. The pending part is handed out before the sweep goes on.
    #[tokio::test]
    async fn remove_skip_range_withdraws_what_the_sweep_has_not_reached_and_requeues_the_rest() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;
        let skip_range_id = add_skip_range(&pool, target_id, "[C-J]", 1).await;
        // As if the sweep had got to F (it would normally jump the whole span).
        sqlx::query("UPDATE target_progress SET next_index = 5 WHERE target_id = ?").bind(target_id).execute(&pool).await.unwrap();

        assert_eq!(remove_skip_range(&pool, skip_range_id).await.unwrap(), Some(3), "C-E is requeued");
        assert_eq!(skipped_rows(&pool, target_id, 1).await, vec![], "nothing stays skipped");
        let claim = claim_range(&pool, &test_config(30), &user).await.unwrap().expect("the requeued C-E");
        assert_eq!((claim.lower_bound_filename.as_str(), claim.upper_bound_filename.as_str()), ("PREC.SUF", "PREE.SUF"));
        let claim = claim_range(&pool, &test_config(30), &user).await.unwrap().expect("F onwards");
        assert_eq!((claim.lower_bound_filename.as_str(), claim.upper_bound_filename.as_str()), ("PREF.SUF", "PREZ.SUF"));
    }

    /// Removing a skip range deletes it whether or not it skipped anything,
    /// and it stops skipping.
    #[tokio::test]
    async fn remove_skip_range_deletes_both_an_unused_and_a_used_one() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "ZZ").await;
        let used = add_skip_range(&pool, target_id, "[C-D]", 1).await;
        let unused = add_skip_range(&pool, target_id, "A", 2).await;

        claim_range(&pool, &test_config(30), &user).await.unwrap().expect("A-B");

        assert_eq!(remove_skip_range(&pool, unused).await.unwrap(), Some(0));
        assert_eq!(remove_skip_range(&pool, used).await.unwrap(), Some(2), "the sweep had jumped C-D");
        assert_eq!(remove_skip_range(&pool, 999_999).await.unwrap(), None);

        let left: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM skip_ranges WHERE target_id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        assert_eq!(left, 0);

        // C-D is searched after all, and length 2 is no longer skipped at "A".
        let claims: Vec<String> = claim_through_len(&pool, &test_config(1000), &user, 2).await.into_iter().map(|c| c.lower_bound_filename).collect();
        assert!(claims.contains(&"PREC.SUF".to_string()), "{claims:?}");
        assert!(claims.contains(&"PREAA.SUF".to_string()), "{claims:?}");
    }

    /// A pending row bigger than a chunk - what a removed skip range can
    /// requeue - is handed out a chunk at a time, the rest staying pending.
    #[tokio::test]
    async fn claim_range_hands_out_a_big_pending_row_a_chunk_at_a_time() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "letters", letters, "A", "Z").await;
        let skip_range_id = add_skip_range(&pool, target_id, "[C-J]", 1).await;
        sqlx::query("UPDATE target_progress SET next_index = 10 WHERE target_id = ?").bind(target_id).execute(&pool).await.unwrap();
        assert_eq!(remove_skip_range(&pool, skip_range_id).await.unwrap(), Some(8), "all of C-J");

        let mut claimed = Vec::new();
        for _ in 0..3 {
            let claim = claim_range(&pool, &test_config(3), &user).await.unwrap().expect("a chunk of C-J");
            claimed.push((claim.lower_bound_filename, claim.upper_bound_filename));
        }
        let expected = [("PREC.SUF", "PREE.SUF"), ("PREF.SUF", "PREH.SUF"), ("PREI.SUF", "PREJ.SUF")];
        assert_eq!(claimed, expected.map(|(lo, hi)| (lo.to_string(), hi.to_string())));
    }

    /// An alphabet patch re-expands a skip range's pattern under the new
    /// alphabet, and rewrites its `skipped` rows that the sweep hasn't
    /// reached yet in the new alphabet.
    #[tokio::test]
    async fn migrate_skip_ranges_to_new_alphabet_re_expands_the_pattern() {
        let pool = test_pool().await;
        let old_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let target_id = insert_target_with_alphabet(&pool, "digits_and_letters", old_alphabet, "AA", "ZZ").await;
        let skip_range_id = add_skip_range(&pool, target_id, "[5B]", 2).await;
        let dead_id = add_skip_range(&pool, target_id, "0", 2).await;
        let rows = |id: i64| {
            let pool = pool.clone();
            async move {
                sqlx::query_as::<_, (String, i64, i64)>("SELECT alphabet, start_index, end_index FROM ranges WHERE skip_range_id = ?")
                    .bind(id)
                    .fetch_all(&pool)
                    .await
                    .unwrap()
            }
        };
        let b = candidate_to_index(old_alphabet, "B0").unwrap();
        assert_eq!(rows(skip_range_id).await, vec![(old_alphabet.to_string(), b, b + 36)], "only B is within the target's bounds");

        let mut tx = pool.begin().await.unwrap();
        migrate_skip_ranges_to_new_alphabet(&mut tx, target_id, "letters_only", new_alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();

        assert_eq!(rows(skip_range_id).await, vec![(new_alphabet.to_string(), 26, 52)], "the sweep hadn't reached the old row, so it's rewritten");
        assert!(rows(dead_id).await.is_empty());

        let segments = |id: i64| {
            let pool = pool.clone();
            async move {
                sqlx::query_as::<_, (i64, i64)>("SELECT start_index, end_index FROM skip_range_segments WHERE skip_range_id = ?")
                    .bind(id)
                    .fetch_all(&pool)
                    .await
                    .unwrap()
            }
        };
        assert_eq!(segments(skip_range_id).await, vec![(26, 52)], "only B is left");
        assert!(segments(dead_id).await.is_empty(), "nothing left to skip");
        let alphabet_name: String = sqlx::query_scalar("SELECT alphabet_name FROM skip_ranges WHERE id = ?").bind(skip_range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(alphabet_name, "letters_only");
    }

    /// Moves `target_id`'s main sweep to `candidate` (in size49).
    async fn set_sweep(pool: &SqlitePool, target_id: i64, candidate: &str) {
        sqlx::query("UPDATE target_progress SET candidate_len = ?, next_block = 0, next_index = ? WHERE target_id = ?")
            .bind(candidate.chars().count() as i64)
            .bind(candidate_to_index(DEFAULT, candidate).unwrap())
            .bind(target_id)
            .execute(pool)
            .await
            .unwrap();
    }

    /// Patches `target_id`'s alphabet the way `admin_patch_target` does, as
    /// far as its skip ranges go.
    async fn switch_alphabet(pool: &SqlitePool, target_id: i64, name: &str, alphabet: &str) {
        let mut tx = pool.begin().await.unwrap();
        sqlx::query("UPDATE targets SET alphabet_name = ?, alphabet = ? WHERE id = ?").bind(name).bind(alphabet).bind(target_id).execute(&mut *tx).await.unwrap();
        migrate_skip_ranges_to_new_alphabet(&mut tx, target_id, name, alphabet, now_unix()).await.unwrap();
        tx.commit().await.unwrap();
    }

    async fn skipped_candidates(pool: &SqlitePool, target_id: i64) -> Vec<(String, String, String)> {
        sqlx::query_as::<_, (String, String, i64, i64, i64)>(
            "SELECT alphabet_name, alphabet, candidate_len, start_index, end_index FROM ranges WHERE target_id = ? AND status = 'skipped' \
             ORDER BY alphabet_name, start_index",
        )
        .bind(target_id)
        .fetch_all(pool)
        .await
        .unwrap()
        .into_iter()
        .map(|(name, alphabet, len, start, end)| (name, index_to_candidate(&alphabet, start, len), index_to_candidate(&alphabet, end - 1, len)))
        .collect()
    }

    /// Switching alphabets back and forth after the sweep has passed a skip
    /// range's length must leave its rows as they were: writing them again
    /// in the other alphabet showed them, on the dashboard, with gaps
    /// between them where the original alphabet's completed ranges were.
    #[tokio::test]
    async fn migrate_skip_ranges_to_new_alphabet_writes_nothing_where_the_sweep_has_been() {
        let pool = test_pool().await;
        let target_id = insert_target(&pool, "  ", "__").await;
        add_skip_range(&pool, target_id, "[AC]", 2).await;
        let before = skipped_candidates(&pool, target_id).await;
        set_sweep(&pool, target_id, "   ").await;

        switch_alphabet(&pool, target_id, "size42", SIZE42).await;
        assert_eq!(skipped_candidates(&pool, target_id).await, before);
        switch_alphabet(&pool, target_id, "size49", DEFAULT).await;
        assert_eq!(skipped_candidates(&pool, target_id).await, before);
    }

    /// At the length the sweep is on, only what's ahead of it is rewritten
    /// in the new alphabet.
    #[tokio::test]
    async fn migrate_skip_ranges_to_new_alphabet_rewrites_only_what_is_ahead_of_the_sweep() {
        let pool = test_pool().await;
        let target_id = insert_target(&pool, "  ", "__").await;
        add_skip_range(&pool, target_id, "[AC]", 2).await;
        // '!' isn't in size42: the sweep carries over to "B(", the first
        // size42 candidate after it.
        set_sweep(&pool, target_id, "B!").await;

        switch_alphabet(&pool, target_id, "size42", SIZE42).await;
        let row = |name: &str, first: &str, last: &str| (name.to_string(), first.to_string(), last.to_string());
        assert_eq!(skipped_candidates(&pool, target_id).await, vec![row("size42", "C ", "C_"), row("size49", "A ", "A_")]);
    }

    /// A target's max_backslash_count must reach the client via ClaimResponse
    /// unchanged, since namebreak itself (not the server) is what enforces it.
    #[tokio::test]
    async fn claim_includes_the_targets_max_backslash_count() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let target_id = insert_target(&pool, &lower, &upper).await;
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

    /// A lease is the same fixed length for every range - however big, and
    /// however fast its claimant - counted from the last sign of life: the
    /// claim, then each heartbeat. A heartbeat must also use the current
    /// setting rather than whatever lease length was stored at claim time.
    #[tokio::test]
    async fn lease_is_fixed_and_renewed_from_the_last_heartbeat() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        insert_target(&pool, &lower, &upper).await;

        // default_rate_per_sec is 1.0, so a size-based lease would be enormous.
        let config = test_config(space_size(DEFAULT, 3));
        let before = now_unix();
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert_eq!(claim.lease_seconds, config.lease_seconds);
        let lease_expires_at: i64 =
            sqlx::query_scalar("SELECT lease_expires_at FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        assert!((before + config.lease_seconds..=now_unix() + config.lease_seconds).contains(&lease_expires_at));

        // A range claimed under an older, much longer lease setting.
        sqlx::query("UPDATE ranges SET lease_seconds = 999999, lease_expires_at = ? WHERE id = ?")
            .bind(now_unix() + 999999)
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();
        let before = now_unix();
        let outcome = heartbeat_range(&pool, &config, &user, claim.range_id, None).await.unwrap();
        assert_eq!(outcome.lease_seconds, config.lease_seconds);
        let lease_expires_at: i64 =
            sqlx::query_scalar("SELECT lease_expires_at FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        assert!((before + config.lease_seconds..=now_unix() + config.lease_seconds).contains(&lease_expires_at));
    }

    #[tokio::test]
    async fn claim_includes_the_targets_prune_unopened_brackets() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let target_id = insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2) / 2);
        let before = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert!(!before.prune_unopened_brackets, "off unless the target asks for it");

        sqlx::query("UPDATE targets SET prune_unopened_brackets = 1 WHERE id = ?").bind(target_id).execute(&pool).await.unwrap();
        let after = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert!(after.prune_unopened_brackets);
    }

    #[tokio::test]
    async fn claim_includes_the_targets_prune_whole_candidate() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let target_id = insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2) / 2);
        let before = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert!(!before.prune_whole_candidate, "off unless the target asks for it");

        sqlx::query("UPDATE targets SET prune_whole_candidate = 1 WHERE id = ?").bind(target_id).execute(&pool).await.unwrap();
        let after = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert!(after.prune_whole_candidate);
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
        let (lower, upper) = full_bounds(DEFAULT, 3);
        insert_target(&pool, &lower, &upper).await;

        // Small enough that the target's space gets carved into (at least) two ranges.
        let config = test_config(space_size(DEFAULT, 3) / 2);

        let claim_a = claim_range(&pool, &config, &finder).await.unwrap().expect("first range available");
        let claim_b = claim_range(&pool, &config, &other).await.unwrap().expect("second range available");
        assert_ne!(claim_a.range_id, claim_b.range_id);

        // Before anything is found: heartbeat behaves normally.
        let before = heartbeat_range(&pool, &config, &other, claim_b.range_id, None).await.unwrap();
        assert!(!before.range_released);

        // `finder` reports a match, solving the target.
        let outcome = complete_range(&pool, &config, &finder, claim_a.range_id, true, Some("PRE???.SUF".into()), 1.0, 1)
            .await
            .unwrap();
        assert!(outcome.target_solved);

        // `other`'s next heartbeat must now signal abort, and its range should be
        // closed out rather than left dangling.
        let after = heartbeat_range(&pool, &config, &other, claim_b.range_id, None).await.unwrap();
        assert!(after.range_released);

        let status: String = sqlx::query_scalar("SELECT status FROM ranges WHERE id = ?")
            .bind(claim_b.range_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "completed");
    }

    /// Heartbeating alone would otherwise renew lease_expires_at forever,
    /// holding a range hostage indefinitely for a client that's stopped
    /// actually making progress on it (most notably: deliberately paused,
    /// which has no dedicated signal of its own - see HeartbeatRequest's own
    /// doc comment - but reads identically to any other stall). Once
    /// stall_release_seconds has passed with no heartbeat reporting genuinely
    /// new progress, the next one must release the claim back to pending
    /// (here: no progress was ever checkpointed, so the whole range goes back
    /// as-is) and say so via range_released, instead of quietly renewing the
    /// lease again.
    #[tokio::test]
    async fn heartbeat_releases_a_range_stalled_past_the_release_threshold() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "staller").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        // Claiming already starts the clock (last_progress_at defaults to
        // assigned_at) - a heartbeat with nothing new to report yet is not
        // itself released, and the lease is still renewed normally.
        let first = heartbeat_range(&pool, &config, &user, claim.range_id, None).await.unwrap();
        assert!(!first.range_released);

        let (status, last_progress_at, lease_expires_at): (String, Option<i64>, Option<i64>) =
            sqlx::query_as("SELECT status, last_progress_at, lease_expires_at FROM ranges WHERE id = ?")
                .bind(claim.range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "in_progress");
        assert!(last_progress_at.is_some());
        assert!(lease_expires_at.unwrap() > now_unix(), "heartbeating must still renew the lease even while stalled");

        // Simulate that no heartbeat has reported new progress for longer
        // than stall_release_seconds.
        sqlx::query("UPDATE ranges SET last_progress_at = ? WHERE id = ?")
            .bind(now_unix() - config.stall_release_seconds - 1)
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();

        let second = heartbeat_range(&pool, &config, &user, claim.range_id, None).await.unwrap();
        assert!(second.range_released, "a heartbeat past the release threshold must release the claim");

        let (status, assigned_user_id, last_progress_at): (String, Option<i64>, Option<i64>) =
            sqlx::query_as("SELECT status, assigned_user_id, last_progress_at FROM ranges WHERE id = ?")
                .bind(claim.range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "pending", "no progress was ever checkpointed, so the whole range goes back as-is");
        assert!(assigned_user_id.is_none());
        assert!(last_progress_at.is_none());

        // And it's genuinely claimable again, by anyone.
        let other = insert_user(&pool, "other").await;
        let reclaimed = claim_range(&pool, &config, &other).await.unwrap().expect("released range should be claimable again");
        assert_eq!(reclaimed.range_id, claim.range_id);

        // A further heartbeat against the now-released range_id, from the
        // original (no longer owning) user, must be rejected exactly like any
        // other stale-ownership heartbeat - not a special case.
        let result = heartbeat_range(&pool, &config, &user, claim.range_id, None).await;
        assert!(matches!(result, Err(AppError::Conflict(_))));
    }

    /// A heartbeat reporting genuinely new progress must reset the stall
    /// clock, not just pause advancing it - otherwise a range that stalls,
    /// recovers, and later stalls again could have its *second* stall
    /// released almost immediately using time accrued from the first,
    /// unrelated one. Repeating an already-known match, by contrast, must
    /// *not* reset it - that's the only signal a genuinely stalled/paused
    /// client (which has nothing new to report) ever produces.
    #[tokio::test]
    async fn heartbeat_stall_clock_resets_on_new_progress_but_not_a_repeated_report() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "worker").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 3));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        let midpoint_index = space_size(DEFAULT, 3) / 2;
        let midpoint_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, midpoint_index, 3));
        heartbeat_range(&pool, &config, &user, claim.range_id, Some(midpoint_filename.clone())).await.unwrap();

        // Back the clock right up against the release threshold - the very
        // next heartbeat would release the claim if nothing resets it.
        sqlx::query("UPDATE ranges SET last_progress_at = ? WHERE id = ?")
            .bind(now_unix() - config.stall_release_seconds + 5)
            .bind(claim.range_id)
            .execute(&pool)
            .await
            .unwrap();

        // Repeating the same match (a paused/stuck client's only heartbeat
        // shape) must NOT reset the clock.
        let repeated = heartbeat_range(&pool, &config, &user, claim.range_id, Some(midpoint_filename)).await.unwrap();
        assert!(!repeated.range_released, "still under the threshold, even though nothing reset the clock");
        let last_progress_at: i64 = sqlx::query_scalar("SELECT last_progress_at FROM ranges WHERE id = ?")
            .bind(claim.range_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(last_progress_at < now_unix() - config.stall_release_seconds + 10, "a repeated report must not look like new progress");

        // A genuinely later match, by contrast, must reset it.
        let later_index = midpoint_index + 1000;
        let later_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, later_index, 3));
        let advanced = heartbeat_range(&pool, &config, &user, claim.range_id, Some(later_filename)).await.unwrap();
        assert!(!advanced.range_released);
        let last_progress_at: i64 = sqlx::query_scalar("SELECT last_progress_at FROM ranges WHERE id = ?")
            .bind(claim.range_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(last_progress_at > now_unix() - config.stall_release_seconds + 10, "genuinely new progress must reset the clock");
    }

    /// Different targets can share the same hash_a/hash_b (e.g. the same
    /// underlying file cataloged under more than one naming convention) -
    /// finding it via one must solve every other still-active target with
    /// that same hash pair too, crediting the same finder and filename,
    /// while a target with a *different* hash pair is left untouched.
    #[tokio::test]
    async fn complete_range_solves_every_target_sharing_the_same_hash_pair() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "finder").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);

        let sibling_a = insert_target_with_hash(&pool, 42, 99, &lower, &upper).await;
        let sibling_b = insert_target_with_hash(&pool, 42, 99, &lower, &upper).await;
        let unrelated = insert_target_with_hash(&pool, 42, 100, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert_eq!(claim.target_id, sibling_a, "the oldest active target should be claimed first");

        let outcome = complete_range(&pool, &config, &user, claim.range_id, true, Some("PREXY.SUF".into()), 1.0, 1).await.unwrap();
        assert!(outcome.target_solved);

        for (target_id, should_be_solved) in [(sibling_a, true), (sibling_b, true), (unrelated, false)] {
            let (status, found_filename, found_by_user_id): (String, Option<String>, Option<i64>) =
                sqlx::query_as("SELECT status, found_filename, found_by_user_id FROM targets WHERE id = ?")
                    .bind(target_id)
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            if should_be_solved {
                assert_eq!(status, "solved");
                assert_eq!(found_filename, Some("PREXY.SUF".to_string()));
                assert_eq!(found_by_user_id, Some(user.id));
            } else {
                assert_eq!(status, "active", "a target with a different hash pair must be untouched");
                assert!(found_filename.is_none());
            }
        }
    }

    /// The gap this closes: a genuine "both hashes matched" find must never
    /// be lost just because the reporting worker's lease already expired
    /// (e.g. a network blip outlasting the heartbeat interval) and the range
    /// was reassigned to someone else before the original finder could
    /// report in. `found: true` always solves the target, crediting whoever
    /// actually reported it - regardless of who currently owns the range.
    #[tokio::test]
    async fn complete_range_accepts_a_late_found_report_even_after_losing_the_range() {
        let pool = test_pool().await;
        let finder = insert_user(&pool, "finder").await;
        let other = insert_user(&pool, "other").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let target_id = insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &finder).await.unwrap().expect("work available");

        // The lease expires and the range is reassigned to someone else
        // before `finder` can report its genuine find.
        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?").bind(claim.range_id).execute(&pool).await.unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);
        let reclaimed = claim_range(&pool, &config, &other).await.unwrap().expect("reassigned to someone else");
        assert_eq!(reclaimed.range_id, claim.range_id, "reassigned onto the same row, since nothing was ever checkpointed");

        let outcome = complete_range(&pool, &config, &finder, claim.range_id, true, Some("PREXY.SUF".into()), 1.0, 1).await.unwrap();
        assert!(outcome.target_solved, "a genuine find must never be rejected just because the range moved on");

        let (status, found_filename, found_by_user_id): (String, Option<String>, Option<i64>) =
            sqlx::query_as("SELECT status, found_filename, found_by_user_id FROM targets WHERE id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        assert_eq!(status, "solved");
        assert_eq!(found_filename, Some("PREXY.SUF".to_string()));
        assert_eq!(found_by_user_id, Some(finder.id), "credited to whoever actually reported the find, not whoever currently holds the range");
    }

    /// A `found: false` report, unlike a genuine find, carries nothing worth
    /// preserving once the range has moved on to someone else - it stays
    /// rejected as stale, exactly as before this change.
    #[tokio::test]
    async fn complete_range_still_rejects_a_not_found_report_from_a_non_owner() {
        let pool = test_pool().await;
        let finder = insert_user(&pool, "finder").await;
        let other = insert_user(&pool, "other").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &finder).await.unwrap().expect("work available");

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?").bind(claim.range_id).execute(&pool).await.unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);
        claim_range(&pool, &config, &other).await.unwrap().expect("reassigned to someone else");

        let result = complete_range(&pool, &config, &finder, claim.range_id, false, None, 1.0, 1).await;
        assert!(matches!(result, Err(AppError::Conflict(_))));
    }

    /// A client that lost its lease but kept searching may finish the whole
    /// range offline, so its first contact on return is /complete rather than
    /// a heartbeat. If nobody else has claimed the range meanwhile, that
    /// report must be accepted - closing out the whole range, credited to
    /// this client - rather than rejected and the range redone.
    #[tokio::test]
    async fn complete_range_accepts_a_not_found_report_for_its_own_released_but_unclaimed_range() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "flaky").await;
        let other = insert_user(&pool, "other").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let target_id = insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        let quarter_index = space_size(DEFAULT, 2) / 4;
        let quarter_filename = format!("PRE{}.SUF", index_to_candidate(DEFAULT, quarter_index, 2));
        heartbeat_range(&pool, &config, &user, claim.range_id, Some(quarter_filename)).await.unwrap();

        sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?").bind(claim.range_id).execute(&pool).await.unwrap();
        assert_eq!(reclaim_expired(&pool).await.unwrap(), 1);

        // Some other user reporting against the pending row is still stale.
        let result = complete_range(&pool, &config, &other, claim.range_id, false, None, 1.0, 1).await;
        assert!(matches!(result, Err(AppError::Conflict(_))));

        complete_range(&pool, &config, &user, claim.range_id, false, None, 1.0, 1).await.unwrap();

        let (status, start, end, assigned, worker): (String, i64, i64, Option<i64>, Option<i64>) =
            sqlx::query_as("SELECT status, start_index, end_index, assigned_user_id, last_assigned_user_id FROM ranges WHERE id = ?")
                .bind(claim.range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "completed");
        assert_eq!((start, end), (0, space_size(DEFAULT, 2)), "the whole range, not just the checkpointed part");
        assert_eq!(assigned, Some(user.id));
        assert_eq!(worker, Some(user.id));

        let pending: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE target_id = ? AND status = 'pending'")
            .bind(target_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(pending, 0, "nothing left over to hand out again");
    }

    #[tokio::test]
    async fn delete_target_removes_it_and_its_ranges_and_progress() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let target_id = insert_target(&pool, &lower, &upper).await;

        let config = test_config(space_size(DEFAULT, 2));
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");

        assert!(delete_target(&pool, target_id).await.unwrap(), "should report the target existed");

        let target_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM targets WHERE id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        let progress_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM target_progress WHERE target_id = ?").bind(target_id).fetch_one(&pool).await.unwrap();
        let range_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ranges WHERE id = ?").bind(claim.range_id).fetch_one(&pool).await.unwrap();
        assert_eq!(target_count, 0);
        assert_eq!(progress_count, 0);
        assert_eq!(range_count, 0);

        // Deleting an id that was never there (or already deleted) is reported
        // as such, not as an error.
        assert!(!delete_target(&pool, target_id).await.unwrap());
    }

    async fn set_priority(pool: &SqlitePool, target_id: i64, priority: i64) {
        sqlx::query("UPDATE targets SET priority = ? WHERE id = ?").bind(priority).bind(target_id).execute(pool).await.unwrap();
    }

    /// Priority must win over creation order: `a` is created (and so would
    /// win on age alone) before `b` is given a higher priority, but `b` must
    /// still be claimed first - and once `b` no longer has any claimable work
    /// (simulated here by pausing it, rather than actually exhausting its
    /// space - a "full bounds" target's space isn't capped at the length its
    /// bounds happen to be written at, it keeps growing up to
    /// `max_supported_len`, so genuinely exhausting one in a test is
    /// impractical), the claim must fall through to `a`.
    #[tokio::test]
    async fn claim_range_prefers_the_higher_priority_target_over_the_older_one() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        let target_a = insert_target(&pool, &lower, &upper).await;
        let target_b = insert_target(&pool, &lower, &upper).await;
        set_priority(&pool, target_b, 10).await;

        let config = test_config(10);

        let first = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert_eq!(first.target_id, target_b, "the higher-priority target must be claimed first even though it's the newer one");

        sqlx::query("UPDATE targets SET status = 'paused' WHERE id = ?").bind(target_b).execute(&pool).await.unwrap();

        let second = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert_eq!(second.target_id, target_a, "falls through to the lower-priority target once the higher one no longer has claimable work");
    }

    /// A higher-priority target's own unclaimed fresh space must win out over
    /// a *lower*-priority target's leftover pending range - priority is
    /// decided target-by-target before either kind of claimable work is
    /// considered, not by picking whichever pending range happens to be
    /// oldest across every target.
    #[tokio::test]
    async fn claim_range_prefers_higher_priority_fresh_space_over_lower_priority_pending_leftover() {
        let pool = test_pool().await;
        let user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 3);
        let target_low = insert_target(&pool, &lower, &upper).await;
        let target_high = insert_target(&pool, &lower, &upper).await;
        set_priority(&pool, target_high, 10).await;

        // Simulate a leftover pending range on the low-priority target (e.g. the
        // unsearched remainder of a reclaimed lease) without needing to run a
        // whole claim/heartbeat/expire cycle to produce one.
        sqlx::query(
            "INSERT INTO ranges (target_id, candidate_len, start_index, end_index, status, created_at) \
             VALUES (?, 3, 0, 100, 'pending', ?)",
        )
        .bind(target_low)
        .bind(now_unix())
        .execute(&pool)
        .await
        .unwrap();

        let config = test_config(10);
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("work available");
        assert_eq!(claim.target_id, target_high, "the higher-priority target's own fresh space must be preferred");

        let pending_status: String = sqlx::query_scalar("SELECT status FROM ranges WHERE target_id = ? AND start_index = 0").bind(target_low).fetch_one(&pool).await.unwrap();
        assert_eq!(pending_status, "pending", "the lower-priority target's leftover must be left untouched");
    }

    /// A canary is a small range of a virtual target with a name planted in
    /// it, copying the real target's settings; its result is recorded,
    /// found or missed, and it counts towards nothing but the volunteer's
    /// canaries.
    #[tokio::test]
    async fn canaries_are_handed_out_recorded_and_kept_out_of_everything_else() {
        let pool = test_pool().await;
        let mut user = insert_user(&pool, "tester").await;
        let (lower, upper) = full_bounds(DEFAULT, 2);
        let real = insert_target(&pool, &lower, &upper).await;
        let mut config = test_config(100);
        config.canary_probability = 1.0;

        // Not before the client's rate is measured: a guess could make it
        // take far too long.
        let claim = claim_range(&pool, &config, &user).await.unwrap().expect("real work");
        assert_eq!(claim.target_id, real);
        sqlx::query("UPDATE ranges SET status = 'pending', assigned_user_id = NULL WHERE id = ?").bind(claim.range_id).execute(&pool).await.unwrap();
        sqlx::query("UPDATE users SET ema_rate_per_sec = 2.0 WHERE id = ?").bind(user.id).execute(&pool).await.unwrap();
        user.ema_rate_per_sec = Some(2.0);

        let mut results = Vec::new();
        for report_it in [true, false] {
            let claim = claim_range(&pool, &config, &user).await.unwrap().expect("a canary");
            assert_ne!(claim.target_id, real);
            assert_eq!(claim.target_name, crate::canary::CANARY_TARGET_NAME);
            assert_eq!((claim.prefix.as_str(), claim.suffix.as_str()), ("PRE", ".SUF"));
            assert_eq!(claim.candidate_count, 10, "2/s, for canary_seconds");
            let planted: String = sqlx::query_scalar("SELECT filename FROM canaries WHERE range_id = ?")
                .bind(claim.range_id)
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(claim.hash_a_hex, format!("0x{:08X}", crate::canary::mpq_hash(&planted, 0x100)));
            assert_eq!(claim.hash_b_hex, format!("0x{:08X}", crate::canary::mpq_hash(&planted, 0x200)));
            assert!(claim.lower_bound_filename <= planted && planted <= claim.upper_bound_filename, "{planted} in {claim:?}");

            let filename = report_it.then(|| planted.clone());
            let outcome = complete_range(&pool, &config, &user, claim.range_id, report_it, filename, 1.0, 5).await.unwrap();
            assert!(!outcome.target_solved, "a canary isn't a real find");
            let (result, status): (String, String) =
                sqlx::query_as("SELECT c.result, t.status FROM canaries c JOIN targets t ON t.id = c.target_id WHERE c.range_id = ?")
                    .bind(claim.range_id)
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            results.push((result, status));
        }
        assert_eq!(results, vec![("found".into(), "solved".into()), ("missed".into(), "paused".into())]);

        let ema: Option<f64> = sqlx::query_scalar("SELECT ema_rate_per_sec FROM users WHERE id = ?").bind(user.id).fetch_one(&pool).await.unwrap();
        assert_eq!(ema, Some(2.0), "a canary doesn't measure the client's rate");
        let real_status: String = sqlx::query_scalar("SELECT status FROM targets WHERE id = ?").bind(real).fetch_one(&pool).await.unwrap();
        assert_eq!(real_status, "active");

        // One more canary, still being searched: not counted yet.
        claim_range(&pool, &config, &user).await.unwrap().expect("a canary");
        let volunteers = crate::dashboard::volunteers(&pool).await.unwrap();
        assert_eq!(volunteers.len(), 1);
        let v = &volunteers[0];
        assert_eq!((v.candidates, v.ranges_completed, v.found), (0, 0, 0), "canaries count towards nothing else");
        assert_eq!((v.canaries_found, v.canaries_total), (1, 2));

    }
}
