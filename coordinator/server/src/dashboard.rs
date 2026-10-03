//! Human-facing dashboard: a static page (served at `GET /`) that polls
//! `GET /api/v1/dashboard` for target/range/worker data. No build step, no JS
//! framework - the page is a single self-contained file embedded at compile time.

use std::collections::HashMap;

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::{Serialize, Serializer};

/// A count of candidates, as a decimal string: past 2^53 a JSON number no
/// longer reads back exactly in the dashboard's JavaScript, which would show
/// two ranges of the same size with different counts.
fn as_decimal<S: Serializer>(n: &Pos, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(n)
}

use crate::alphabet::{bound_indices_at_len, ceil_index, floor_index, index_to_candidate, join_pos, max_supported_len, space_size, Pos};
use crate::error::AppError;
use crate::models::{i64_to_u32, PriorityRange, Segment, SkipRange, Target};
use crate::ranges::{count_within, owned_spans};
use crate::state::AppState;

#[derive(Serialize)]
pub struct DashboardResponse {
    pub targets: Vec<DashboardTarget>,
    /// Everyone who has claimed work or found a name, most candidates
    /// searched first - see `volunteers`.
    pub volunteers: Vec<DashboardVolunteer>,
}

/// One username's contribution, across every hostname it has registered
/// from (each (username, hostname) pair is its own `users` row).
#[derive(Serialize)]
pub struct DashboardVolunteer {
    pub username: String,
    /// How many hostnames this username has registered from.
    pub hostnames: i64,
    /// Candidates in the completed ranges credited to it
    /// (`last_assigned_user_id`). A little generous: a range closed because
    /// its target was solved elsewhere, or because a match was found in it,
    /// counts in full even though the search stopped partway.
    #[serde(serialize_with = "as_decimal")]
    pub candidates: Pos,
    pub ranges_completed: i64,
    /// Distinct names found: targets sharing a Hash A/Hash B pair are the
    /// same file, solved together by one find, so they count once.
    pub found: i64,
    /// Canaries (see `canary.rs`) found, of those handed to it that are
    /// over: completed, or released unfinished (the client quit or stalled).
    /// One still being searched isn't counted yet. None of the above counts
    /// a canary.
    pub canaries_found: i64,
    pub canaries_total: i64,
}

/// Every username that has claimed work or found a name, sorted by
/// candidates searched, then names found, then username.
pub(crate) async fn volunteers(pool: &sqlx::SqlitePool) -> Result<Vec<DashboardVolunteer>, AppError> {
    #[derive(sqlx::FromRow)]
    struct CompletedRow {
        username: String,
        candidate_len: i64,
        start_block: i64,
        start_index: i64,
        end_block: i64,
        end_index: i64,
        alphabet: String,
    }
    let completed: Vec<CompletedRow> = sqlx::query_as(
        "SELECT u.username, r.candidate_len, r.start_block, r.start_index, r.end_block, r.end_index, r.alphabet \
         FROM ranges r JOIN users u ON u.id = r.last_assigned_user_id JOIN targets t ON t.id = r.target_id \
         WHERE r.status = 'completed' AND t.is_virtual = 0",
    )
    .fetch_all(pool)
    .await?;
    let contributors: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT u.username FROM ranges r JOIN users u ON u.id = r.last_assigned_user_id \
         JOIN targets t ON t.id = r.target_id WHERE t.is_virtual = 0 \
         UNION SELECT u.username FROM targets t JOIN users u ON u.id = t.found_by_user_id WHERE t.is_virtual = 0 \
         UNION SELECT u.username FROM canaries c JOIN users u ON u.id = c.user_id",
    )
    .fetch_all(pool)
    .await?;
    let hostnames: HashMap<String, i64> = sqlx::query_as::<_, (String, i64)>("SELECT username, COUNT(*) FROM users GROUP BY username")
        .fetch_all(pool)
        .await?
        .into_iter()
        .collect();
    let found: HashMap<String, i64> = sqlx::query_as::<_, (String, i64)>(
        "SELECT u.username, COUNT(DISTINCT t.hash_a || ':' || t.hash_b) FROM targets t JOIN users u ON u.id = t.found_by_user_id \
         WHERE t.found_filename IS NOT NULL AND t.is_virtual = 0 GROUP BY u.username",
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .collect();
    let canaries: HashMap<String, (i64, i64)> = sqlx::query_as::<_, (String, i64, i64)>(
        "SELECT u.username, SUM(c.result = 'found'), COUNT(*) FROM canaries c JOIN users u ON u.id = c.user_id \
         JOIN ranges r ON r.id = c.range_id WHERE c.result != 'pending' OR r.status != 'in_progress' GROUP BY u.username",
    )
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|(username, found, total)| (username, (found, total)))
    .collect();

    let mut by_username: HashMap<String, DashboardVolunteer> = contributors
        .into_iter()
        .map(|(username,)| {
            let volunteer = DashboardVolunteer {
                hostnames: hostnames.get(&username).copied().unwrap_or(0),
                found: found.get(&username).copied().unwrap_or(0),
                canaries_found: canaries.get(&username).map_or(0, |c| c.0),
                canaries_total: canaries.get(&username).map_or(0, |c| c.1),
                candidates: 0,
                ranges_completed: 0,
                username: username.clone(),
            };
            (username, volunteer)
        })
        .collect();
    for row in completed {
        let Some(volunteer) = by_username.get_mut(&row.username) else { continue };
        let start = join_pos(&row.alphabet, row.candidate_len, row.start_block, row.start_index);
        let end = join_pos(&row.alphabet, row.candidate_len, row.end_block, row.end_index);
        volunteer.candidates += end - start;
        volunteer.ranges_completed += 1;
    }

    let mut volunteers: Vec<DashboardVolunteer> = by_username.into_values().collect();
    volunteers.sort_by(|a, b| b.candidates.cmp(&a.candidates).then(b.found.cmp(&a.found)).then_with(|| a.username.cmp(&b.username)));
    Ok(volunteers)
}

#[derive(Serialize)]
pub struct DashboardTarget {
    pub id: i64,
    pub name: String,
    pub status: String,
    /// The bounds the target was created with, exactly as given - the primary way a
    /// target's scope is defined now that there's no separate min_len/max_len. Not
    /// wrapped in prefix/suffix: a bound doesn't have to relate to either (see
    /// `admin_create_target`), so doing that would often produce a nonsensical
    /// filename (e.g. a double extension) rather than a real one.
    pub lower_bound: String,
    pub upper_bound: String,
    /// Shown just below the target's name - see `models::Target::hash_a`/`hash_b`.
    pub hash_a_hex: String,
    pub hash_b_hex: String,
    /// Shown below the hashes, with the settings below.
    pub prefix: String,
    pub suffix: String,
    /// The pruning rules and insertions ranges claimed from now on are
    /// searched with - see `AdminCreateTargetRequest`'s fields of the same
    /// names. Ranges already in progress may have been claimed with others.
    pub prune_symbol_runs: bool,
    pub prune_unopened_brackets: bool,
    pub prune_whole_candidate: bool,
    pub max_backslash_count: i64,
    pub min_backslash_count: i64,
    pub prune_adjacent_backslashes: bool,
    pub insert_from_start: Option<(String, i64)>,
    pub insert_from_end: Option<(String, i64)>,
    pub found_filename: Option<String>,
    pub found_by: Option<String>,
    /// The backend `found_by` searches with ("cuda", "cpu", ...) - see
    /// `DashboardRange::worker_backend`.
    pub found_by_backend: Option<String>,
    /// Unix seconds - `None` for a target solved before this was recorded.
    pub found_at: Option<i64>,
    /// Higher claims first - see ranges::claim_range. Defaults to 0.
    pub priority: i64,
    /// Operator note shown in the target's card header. Rendered as raw
    /// HTML by the dashboard, not escaped - see admin_create_target.
    pub description: Option<String>,
    /// The target's *current* alphabet - i.e. what future ranges will be
    /// carved with (see `handlers::admin_patch_target`). Individual ranges
    /// may have been carved under a different (older) one - see
    /// `DashboardRange::alphabet_name`.
    pub alphabet_name: String,
    /// The resolved characters behind `alphabet_name`, for the dashboard's
    /// hover tooltip.
    pub alphabet: String,
    /// The shortest candidate length the main sweep carves - see
    /// `AdminCreateTargetRequest::start_len`.
    pub start_len: i64,
    /// Where the target's own main sweep will carve next - its stored
    /// position, moved past any priority range span it would jump over on
    /// its next carve (see `effective_sweep_position`). Compared against
    /// each `DashboardPriorityRange` so the dashboard can show the (never
    /// persisted - see `ranges::priority_spans_at`) gap between them.
    pub cursor_candidate_len: i64,
    pub cursor_candidate: String,
    /// Where the main sweep's position falls among `ranges`: right after
    /// this range, or before all of them if `None`.
    pub sweep_after_range_id: Option<i64>,
    pub priority_ranges: Vec<DashboardPriorityRange>,
    /// Including removed ones, so the `skipped` ranges they already produced
    /// can still show their reason.
    pub skip_ranges: Vec<DashboardSkipRange>,
    /// Ordered by candidate length, then first candidate - the order the
    /// dashboard lists them in, and the order `gaps` refers to. Not by
    /// position, which isn't comparable between ranges in different
    /// alphabets; candidates are, since every alphabet lists its characters
    /// in ascending order (see `alphabet::floor_index`).
    pub ranges: Vec<DashboardRange>,
    /// Candidates not covered by any range, between two consecutive
    /// `ranges` - see `gaps_between`.
    pub gaps: Vec<DashboardGap>,
}

/// A stretch of candidates between two consecutive ranges (in `ranges`'
/// order) that no range covers. May run across candidate lengths.
#[derive(Serialize)]
pub struct DashboardGap {
    /// The range this gap is listed right after.
    pub after_range_id: i64,
    pub first_len: i64,
    pub first_candidate: String,
    pub last_len: i64,
    pub last_candidate: String,
    /// How many candidates it covers.
    #[serde(serialize_with = "as_decimal")]
    pub count: Pos,
}

/// The gaps between the ranges in `ranges` (already ordered by length,
/// then first candidate): before each range, whatever lies between it and
/// the furthest any earlier range reaches. Within a length, that's simply
/// the space between the two; across lengths, it's the rest of the earlier
/// length, any whole lengths in between, and the start of the later length,
/// each limited to what the target's bounds cover at that length.
///
/// Ranges in different alphabets are compared in the smaller of the two -
/// a range carved in a bigger alphabet covers every candidate of a smaller
/// one within it - so e.g. `skipped` rows an alphabet change wrote in one
/// alphabet don't show gaps where ranges in another already searched. Two
/// alphabets where neither has all of the other's characters aren't
/// comparable (no gap is reported there), nor is a gap reported in a bigger
/// alphabet's candidates between ranges in a smaller one.
fn gaps_between(ranges: &[DashboardRange], lower_bound: &str, upper_bound: &str) -> Vec<DashboardGap> {
    let mut gaps = Vec::new();
    let Some(mut reach) = ranges.first() else {
        return gaps;
    };
    for pair in ranges.windows(2) {
        let (previous, b) = (&pair[0], &pair[1]);
        if let Some(alphabet) = common_alphabet(&reach.alphabet, &b.alphabet) {
            gaps.extend(gap_between(reach, b, previous.id, alphabet, lower_bound, upper_bound));
        }
        if (b.candidate_len, &b.last_candidate) > (reach.candidate_len, &reach.last_candidate) {
            reach = b;
        }
    }
    gaps
}

/// Whichever of two alphabets has all of the other's characters in it -
/// the smaller one - or `None` if neither does.
fn common_alphabet<'a>(x: &'a str, y: &'a str) -> Option<&'a str> {
    let within = |small: &str, big: &str| small.chars().all(|c| big.contains(c));
    if within(x, y) {
        Some(x)
    } else if within(y, x) {
        Some(y)
    } else {
        None
    }
}

/// The candidates of `alphabet` after range `a`'s last and before range
/// `b`'s first (see `gaps_between`), listed after range `after_range_id`.
fn gap_between(a: &DashboardRange, b: &DashboardRange, after_range_id: i64, alphabet: &str, lower_bound: &str, upper_bound: &str) -> Option<DashboardGap> {
    let a_end = floor_index(alphabet, &a.last_candidate).map_or(0, |last| last + 1);
    let b_start = ceil_index(alphabet, &b.first_candidate).unwrap_or_else(|| space_size(alphabet, b.candidate_len));
    // (length, first index, end index exclusive), one per length.
    let mut pieces: Vec<(i64, Pos, Pos)> = Vec::new();
    if a.candidate_len == b.candidate_len {
        pieces.push((a.candidate_len, a_end, b_start));
    } else {
        for len in a.candidate_len..=b.candidate_len {
            let (lo, hi) = bound_indices_at_len(alphabet, lower_bound, upper_bound, len);
            let from = if len == a.candidate_len { a_end } else { lo };
            let to = if len == b.candidate_len { b_start } else { hi + 1 };
            pieces.push((len, from, to));
        }
    }
    pieces.retain(|&(_, from, to)| from < to);
    let (&(first_len, first, _), &(last_len, _, last_end)) = (pieces.first()?, pieces.last()?);
    Some(DashboardGap {
        after_range_id,
        first_len,
        first_candidate: index_to_candidate(alphabet, first, first_len),
        last_len,
        last_candidate: index_to_candidate(alphabet, last_end - 1, last_len),
        count: pieces.iter().map(|&(_, from, to)| to - from).sum(),
    })
}

/// Where the main sweep will carve next. Its stored position only moves
/// when it carves, and it jumps over a priority or skip range's span only
/// then (see `ranges::claim_range`) - so until it next carves, the stored
/// position can sit at the start of work a priority range has already
/// handed out, or of a skip range's `skipped` row. This moves past any such
/// span - given in `jumps` as (length, alphabet, start, end) - continuing at
/// the next length (within the target's bounds) when one runs to a length's
/// end, just as the sweep itself will.
fn effective_sweep_position(
    mut len: i64,
    mut index: Pos,
    alphabet: &str,
    jumps: &[(i64, &str, Pos, Pos)],
    lower_bound: &str,
    upper_bound: &str,
) -> (i64, Pos) {
    let max_len = max_supported_len(alphabet);
    loop {
        let covering = jumps.iter().find(|&&(jump_len, jump_alphabet, start, end)| {
            jump_alphabet == alphabet && jump_len == len && start <= index && index < end
        });
        if let Some(&(_, _, _, end)) = covering {
            index = end;
            continue;
        }
        let (_, upper) = bound_indices_at_len(alphabet, lower_bound, upper_bound, len);
        if index <= upper || len >= max_len {
            return (len, index);
        }
        len += 1;
        index = bound_indices_at_len(alphabet, lower_bound, upper_bound, len).0;
    }
}

/// The last range in `ranges` (ordered by length, then first candidate)
/// that starts before the main sweep's position - so the sweep marker goes
/// right after it. `None` if the sweep is before every range.
fn sweep_after_range_id(ranges: &[DashboardRange], cursor_len: i64, cursor_candidate: &str) -> Option<i64> {
    ranges
        .iter()
        .take_while(|r| (r.candidate_len, r.first_candidate.as_str()) < (cursor_len, cursor_candidate))
        .last()
        .map(|r| r.id)
}

#[derive(Serialize)]
pub struct DashboardPriorityRange {
    pub id: i64,
    pub priority: i64,
    pub pattern: String,
    pub candidate_len: i64,
    pub first_candidate: String,
    /// `None` once this priority range is exhausted (its next position is
    /// its end) - see `ranges::claim_priority_range_chunk`.
    pub next_candidate: Option<String>,
    pub alphabet: String,
    /// How many candidates it owns, and how many of those it has handed out.
    /// Not simply its end minus its start, since there can be gaps between
    /// its segments (see `ranges::owned_spans`).
    #[serde(serialize_with = "as_decimal")]
    pub candidate_count: Pos,
    #[serde(serialize_with = "as_decimal")]
    pub handed_out_count: Pos,
    /// How many separate stretches of candidates it owns.
    pub segment_count: usize,
    /// What it owns - only used here, by `effective_sweep_position`.
    #[serde(skip)]
    pub spans: Vec<(Pos, Pos)>,
}

#[derive(Serialize)]
pub struct DashboardSkipRange {
    pub id: i64,
    pub pattern: String,
    pub reason: String,
    /// Removed by an operator after it had already skipped something - see
    /// `ranges::remove_skip_range`.
    pub removed: bool,
}

/// One row of the dashboard's range query - positions still split into
/// block and index, as stored (see `alphabet::split_pos`).
#[derive(sqlx::FromRow)]
struct RangeRow {
    target_id: i64,
    id: i64,
    status: String,
    candidate_len: i64,
    start_block: i64,
    start_index: i64,
    end_block: i64,
    end_index: i64,
    progress_block: i64,
    progress_index: Option<i64>,
    worker: Option<String>,
    worker_backend: Option<String>,
    assigned_at: Option<i64>,
    lease_expires_at: Option<i64>,
    completed_at: Option<i64>,
    alphabet_name: String,
    alphabet: String,
    priority_range_id: Option<i64>,
    skip_range_id: Option<i64>,
}

#[derive(Serialize)]
pub struct DashboardRange {
    pub id: i64,
    pub status: String,
    pub candidate_len: i64,
    /// Its end minus its start, worked out here rather than by the
    /// dashboard - see `as_decimal`.
    #[serde(serialize_with = "as_decimal")]
    pub candidate_count: Pos,
    /// The actual first/last candidate strings this range covers - its end
    /// is exclusive (see `range_bound_filenames`), so the last candidate is
    /// decoded from the position before it.
    pub first_candidate: String,
    pub last_candidate: String,
    /// The candidate at the last heartbeat-reported progress index, if any -
    /// how far into the range its current (or last) worker has actually
    /// searched, as opposed to `first_candidate`/`last_candidate` which just
    /// describe the range's bounds.
    pub progress_candidate: Option<String>,
    /// How far into the range (0-100) `progress_candidate` represents -
    /// `(progress_index - start_index + 1) / (end_index - start_index)`, i.e.
    /// what fraction of this range's candidates are behind the checkpoint.
    pub progress_percent: Option<f64>,
    /// "username@hostname" of whoever last claimed this range, even if it was
    /// since reclaimed - see the migration adding `last_assigned_user_id`.
    pub worker: Option<String>,
    /// The backend `worker` searches with ("cuda", "cpu", ...), as declared
    /// when it last registered - `None` for users that registered before
    /// clients sent it. Shown as a badge next to the worker.
    pub worker_backend: Option<String>,
    pub assigned_at: Option<i64>,
    pub lease_expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    /// The alphabet this specific range was carved with - see
    /// `models::Range::alphabet`. Not necessarily the target's current
    /// alphabet (`DashboardTarget::alphabet_name`) if the target was patched
    /// since this range was carved.
    pub alphabet_name: String,
    /// The resolved characters behind `alphabet_name`, for the dashboard's
    /// hover tooltip.
    pub alphabet: String,
    /// Which `DashboardPriorityRange` (if any) this range was carved from -
    /// see `models::Range::priority_range_id`. Display only.
    pub priority_range_id: Option<i64>,
    /// For a `skipped` range, which `DashboardSkipRange` it came from - `None`
    /// for one an alphabet change left behind. Display only.
    pub skip_range_id: Option<i64>,
}

// Both queries below build a user's "username@hostname" in SQL (`||` gives
// NULL if either part is NULL, i.e. no such user) rather than selecting the
// two separately, since sqlx only maps rows of up to 16 columns to a tuple.
// An empty backend (users registered before clients sent one) reads as NULL.

pub async fn dashboard_data(State(state): State<AppState>) -> Result<Json<DashboardResponse>, AppError> {
    #[allow(clippy::type_complexity)]
    let target_rows: Vec<(i64, String, String, String, String, i64, i64, Option<String>, Option<String>, Option<String>, Option<i64>, String, String, i64, Option<String>)> = sqlx::query_as(
        "SELECT targets.id, targets.name, targets.status, \
                targets.lower_bound, targets.upper_bound, targets.hash_a, targets.hash_b, targets.found_filename, \
                found_user.username || '@' || found_user.hostname, NULLIF(found_user.backend, ''), targets.found_at, targets.alphabet_name, targets.alphabet, targets.priority, \
                targets.description \
         FROM targets LEFT JOIN users AS found_user ON found_user.id = targets.found_by_user_id \
         WHERE targets.is_virtual = 0 \
         ORDER BY targets.priority DESC, targets.created_at ASC",
    )
    .fetch_all(&state.pool)
    .await?;

    // The rest of each target's settings, which don't fit in target_rows
    // above - it's already at sqlx's 16-column tuple limit.
    let mut settings_by_target: HashMap<i64, Target> = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE is_virtual = 0")
        .fetch_all(&state.pool)
        .await?
        .into_iter()
        .map(|target| (target.id, target))
        .collect();

    // The dashboard renders every target unconditionally, so there's no
    // filtering benefit to a per-target `WHERE target_id = ?` on the child
    // tables below - fetch each one whole (1 query apiece, independent of
    // how many targets exist) and group by target_id in memory instead.
    // Each table's own ORDER BY is preserved per-group because HashMap's
    // `entry().or_default().push()` keeps insertion order within a bucket.
    let all_ranges: Vec<RangeRow> = sqlx::query_as(
        "SELECT ranges.target_id, ranges.id, ranges.status, ranges.candidate_len, ranges.start_block, ranges.start_index, \
                ranges.end_block, ranges.end_index, ranges.progress_block, ranges.progress_index, \
                worker.username || '@' || worker.hostname AS worker, NULLIF(worker.backend, '') AS worker_backend, \
                ranges.assigned_at, ranges.lease_expires_at, ranges.completed_at, \
                ranges.alphabet_name, ranges.alphabet, ranges.priority_range_id, ranges.skip_range_id \
         FROM ranges LEFT JOIN users AS worker ON worker.id = ranges.last_assigned_user_id \
         WHERE ranges.target_id IN (SELECT id FROM targets WHERE is_virtual = 0) \
         ORDER BY ranges.candidate_len, ranges.start_block, ranges.start_index",
    )
    .fetch_all(&state.pool)
    .await?;
    let mut ranges_by_target: HashMap<i64, Vec<RangeRow>> = HashMap::new();
    for row in all_ranges {
        ranges_by_target.entry(row.target_id).or_default().push(row);
    }

    // start_len rides along here rather than in target_rows above, which is
    // already at sqlx's 16-column tuple limit.
    let all_progress: Vec<(i64, i64, i64, i64, String, i64)> = sqlx::query_as(
        "SELECT target_progress.target_id, target_progress.candidate_len, target_progress.next_block, target_progress.next_index, \
                target_progress.alphabet, targets.start_len \
         FROM target_progress JOIN targets ON targets.id = target_progress.target_id",
    )
    .fetch_all(&state.pool)
    .await?;
    let progress_by_target: HashMap<i64, (i64, Pos, String, i64)> = all_progress
        .into_iter()
        .map(|(target_id, candidate_len, next_block, next_index, alphabet, start_len)| {
            let next = join_pos(&alphabet, candidate_len, next_block, next_index);
            (target_id, (candidate_len, next, alphabet, start_len))
        })
        .collect();

    let all_priority_ranges: Vec<PriorityRange> =
        sqlx::query_as("SELECT * FROM priority_ranges ORDER BY priority DESC, created_at ASC").fetch_all(&state.pool).await?;
    let mut priority_ranges_by_target: HashMap<i64, Vec<PriorityRange>> = HashMap::new();
    for pr in all_priority_ranges {
        priority_ranges_by_target.entry(pr.target_id).or_default().push(pr);
    }
    let priority_segments = segments_by_owner(
        &state.pool,
        "SELECT priority_range_id AS owner_id, start_block, start_index, end_block, end_index FROM priority_range_segments",
    )
    .await?;

    let all_skip_ranges: Vec<SkipRange> = sqlx::query_as("SELECT * FROM skip_ranges ORDER BY candidate_len, created_at").fetch_all(&state.pool).await?;
    let skip_segments =
        segments_by_owner(&state.pool, "SELECT skip_range_id AS owner_id, start_block, start_index, end_block, end_index FROM skip_range_segments").await?;
    let mut skip_ranges_by_target: HashMap<i64, Vec<DashboardSkipRange>> = HashMap::new();
    // (length, alphabet, start, end) of every skip range segment - see effective_sweep_position.
    let mut skip_spans_by_target: HashMap<i64, Vec<(i64, String, Pos, Pos)>> = HashMap::new();
    for sr in all_skip_ranges {
        let spans: Vec<(Pos, Pos)> = skip_segments.get(&sr.id).into_iter().flatten().map(|s| s.span(&sr.alphabet, sr.candidate_len)).collect();
        skip_spans_by_target
            .entry(sr.target_id)
            .or_default()
            .extend(spans.into_iter().map(|(start, end)| (sr.candidate_len, sr.alphabet.clone(), start, end)));
        skip_ranges_by_target.entry(sr.target_id).or_default().push(DashboardSkipRange {
            id: sr.id,
            pattern: sr.pattern,
            reason: sr.reason,
            removed: sr.removed_at.is_some(),
        });
    }

    let mut targets = Vec::with_capacity(target_rows.len());
    for (id, name, status, lower_bound, upper_bound, hash_a, hash_b, found_filename, found_by, found_by_backend, found_at, alphabet_name, alphabet, priority, description) in target_rows {
        // Deleted since target_rows was read - leave it out, as the next refresh will.
        let Some(settings) = settings_by_target.remove(&id) else { continue };
        let range_rows = ranges_by_target.remove(&id).unwrap_or_default();

        let mut ranges: Vec<DashboardRange> = range_rows
            .into_iter()
            .map(|row| {
                // Each range is decoded with its OWN alphabet, not the
                // target's current one - a range carved before the
                // target's alphabet was last patched (see
                // handlers::admin_patch_target) must still be shown with
                // the alphabet it actually holds candidates in.
                let len = row.candidate_len;
                let start_index = join_pos(&row.alphabet, len, row.start_block, row.start_index);
                let end_index = join_pos(&row.alphabet, len, row.end_block, row.end_index);
                let progress_index = row.progress_index.map(|p| join_pos(&row.alphabet, len, row.progress_block, p));
                DashboardRange {
                    id: row.id,
                    status: row.status,
                    candidate_len: len,
                    candidate_count: end_index - start_index,
                    first_candidate: index_to_candidate(&row.alphabet, start_index, len),
                    last_candidate: index_to_candidate(&row.alphabet, end_index - 1, len),
                    progress_candidate: progress_index.map(|p| index_to_candidate(&row.alphabet, p, len)),
                    progress_percent: progress_index.map(|p| (p - start_index + 1) as f64 / (end_index - start_index) as f64 * 100.0),
                    worker: row.worker,
                    worker_backend: row.worker_backend,
                    assigned_at: row.assigned_at,
                    lease_expires_at: row.lease_expires_at,
                    completed_at: row.completed_at,
                    alphabet_name: row.alphabet_name,
                    alphabet: row.alphabet,
                    priority_range_id: row.priority_range_id,
                    skip_range_id: row.skip_range_id,
                }
            })
            .collect();
        // Stable, so ranges starting at the same candidate stay in position order.
        ranges.sort_by(|a, b| (a.candidate_len, &a.first_candidate).cmp(&(b.candidate_len, &b.first_candidate)));

        let (mut stored_cursor_len, mut stored_cursor_next_index, mut cursor_alphabet, start_len) = progress_by_target
            .get(&id)
            .cloned()
            .ok_or_else(|| AppError::Internal(format!("missing target_progress row for target {id}")))?;
        // Mirrors ranges::claim_range: a cursor below start_len (raised by a
        // patch since carving last ran) jumps to the start of start_len, in
        // the target's current alphabet, on its next carve.
        if stored_cursor_len < start_len {
            stored_cursor_len = start_len;
            stored_cursor_next_index = bound_indices_at_len(&alphabet, &lower_bound, &upper_bound, start_len).0;
            cursor_alphabet = alphabet.clone();
        }
        let gaps = gaps_between(&ranges, &lower_bound, &upper_bound);

        let priority_range_rows = priority_ranges_by_target.remove(&id).unwrap_or_default();
        let priority_ranges = priority_range_rows
            .into_iter()
            .map(|pr| {
                let (start_index, end_index, next_index) = (pr.start(), pr.end(), pr.next());
                let spans = owned_spans(&pr, priority_segments.get(&pr.id));
                DashboardPriorityRange {
                    candidate_count: count_within(&spans, start_index, end_index),
                    handed_out_count: count_within(&spans, start_index, next_index),
                    segment_count: spans.len(),
                    spans,
                    id: pr.id,
                    priority: pr.priority,
                    candidate_len: pr.candidate_len,
                    first_candidate: index_to_candidate(&pr.alphabet, start_index, pr.candidate_len),
                    next_candidate: (next_index < end_index).then(|| index_to_candidate(&pr.alphabet, next_index, pr.candidate_len)),
                    pattern: pr.pattern,
                    alphabet: pr.alphabet,
                }
            })
            .collect::<Vec<_>>();

        let skip_spans = skip_spans_by_target.remove(&id).unwrap_or_default();
        let jumps: Vec<(i64, &str, Pos, Pos)> = priority_ranges
            .iter()
            .flat_map(|pr| pr.spans.iter().map(move |&(start, end)| (pr.candidate_len, pr.alphabet.as_str(), start, end)))
            .chain(skip_spans.iter().map(|(len, alphabet, start, end)| (*len, alphabet.as_str(), *start, *end)))
            .collect();
        let (cursor_candidate_len, cursor_next_index) =
            effective_sweep_position(stored_cursor_len, stored_cursor_next_index, &cursor_alphabet, &jumps, &lower_bound, &upper_bound);
        let cursor_candidate = index_to_candidate(&cursor_alphabet, cursor_next_index, cursor_candidate_len);
        let sweep_after_range_id = sweep_after_range_id(&ranges, cursor_candidate_len, &cursor_candidate);

        targets.push(DashboardTarget {
            id,
            name,
            status,
            lower_bound,
            upper_bound,
            hash_a_hex: format!("0x{:08X}", i64_to_u32(hash_a)),
            hash_b_hex: format!("0x{:08X}", i64_to_u32(hash_b)),
            prefix: settings.prefix,
            suffix: settings.suffix,
            prune_symbol_runs: settings.prune_symbol_runs != 0,
            prune_unopened_brackets: settings.prune_unopened_brackets != 0,
            prune_whole_candidate: settings.prune_whole_candidate != 0,
            max_backslash_count: settings.max_backslash_count,
            min_backslash_count: settings.min_backslash_count,
            prune_adjacent_backslashes: settings.prune_adjacent_backslashes != 0,
            insert_from_start: settings.insert_from_start_text.map(|text| (text, settings.insert_from_start_position)),
            insert_from_end: settings.insert_from_end_text.map(|text| (text, settings.insert_from_end_position)),
            found_filename,
            found_by,
            found_by_backend,
            found_at,
            priority,
            description,
            alphabet_name,
            alphabet,
            start_len,
            cursor_candidate_len,
            cursor_candidate,
            sweep_after_range_id,
            priority_ranges,
            skip_ranges: skip_ranges_by_target.remove(&id).unwrap_or_default(),
            ranges,
            gaps,
        });
    }

    Ok(Json(DashboardResponse { targets, volunteers: volunteers(&state.pool).await? }))
}

/// Every row `sql` selects (as `models::Segment`s), grouped by owner.
async fn segments_by_owner(pool: &sqlx::SqlitePool, sql: &'static str) -> Result<HashMap<i64, Vec<Segment>>, AppError> {
    let mut by_owner: HashMap<i64, Vec<Segment>> = HashMap::new();
    for segment in sqlx::query_as::<_, Segment>(sql).fetch_all(pool).await? {
        by_owner.entry(segment.owner_id).or_default().push(segment);
    }
    Ok(by_owner)
}

pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    fn range(id: i64, candidate_len: i64, start_index: Pos, end_index: Pos) -> DashboardRange {
        range_in("letters", LETTERS, id, candidate_len, start_index, end_index)
    }

    fn range_in(alphabet_name: &str, alphabet: &str, id: i64, candidate_len: i64, start_index: Pos, end_index: Pos) -> DashboardRange {
        DashboardRange {
            id,
            status: "completed".into(),
            candidate_len,
            candidate_count: end_index - start_index,
            first_candidate: index_to_candidate(alphabet, start_index, candidate_len),
            last_candidate: index_to_candidate(alphabet, end_index - 1, candidate_len),
            progress_candidate: None,
            progress_percent: None,
            worker: None,
            worker_backend: None,
            assigned_at: None,
            lease_expires_at: None,
            completed_at: None,
            alphabet_name: alphabet_name.into(),
            alphabet: alphabet.into(),
            priority_range_id: None,
            skip_range_id: None,
        }
    }

    fn summary(g: &DashboardGap) -> (i64, i64, &str, i64, &str, Pos) {
        (g.after_range_id, g.first_len, g.first_candidate.as_str(), g.last_len, g.last_candidate.as_str(), g.count)
    }

    #[test]
    fn gaps_between_finds_gaps_within_a_length_but_not_between_adjacent_ranges() {
        // Length 2 (26*26 = 676): 0..10 and 10..20 back to back, then 30..40.
        let ranges = vec![range(1, 2, 0, 10), range(2, 2, 10, 20), range(3, 2, 30, 40)];
        let gaps = gaps_between(&ranges, "AA", "ZZ");
        assert_eq!(gaps.len(), 1);
        assert_eq!(summary(&gaps[0]), (2, 2, "AU", 2, "BD", 10));
    }

    #[test]
    fn gaps_between_runs_across_lengths_within_the_targets_bounds() {
        // Length 1 covered up to "X" (0..24), then length 3 starting at "AAC" (index 2):
        // the gap is Y-Z, all of length 2, then AAA-AAB.
        let ranges = vec![range(1, 1, 0, 24), range(2, 3, 2, 10)];
        let gaps = gaps_between(&ranges, "A", "Z");
        assert_eq!(summary(&gaps[0]), (1, 1, "Y", 3, "AAB", 2 + 676 + 2));

        // Back to back across a length boundary: no gap.
        let ranges = vec![range(1, 1, 0, 26), range(2, 2, 0, 5)];
        assert!(gaps_between(&ranges, "A", "Z").is_empty());
    }

    #[test]
    fn gaps_between_measures_from_the_furthest_reaching_range() {
        // 0..30 reaches past 10..20, which it overlaps: nothing is missing before 30..40.
        let ranges = vec![range(1, 2, 0, 30), range(2, 2, 10, 20), range(3, 2, 30, 40)];
        assert!(gaps_between(&ranges, "AA", "ZZ").is_empty());
    }

    /// Ranges in a smaller alphabet ("ACE") inside ones in a bigger one
    /// (LETTERS) - like the `skipped` rows an alphabet change used to write -
    /// show no gaps where the bigger one's ranges searched, but do where
    /// nothing did, counted in the smaller alphabet.
    #[test]
    fn gaps_between_compares_ranges_in_different_alphabets_in_the_smaller_one() {
        const ACE: &str = "ACE";
        let at = |alphabet: &str, candidate: &str| crate::alphabet::candidate_to_index(alphabet, candidate).unwrap();
        let mut ranges = vec![
            range_in("ace", ACE, 1, 2, at(ACE, "AA"), at(ACE, "AE") + 1),
            range_in("ace", ACE, 2, 2, at(ACE, "CA"), at(ACE, "CE") + 1),
            range_in("ace", ACE, 3, 2, at(ACE, "EA"), at(ACE, "EE") + 1),
            range(4, 2, at(LETTERS, "AA"), at(LETTERS, "BZ") + 1),
            range(5, 2, at(LETTERS, "CA"), at(LETTERS, "CZ") + 1),
        ];
        ranges.sort_by(|a, b| (a.candidate_len, &a.first_candidate).cmp(&(b.candidate_len, &b.first_candidate)));
        assert_eq!(ranges.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1, 4, 2, 5, 3]);
        assert!(gaps_between(&ranges, "AA", "ZZ").is_empty(), "letters A-C searched, then E in ACE");

        // Without letters' CA-CZ, ACE's CA-CE still covers what ACE has between them.
        ranges.retain(|r| r.id != 5);
        assert!(gaps_between(&ranges, "AA", "ZZ").is_empty());

        // Without ACE's CA-CE too, ACE's CA-CE is missing - but not letters'
        // own D-candidates, which no range in letters is after.
        ranges.retain(|r| r.id != 2);
        let gaps = gaps_between(&ranges, "AA", "ZZ");
        assert_eq!(gaps.iter().map(summary).collect::<Vec<_>>(), vec![(4, 2, "CA", 2, "CE", 3)]);
    }

    /// A priority or skip range span at `candidate_len`, in LETTERS.
    fn jump(candidate_len: i64, start: Pos, end: Pos) -> (i64, &'static str, Pos, Pos) {
        (candidate_len, LETTERS, start, end)
    }

    #[test]
    fn effective_sweep_position_moves_past_spans_at_the_stored_position() {
        // A priority range created while the sweep was partway into its span
        // (MA-MZ = 312..338, sweep at 320): the sweep will jump to its end.
        let jumps = vec![jump(2, 312, 338)];
        assert_eq!(effective_sweep_position(2, 320, LETTERS, &jumps, "AA", "ZZ"), (2, 338));
        // Back-to-back spans (say, a priority range, then a skip range) are all jumped.
        let jumps = vec![jump(2, 312, 338), jump(2, 338, 364)];
        assert_eq!(effective_sweep_position(2, 312, LETTERS, &jumps, "AA", "ZZ"), (2, 364));
        // Nothing at the stored position: it stays.
        assert_eq!(effective_sweep_position(2, 300, LETTERS, &jumps, "AA", "ZZ"), (2, 300));
        // A span in another alphabet isn't comparable.
        assert_eq!(effective_sweep_position(2, 320, LETTERS, &[(2, "OTHER", 312, 338)], "AA", "ZZ"), (2, 320));
    }

    #[test]
    fn effective_sweep_position_continues_at_the_next_length() {
        // A span running to the end of length 2 (ZA-ZZ = 650..676) moves the
        // sweep to the start of length 3.
        let jumps = vec![jump(2, 650, 676)];
        assert_eq!(effective_sweep_position(2, 650, LETTERS, &jumps, "AA", "ZZZ"), (3, 0));
    }

    /// A username's hostnames are counted together; only completed ranges
    /// count; and two targets solved by one find (same Hash A/Hash B) count
    /// as one name found.
    #[tokio::test]
    async fn volunteers_are_tallied_per_username_and_ranked_by_candidates() {
        let pool = crate::db::connect("sqlite::memory:").await.unwrap();
        let mut users = HashMap::new();
        for (username, hostname) in [("alice", "a"), ("alice", "b"), ("bob", "c"), ("carol", "d")] {
            let id: i64 = sqlx::query_scalar(
                "INSERT INTO users (username, hostname, token, created_at, last_seen_at, protocol_version, backend) VALUES (?, ?, ?, 0, 0, '1.0.0', 'cuda') RETURNING id",
            )
            .bind(username)
            .bind(hostname)
            .bind(format!("{username}-{hostname}"))
            .fetch_one(&pool)
            .await
            .unwrap();
            users.insert(hostname, id);
        }
        let mut targets = Vec::new();
        for name in ["t1", "t2", "t3"] {
            let hash_b = if name == "t3" { 3 } else { 2 };
            let id: i64 = sqlx::query_scalar(
                "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, alphabet_name, alphabet, status, created_at) \
                 VALUES (?, '', '', 1, ?, 'A', 'Z', 0, 'letters', ?, 'active', 0) RETURNING id",
            )
            .bind(name)
            .bind(hash_b)
            .bind(LETTERS)
            .fetch_one(&pool)
            .await
            .unwrap();
            targets.push(id);
        }
        // t1 and t2 share their hashes: one find by bob solves both.
        sqlx::query("UPDATE targets SET status = 'solved', found_filename = 'X', found_by_user_id = ? WHERE id IN (?, ?)")
            .bind(users["c"])
            .bind(targets[0])
            .bind(targets[1])
            .execute(&pool)
            .await
            .unwrap();
        for (hostname, start, end, status) in [("a", 0, 100, "completed"), ("b", 100, 250, "completed"), ("a", 250, 1000, "in_progress"), ("c", 0, 50, "completed")] {
            sqlx::query(
                "INSERT INTO ranges (target_id, candidate_len, start_index, end_index, status, last_assigned_user_id, created_at, alphabet_name, alphabet) \
                 VALUES (?, 3, ?, ?, ?, ?, 0, 'letters', ?)",
            )
            .bind(targets[2])
            .bind(start)
            .bind(end)
            .bind(status)
            .bind(users[hostname])
            .bind(LETTERS)
            .execute(&pool)
            .await
            .unwrap();
        }

        let tally: Vec<(String, i64, Pos, i64, i64)> =
            volunteers(&pool).await.unwrap().into_iter().map(|v| (v.username, v.hostnames, v.candidates, v.ranges_completed, v.found)).collect();
        assert_eq!(tally, vec![("alice".into(), 2, 250, 2, 0), ("bob".into(), 1, 50, 1, 1)], "carol never claimed anything");
    }

    #[test]
    fn sweep_after_range_id_is_the_last_range_starting_before_the_sweep() {
        let ranges = vec![range(1, 2, 0, 10), range(2, 2, 10, 20), range(3, 3, 5, 9)];
        assert_eq!(sweep_after_range_id(&ranges, 2, "AU"), Some(2));
        assert_eq!(sweep_after_range_id(&ranges, 2, "AA"), None);
        assert_eq!(sweep_after_range_id(&ranges, 4, "AAAA"), Some(3));
    }
}
