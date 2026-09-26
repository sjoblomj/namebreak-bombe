//! Human-facing dashboard: a static page (served at `GET /`) that polls
//! `GET /api/v1/dashboard` for target/range/worker data. No build step, no JS
//! framework - the page is a single self-contained file embedded at compile time.

use std::collections::HashMap;

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::Serialize;

use crate::alphabet::{bound_indices_at_len, index_to_candidate, join_pos, max_supported_len, Pos};
use crate::error::AppError;
use crate::models::{i64_to_u32, PriorityRange};
use crate::state::AppState;

#[derive(Serialize)]
pub struct DashboardResponse {
    pub targets: Vec<DashboardTarget>,
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
    /// See `alphabet::compile_skip_regex` - shown on the target card so an
    /// operator can see what's configured without a separate API call.
    pub skip_regex: Option<String>,
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
    /// persisted - see `ranges::find_priority_boundary`) gap between them.
    pub cursor_candidate_len: i64,
    pub cursor_next_index: Pos,
    pub cursor_candidate: String,
    /// Where the main sweep's position falls among `ranges`: right after
    /// this range, or before all of them if `None`.
    pub sweep_after_range_id: Option<i64>,
    pub priority_ranges: Vec<DashboardPriorityRange>,
    /// Ordered by candidate length, then position - the order the dashboard
    /// lists them in, and the order `gaps` refers to.
    pub ranges: Vec<DashboardRange>,
    /// Candidates not covered by any range, between two consecutive
    /// `ranges` - see `gaps_between`.
    pub gaps: Vec<DashboardGap>,
}

/// A stretch of candidates between two consecutive ranges (in `ranges`'
/// order) that no range covers. May run across candidate lengths.
#[derive(Serialize)]
pub struct DashboardGap {
    /// The range this gap comes right after.
    pub after_range_id: i64,
    pub first_len: i64,
    pub first_candidate: String,
    pub last_len: i64,
    pub last_candidate: String,
    /// How many candidates it covers.
    pub count: Pos,
}

/// The gaps between consecutive ranges in `ranges` (already ordered by
/// length, then start). Within a length, a gap is simply the space between
/// one range's end and the next one's start; across lengths, it's the rest
/// of the earlier length, any whole lengths in between, and the start of
/// the later length, each limited to what the target's bounds cover at that
/// length. Two ranges carved under different alphabets aren't comparable
/// (an alphabet patch happened in between), so no gap is reported there.
fn gaps_between(ranges: &[DashboardRange], lower_bound: &str, upper_bound: &str) -> Vec<DashboardGap> {
    let mut gaps = Vec::new();
    for pair in ranges.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a.alphabet != b.alphabet {
            continue;
        }
        // (length, first index, end index exclusive), one per length.
        let mut pieces: Vec<(i64, Pos, Pos)> = Vec::new();
        if a.candidate_len == b.candidate_len {
            pieces.push((a.candidate_len, a.end_index, b.start_index));
        } else {
            for len in a.candidate_len..=b.candidate_len {
                let (lo, hi) = bound_indices_at_len(&a.alphabet, lower_bound, upper_bound, len);
                let from = if len == a.candidate_len { a.end_index } else { lo };
                let to = if len == b.candidate_len { b.start_index } else { hi + 1 };
                pieces.push((len, from, to));
            }
        }
        pieces.retain(|&(_, from, to)| from < to);
        let (Some(&(first_len, first, _)), Some(&(last_len, _, last_end))) = (pieces.first(), pieces.last()) else {
            continue;
        };
        gaps.push(DashboardGap {
            after_range_id: a.id,
            first_len,
            first_candidate: index_to_candidate(&a.alphabet, first, first_len),
            last_len,
            last_candidate: index_to_candidate(&a.alphabet, last_end - 1, last_len),
            count: pieces.iter().map(|&(_, from, to)| to - from).sum(),
        });
    }
    gaps
}

/// Where the main sweep will carve next. Its stored position only moves
/// when it carves, and it jumps over a priority range's span only then (see
/// `ranges::claim_range`) - so until it next carves, the stored position can
/// sit at the start of work a priority range has already handed out, for
/// instance when that priority range was created while the sweep was
/// partway into its span. This moves past any such span, continuing at the
/// next length (within the target's bounds) when one runs to a length's
/// end, just as the sweep itself will.
fn effective_sweep_position(
    mut len: i64,
    mut index: Pos,
    alphabet: &str,
    priority_ranges: &[DashboardPriorityRange],
    lower_bound: &str,
    upper_bound: &str,
) -> (i64, Pos) {
    let max_len = max_supported_len(alphabet);
    loop {
        let covering = priority_ranges
            .iter()
            .find(|pr| pr.alphabet == alphabet && pr.candidate_len == len && pr.start_index <= index && index < pr.end_index);
        if let Some(pr) = covering {
            index = pr.end_index;
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

/// The last range in `ranges` (ordered by length, then start) that starts
/// before the main sweep's position - so the sweep marker goes right after
/// it. `None` if the sweep is before every range.
fn sweep_after_range_id(ranges: &[DashboardRange], cursor_len: i64, cursor_next_index: Pos) -> Option<i64> {
    ranges
        .iter()
        .take_while(|r| (r.candidate_len, r.start_index) < (cursor_len, cursor_next_index))
        .last()
        .map(|r| r.id)
}

#[derive(Serialize)]
pub struct DashboardPriorityRange {
    pub id: i64,
    pub priority: i64,
    pub pattern: String,
    pub candidate_len: i64,
    pub start_index: Pos,
    pub end_index: Pos,
    pub next_index: Pos,
    pub first_candidate: String,
    pub last_candidate: String,
    /// `None` once this priority range is exhausted (`next_index == end_index`)
    /// - see `ranges::claim_priority_range_chunk`.
    pub next_candidate: Option<String>,
    pub alphabet_name: String,
    pub alphabet: String,
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
    created_at: i64,
    alphabet_name: String,
    alphabet: String,
    priority_range_id: Option<i64>,
}

#[derive(Serialize)]
pub struct DashboardRange {
    pub id: i64,
    pub status: String,
    pub candidate_len: i64,
    pub start_index: Pos,
    pub end_index: Pos,
    /// The actual first/last candidate strings this range covers - `end_index`
    /// itself is exclusive (see `range_bound_filenames`), so the last candidate
    /// is decoded from `end_index - 1`.
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
    pub created_at: i64,
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
}

// Both queries below build a user's "username@hostname" in SQL (`||` gives
// NULL if either part is NULL, i.e. no such user) rather than selecting the
// two separately, since sqlx only maps rows of up to 16 columns to a tuple.
// An empty backend (users registered before clients sent one) reads as NULL.

pub async fn dashboard_data(State(state): State<AppState>) -> Result<Json<DashboardResponse>, AppError> {
    #[allow(clippy::type_complexity)]
    let target_rows: Vec<(i64, String, String, String, String, i64, i64, Option<String>, Option<String>, Option<String>, Option<i64>, String, String, i64, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT targets.id, targets.name, targets.status, \
                targets.lower_bound, targets.upper_bound, targets.hash_a, targets.hash_b, targets.found_filename, \
                found_user.username || '@' || found_user.hostname, NULLIF(found_user.backend, ''), targets.found_at, targets.alphabet_name, targets.alphabet, targets.priority, \
                targets.description, targets.skip_regex \
         FROM targets LEFT JOIN users AS found_user ON found_user.id = targets.found_by_user_id \
         ORDER BY targets.priority DESC, targets.created_at ASC",
    )
    .fetch_all(&state.pool)
    .await?;

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
                ranges.assigned_at, ranges.lease_expires_at, ranges.completed_at, ranges.created_at, \
                ranges.alphabet_name, ranges.alphabet, ranges.priority_range_id \
         FROM ranges LEFT JOIN users AS worker ON worker.id = ranges.last_assigned_user_id \
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

    let mut targets = Vec::with_capacity(target_rows.len());
    for (id, name, status, lower_bound, upper_bound, hash_a, hash_b, found_filename, found_by, found_by_backend, found_at, alphabet_name, alphabet, priority, description, skip_regex) in target_rows {
        let range_rows = ranges_by_target.remove(&id).unwrap_or_default();

        let ranges: Vec<DashboardRange> = range_rows
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
                    start_index,
                    end_index,
                    first_candidate: index_to_candidate(&row.alphabet, start_index, len),
                    last_candidate: index_to_candidate(&row.alphabet, end_index - 1, len),
                    progress_candidate: progress_index.map(|p| index_to_candidate(&row.alphabet, p, len)),
                    progress_percent: progress_index.map(|p| (p - start_index + 1) as f64 / (end_index - start_index) as f64 * 100.0),
                    worker: row.worker,
                    worker_backend: row.worker_backend,
                    assigned_at: row.assigned_at,
                    lease_expires_at: row.lease_expires_at,
                    completed_at: row.completed_at,
                    created_at: row.created_at,
                    alphabet_name: row.alphabet_name,
                    alphabet: row.alphabet,
                    priority_range_id: row.priority_range_id,
                }
            })
            .collect();

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
                DashboardPriorityRange {
                    id: pr.id,
                    priority: pr.priority,
                    candidate_len: pr.candidate_len,
                    start_index,
                    end_index,
                    next_index,
                    first_candidate: index_to_candidate(&pr.alphabet, start_index, pr.candidate_len),
                    last_candidate: index_to_candidate(&pr.alphabet, end_index - 1, pr.candidate_len),
                    next_candidate: (next_index < end_index).then(|| index_to_candidate(&pr.alphabet, next_index, pr.candidate_len)),
                    pattern: pr.pattern,
                    alphabet_name: pr.alphabet_name,
                    alphabet: pr.alphabet,
                }
            })
            .collect::<Vec<_>>();

        let (cursor_candidate_len, cursor_next_index) =
            effective_sweep_position(stored_cursor_len, stored_cursor_next_index, &cursor_alphabet, &priority_ranges, &lower_bound, &upper_bound);
        let cursor_candidate = index_to_candidate(&cursor_alphabet, cursor_next_index, cursor_candidate_len);
        let sweep_after_range_id = sweep_after_range_id(&ranges, cursor_candidate_len, cursor_next_index);

        targets.push(DashboardTarget {
            id,
            name,
            status,
            lower_bound,
            upper_bound,
            hash_a_hex: format!("0x{:08X}", i64_to_u32(hash_a)),
            hash_b_hex: format!("0x{:08X}", i64_to_u32(hash_b)),
            found_filename,
            found_by,
            found_by_backend,
            found_at,
            priority,
            description,
            skip_regex,
            alphabet_name,
            alphabet,
            start_len,
            cursor_candidate_len,
            cursor_next_index,
            cursor_candidate,
            sweep_after_range_id,
            priority_ranges,
            ranges,
            gaps,
        });
    }

    Ok(Json(DashboardResponse { targets }))
}

pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    fn range(id: i64, candidate_len: i64, start_index: Pos, end_index: Pos) -> DashboardRange {
        DashboardRange {
            id,
            status: "completed".into(),
            candidate_len,
            start_index,
            end_index,
            first_candidate: String::new(),
            last_candidate: String::new(),
            progress_candidate: None,
            progress_percent: None,
            worker: None,
            worker_backend: None,
            assigned_at: None,
            lease_expires_at: None,
            completed_at: None,
            created_at: 0,
            alphabet_name: "letters".into(),
            alphabet: LETTERS.into(),
            priority_range_id: None,
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

    fn priority_range(candidate_len: i64, start_index: Pos, end_index: Pos) -> DashboardPriorityRange {
        DashboardPriorityRange {
            id: 0,
            priority: 0,
            pattern: String::new(),
            candidate_len,
            start_index,
            end_index,
            next_index: end_index,
            first_candidate: String::new(),
            last_candidate: String::new(),
            next_candidate: None,
            alphabet_name: "letters".into(),
            alphabet: LETTERS.into(),
        }
    }

    #[test]
    fn effective_sweep_position_moves_past_priority_ranges_at_the_stored_position() {
        // A priority range created while the sweep was partway into its span
        // (MA-MZ = 312..338, sweep at 320): the sweep will jump to its end.
        let prs = vec![priority_range(2, 312, 338)];
        assert_eq!(effective_sweep_position(2, 320, LETTERS, &prs, "AA", "ZZ"), (2, 338));
        // Back-to-back spans are all skipped.
        let prs = vec![priority_range(2, 312, 338), priority_range(2, 338, 364)];
        assert_eq!(effective_sweep_position(2, 312, LETTERS, &prs, "AA", "ZZ"), (2, 364));
        // Nothing at the stored position: it stays.
        assert_eq!(effective_sweep_position(2, 300, LETTERS, &prs, "AA", "ZZ"), (2, 300));
    }

    #[test]
    fn effective_sweep_position_continues_at_the_next_length() {
        // A span running to the end of length 2 (ZA-ZZ = 650..676) moves the
        // sweep to the start of length 3.
        let prs = vec![priority_range(2, 650, 676)];
        assert_eq!(effective_sweep_position(2, 650, LETTERS, &prs, "AA", "ZZZ"), (3, 0));
    }

    #[test]
    fn sweep_after_range_id_is_the_last_range_starting_before_the_sweep() {
        let ranges = vec![range(1, 2, 0, 10), range(2, 2, 10, 20), range(3, 3, 5, 9)];
        assert_eq!(sweep_after_range_id(&ranges, 2, 20), Some(2));
        assert_eq!(sweep_after_range_id(&ranges, 2, 0), None);
        assert_eq!(sweep_after_range_id(&ranges, 4, 0), Some(3));
    }
}
