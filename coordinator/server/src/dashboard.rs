//! Human-facing dashboard: a static page (served at `GET /`) that polls
//! `GET /api/v1/dashboard` for target/range/worker data. No build step, no JS
//! framework - the page is a single self-contained file embedded at compile time.

use std::collections::HashMap;

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::Serialize;

use crate::alphabet::index_to_candidate;
use crate::error::AppError;
use crate::models::i64_to_u32;
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
    /// Where the target's own main sweep currently is - see
    /// `models::TargetProgress`. Compared against each `DashboardPriorityRange`
    /// so the dashboard can show the (never persisted - see
    /// `ranges::find_priority_boundary`) gap between them.
    pub cursor_candidate_len: i64,
    pub cursor_next_index: i64,
    pub cursor_candidate: String,
    pub priority_ranges: Vec<DashboardPriorityRange>,
    pub ranges: Vec<DashboardRange>,
}

#[derive(Serialize)]
pub struct DashboardPriorityRange {
    pub id: i64,
    pub priority: i64,
    pub pattern: String,
    pub candidate_len: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub next_index: i64,
    pub first_candidate: String,
    pub last_candidate: String,
    /// `None` once this priority range is exhausted (`next_index == end_index`)
    /// - see `ranges::claim_priority_range_chunk`.
    pub next_candidate: Option<String>,
    pub alphabet_name: String,
    pub alphabet: String,
}

#[derive(Serialize)]
pub struct DashboardRange {
    pub id: i64,
    pub status: String,
    pub candidate_len: i64,
    pub start_index: i64,
    pub end_index: i64,
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

fn display_name(username: Option<String>, hostname: Option<String>) -> Option<String> {
    match (username, hostname) {
        (Some(u), Some(h)) => Some(format!("{u}@{h}")),
        _ => None,
    }
}

pub async fn dashboard_data(State(state): State<AppState>) -> Result<Json<DashboardResponse>, AppError> {
    #[allow(clippy::type_complexity)]
    let target_rows: Vec<(i64, String, String, String, String, i64, i64, Option<String>, Option<String>, Option<String>, Option<i64>, String, String, i64, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT targets.id, targets.name, targets.status, \
                targets.lower_bound, targets.upper_bound, targets.hash_a, targets.hash_b, targets.found_filename, \
                found_user.username, found_user.hostname, targets.found_at, targets.alphabet_name, targets.alphabet, targets.priority, \
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
    #[allow(clippy::type_complexity)]
    let all_ranges: Vec<(i64, i64, String, i64, i64, i64, Option<i64>, Option<String>, Option<String>, Option<i64>, Option<i64>, Option<i64>, i64, String, String, Option<i64>)> = sqlx::query_as(
        "SELECT ranges.target_id, ranges.id, ranges.status, ranges.candidate_len, ranges.start_index, ranges.end_index, \
                ranges.progress_index, worker.username, worker.hostname, \
                ranges.assigned_at, ranges.lease_expires_at, ranges.completed_at, ranges.created_at, \
                ranges.alphabet_name, ranges.alphabet, ranges.priority_range_id \
         FROM ranges LEFT JOIN users AS worker ON worker.id = ranges.last_assigned_user_id \
         ORDER BY ranges.created_at DESC",
    )
    .fetch_all(&state.pool)
    .await?;
    let mut ranges_by_target: HashMap<i64, Vec<_>> = HashMap::new();
    for row in all_ranges {
        let target_id = row.0;
        ranges_by_target.entry(target_id).or_default().push(row);
    }

    let all_progress: Vec<(i64, i64, i64, String)> =
        sqlx::query_as("SELECT target_id, candidate_len, next_index, alphabet FROM target_progress").fetch_all(&state.pool).await?;
    let progress_by_target: HashMap<i64, (i64, i64, String)> =
        all_progress.into_iter().map(|(target_id, candidate_len, next_index, alphabet)| (target_id, (candidate_len, next_index, alphabet))).collect();

    #[allow(clippy::type_complexity)]
    let all_priority_ranges: Vec<(i64, i64, i64, String, i64, i64, i64, i64, String, String)> = sqlx::query_as(
        "SELECT target_id, id, priority, pattern, candidate_len, start_index, end_index, next_index, alphabet_name, alphabet \
         FROM priority_ranges ORDER BY priority DESC, created_at ASC",
    )
    .fetch_all(&state.pool)
    .await?;
    let mut priority_ranges_by_target: HashMap<i64, Vec<_>> = HashMap::new();
    for row in all_priority_ranges {
        let target_id = row.0;
        priority_ranges_by_target.entry(target_id).or_default().push(row);
    }

    let mut targets = Vec::with_capacity(target_rows.len());
    for (id, name, status, lower_bound, upper_bound, hash_a, hash_b, found_filename, found_username, found_hostname, found_at, alphabet_name, alphabet, priority, description, skip_regex) in target_rows {
        let range_rows = ranges_by_target.remove(&id).unwrap_or_default();

        let ranges = range_rows
            .into_iter()
            .map(
                |(_target_id, range_id, r_status, candidate_len, start_index, end_index, progress_index, worker_username, worker_hostname, assigned_at, lease_expires_at, completed_at, created_at, range_alphabet_name, range_alphabet, priority_range_id)| {
                    // Each range is decoded with its OWN alphabet, not the
                    // target's current one - a range carved before the
                    // target's alphabet was last patched (see
                    // handlers::admin_patch_target) must still be shown with
                    // the alphabet it actually holds candidates in.
                    DashboardRange {
                        id: range_id,
                        status: r_status,
                        candidate_len,
                        start_index,
                        end_index,
                        first_candidate: index_to_candidate(&range_alphabet, start_index, candidate_len),
                        last_candidate: index_to_candidate(&range_alphabet, end_index - 1, candidate_len),
                        progress_candidate: progress_index.map(|p| index_to_candidate(&range_alphabet, p, candidate_len)),
                        progress_percent: progress_index.map(|p| {
                            (p - start_index + 1) as f64 / (end_index - start_index) as f64 * 100.0
                        }),
                        worker: display_name(worker_username, worker_hostname),
                        assigned_at,
                        lease_expires_at,
                        completed_at,
                        created_at,
                        alphabet_name: range_alphabet_name,
                        alphabet: range_alphabet,
                        priority_range_id,
                    }
                },
            )
            .collect();

        let (cursor_candidate_len, cursor_next_index, cursor_alphabet) = progress_by_target
            .get(&id)
            .cloned()
            .ok_or_else(|| AppError::Internal(format!("missing target_progress row for target {id}")))?;
        let cursor_candidate = index_to_candidate(&cursor_alphabet, cursor_next_index, cursor_candidate_len);

        let priority_range_rows = priority_ranges_by_target.remove(&id).unwrap_or_default();
        let priority_ranges = priority_range_rows
            .into_iter()
            .map(|(_target_id, pr_id, pr_priority, pattern, candidate_len, start_index, end_index, next_index, pr_alphabet_name, pr_alphabet)| DashboardPriorityRange {
                id: pr_id,
                priority: pr_priority,
                pattern,
                candidate_len,
                start_index,
                end_index,
                next_index,
                first_candidate: index_to_candidate(&pr_alphabet, start_index, candidate_len),
                last_candidate: index_to_candidate(&pr_alphabet, end_index - 1, candidate_len),
                next_candidate: (next_index < end_index).then(|| index_to_candidate(&pr_alphabet, next_index, candidate_len)),
                alphabet_name: pr_alphabet_name,
                alphabet: pr_alphabet,
            })
            .collect();

        targets.push(DashboardTarget {
            id,
            name,
            status,
            lower_bound,
            upper_bound,
            hash_a_hex: format!("0x{:08X}", i64_to_u32(hash_a)),
            hash_b_hex: format!("0x{:08X}", i64_to_u32(hash_b)),
            found_filename,
            found_by: display_name(found_username, found_hostname),
            found_at,
            priority,
            description,
            skip_regex,
            alphabet_name,
            alphabet,
            cursor_candidate_len,
            cursor_next_index,
            cursor_candidate,
            priority_ranges,
            ranges,
        });
    }

    Ok(Json(DashboardResponse { targets }))
}

pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}
