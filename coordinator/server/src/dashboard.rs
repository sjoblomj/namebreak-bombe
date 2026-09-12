//! Human-facing dashboard: a static page (served at `GET /`) that polls
//! `GET /api/v1/dashboard` for target/range/worker data. No build step, no JS
//! framework - the page is a single self-contained file embedded at compile time.

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::Serialize;

use crate::alphabet::index_to_candidate;
use crate::error::AppError;
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
    pub found_filename: Option<String>,
    pub found_by: Option<String>,
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
    pub ranges: Vec<DashboardRange>,
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
}

fn display_name(username: Option<String>, hostname: Option<String>) -> Option<String> {
    match (username, hostname) {
        (Some(u), Some(h)) => Some(format!("{u}@{h}")),
        _ => None,
    }
}

pub async fn dashboard_data(State(state): State<AppState>) -> Result<Json<DashboardResponse>, AppError> {
    #[allow(clippy::type_complexity)]
    let target_rows: Vec<(i64, String, String, String, String, Option<String>, Option<String>, Option<String>, String, String, i64, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT targets.id, targets.name, targets.status, \
                targets.lower_bound, targets.upper_bound, targets.found_filename, \
                found_user.username, found_user.hostname, targets.alphabet_name, targets.alphabet, targets.priority, \
                targets.description, targets.skip_regex \
         FROM targets LEFT JOIN users AS found_user ON found_user.id = targets.found_by_user_id \
         ORDER BY targets.priority DESC, targets.created_at ASC",
    )
    .fetch_all(&state.pool)
    .await?;

    let mut targets = Vec::with_capacity(target_rows.len());
    for (id, name, status, lower_bound, upper_bound, found_filename, found_username, found_hostname, alphabet_name, alphabet, priority, description, skip_regex) in target_rows {
        #[allow(clippy::type_complexity)]
        let range_rows: Vec<(i64, String, i64, i64, i64, Option<i64>, Option<String>, Option<String>, Option<i64>, Option<i64>, Option<i64>, i64, String, String)> = sqlx::query_as(
            "SELECT ranges.id, ranges.status, ranges.candidate_len, ranges.start_index, ranges.end_index, \
                    ranges.progress_index, worker.username, worker.hostname, \
                    ranges.assigned_at, ranges.lease_expires_at, ranges.completed_at, ranges.created_at, \
                    ranges.alphabet_name, ranges.alphabet \
             FROM ranges LEFT JOIN users AS worker ON worker.id = ranges.last_assigned_user_id \
             WHERE ranges.target_id = ? \
             ORDER BY ranges.created_at DESC",
        )
        .bind(id)
        .fetch_all(&state.pool)
        .await?;

        let ranges = range_rows
            .into_iter()
            .map(
                |(range_id, r_status, candidate_len, start_index, end_index, progress_index, worker_username, worker_hostname, assigned_at, lease_expires_at, completed_at, created_at, range_alphabet_name, range_alphabet)| {
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
                    }
                },
            )
            .collect();

        targets.push(DashboardTarget {
            id,
            name,
            status,
            lower_bound,
            upper_bound,
            found_filename,
            found_by: display_name(found_username, found_hostname),
            priority,
            description,
            skip_regex,
            alphabet_name,
            alphabet,
            ranges,
        });
    }

    Ok(Json(DashboardResponse { targets }))
}

pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}
