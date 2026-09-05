//! Human-facing dashboard: a static page (served at `GET /`) that polls
//! `GET /api/v1/dashboard` for target/range/worker data. No build step, no JS
//! framework - the page is a single self-contained file embedded at compile time.

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::Serialize;

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
    pub found_filename: Option<String>,
    pub found_by: Option<String>,
    pub ranges: Vec<DashboardRange>,
}

#[derive(Serialize)]
pub struct DashboardRange {
    pub id: i64,
    pub status: String,
    pub candidate_len: i64,
    pub start_index: i64,
    pub end_index: i64,
    /// "username@hostname" of whoever last claimed this range, even if it was
    /// since reclaimed - see the migration adding `last_assigned_user_id`.
    pub worker: Option<String>,
    pub assigned_at: Option<i64>,
    pub lease_expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub created_at: i64,
}

fn display_name(username: Option<String>, hostname: Option<String>) -> Option<String> {
    match (username, hostname) {
        (Some(u), Some(h)) => Some(format!("{u}@{h}")),
        _ => None,
    }
}

pub async fn dashboard_data(State(state): State<AppState>) -> Result<Json<DashboardResponse>, AppError> {
    let target_rows: Vec<(i64, String, String, Option<String>, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT targets.id, targets.name, targets.status, targets.found_filename, \
                found_user.username, found_user.hostname \
         FROM targets LEFT JOIN users AS found_user ON found_user.id = targets.found_by_user_id \
         ORDER BY targets.created_at ASC",
    )
    .fetch_all(&state.pool)
    .await?;

    let mut targets = Vec::with_capacity(target_rows.len());
    for (id, name, status, found_filename, found_username, found_hostname) in target_rows {
        let range_rows: Vec<(i64, String, i64, i64, i64, Option<String>, Option<String>, Option<i64>, Option<i64>, Option<i64>, i64)> = sqlx::query_as(
            "SELECT ranges.id, ranges.status, ranges.candidate_len, ranges.start_index, ranges.end_index, \
                    worker.username, worker.hostname, \
                    ranges.assigned_at, ranges.lease_expires_at, ranges.completed_at, ranges.created_at \
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
                |(range_id, r_status, candidate_len, start_index, end_index, worker_username, worker_hostname, assigned_at, lease_expires_at, completed_at, created_at)| {
                    DashboardRange {
                        id: range_id,
                        status: r_status,
                        candidate_len,
                        start_index,
                        end_index,
                        worker: display_name(worker_username, worker_hostname),
                        assigned_at,
                        lease_expires_at,
                        completed_at,
                        created_at,
                    }
                },
            )
            .collect();

        targets.push(DashboardTarget {
            id,
            name,
            status,
            found_filename,
            found_by: display_name(found_username, found_hostname),
            ranges,
        });
    }

    Ok(Json(DashboardResponse { targets }))
}

pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}
