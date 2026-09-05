use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use namebreak_protocol::{
    AdminCreateTargetRequest, AdminCreateTargetResponse, AdminPatchTargetRequest,
    CompleteRequest, HeartbeatResponse, RegisterRequest, RegisterResponse, StatusResponse, TargetStatus,
};

use crate::alphabet::max_supported_len;
use crate::auth::{AdminAuth, AuthedUser};
use crate::error::AppError;
use crate::models::{parse_hash_hex, u32_to_i64, User};
use crate::ranges;
use crate::state::{generate_token, now_unix, AppState};

pub async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, AppError> {
    let username = req.username.trim();
    let hostname = req.hostname.trim();
    if username.is_empty() || hostname.is_empty() {
        return Err(AppError::BadRequest("username and hostname are required".into()));
    }

    let now = now_unix();

    if let Some(existing) = sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = ? AND hostname = ?")
        .bind(username)
        .bind(hostname)
        .fetch_optional(&state.pool)
        .await?
    {
        sqlx::query("UPDATE users SET last_seen_at = ? WHERE id = ?")
            .bind(now)
            .bind(existing.id)
            .execute(&state.pool)
            .await?;
        return Ok(Json(RegisterResponse { user_id: existing.id, token: existing.token }));
    }

    let token = generate_token();
    let user_id: i64 = sqlx::query_scalar(
        "INSERT INTO users (username, hostname, token, created_at, last_seen_at) VALUES (?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(username)
    .bind(hostname)
    .bind(&token)
    .bind(now)
    .bind(now)
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(RegisterResponse { user_id, token }))
}

pub async fn claim(
    State(state): State<AppState>,
    AuthedUser(user): AuthedUser,
) -> Result<Response, AppError> {
    match ranges::claim_range(&state.pool, &state.config, &user).await? {
        Some(resp) => Ok((StatusCode::OK, Json(resp)).into_response()),
        None => Ok(StatusCode::NO_CONTENT.into_response()),
    }
}

pub async fn heartbeat(
    State(state): State<AppState>,
    AuthedUser(user): AuthedUser,
    Path(range_id): Path<i64>,
) -> Result<Json<HeartbeatResponse>, AppError> {
    let lease_seconds = ranges::heartbeat_range(&state.pool, &user, range_id).await?;
    Ok(Json(HeartbeatResponse { lease_seconds }))
}

pub async fn complete(
    State(state): State<AppState>,
    AuthedUser(user): AuthedUser,
    Path(range_id): Path<i64>,
    Json(req): Json<CompleteRequest>,
) -> Result<StatusCode, AppError> {
    let outcome = ranges::complete_range(
        &state.pool,
        &state.config,
        &user,
        range_id,
        req.found,
        req.filename,
        req.elapsed_seconds,
        req.candidates_processed,
    )
    .await?;
    if outcome.target_solved {
        tracing::info!(range_id, user_id = user.id, "target solved");
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn status(State(state): State<AppState>) -> Result<Json<StatusResponse>, AppError> {
    let rows: Vec<(i64, String, String, Option<String>)> =
        sqlx::query_as("SELECT id, name, status, found_filename FROM targets ORDER BY created_at ASC")
            .fetch_all(&state.pool)
            .await?;
    let targets = rows
        .into_iter()
        .map(|(id, name, status, found_filename)| TargetStatus { id, name, status, found_filename })
        .collect();
    Ok(Json(StatusResponse { targets }))
}

pub async fn admin_create_target(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Json(req): Json<AdminCreateTargetRequest>,
) -> Result<Json<AdminCreateTargetResponse>, AppError> {
    if req.name.trim().is_empty() {
        return Err(AppError::BadRequest("name is required".into()));
    }
    if req.min_len < 1 || req.max_len < req.min_len {
        return Err(AppError::BadRequest("min_len must be >= 1 and <= max_len".into()));
    }
    let cap = max_supported_len();
    if req.max_len > cap {
        return Err(AppError::BadRequest(format!(
            "max_len ({}) exceeds this server's supported maximum ({cap}) - beyond this a range's index no longer fits an i64",
            req.max_len
        )));
    }
    let hash_a = parse_hash_hex(&req.hash_a_hex).map_err(|_| AppError::BadRequest("invalid hash_a_hex".into()))?;
    let hash_b = parse_hash_hex(&req.hash_b_hex).map_err(|_| AppError::BadRequest("invalid hash_b_hex".into()))?;

    let mut tx = state.pool.begin().await?;
    let now = now_unix();
    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, min_len, max_len, prune_symbol_runs, status, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&req.prefix)
    .bind(&req.suffix)
    .bind(u32_to_i64(hash_a))
    .bind(u32_to_i64(hash_b))
    .bind(req.min_len)
    .bind(req.max_len)
    .bind(req.prune_symbol_runs as i64)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;

    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_index) VALUES (?, ?, 0)")
        .bind(target_id)
        .bind(req.min_len)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(Json(AdminCreateTargetResponse { target_id }))
}

pub async fn admin_patch_target(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(target_id): Path<i64>,
    Json(req): Json<AdminPatchTargetRequest>,
) -> Result<StatusCode, AppError> {
    if req.status != "active" && req.status != "paused" {
        return Err(AppError::BadRequest("status must be 'active' or 'paused'".into()));
    }
    let result = sqlx::query("UPDATE targets SET status = ? WHERE id = ? AND status != 'solved'")
        .bind(&req.status)
        .bind(target_id)
        .execute(&state.pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(AppError::NotFound);
    }
    Ok(StatusCode::NO_CONTENT)
}
