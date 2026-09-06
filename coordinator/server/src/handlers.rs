use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use namebreak_protocol::{
    AdminCreateTargetRequest, AdminCreateTargetResponse, AdminPatchTargetRequest, AlphabetInfo, AlphabetsResponse,
    CompleteRequest, HeartbeatRequest, HeartbeatResponse, RegisterRequest, RegisterResponse, StatusResponse, TargetStatus,
};

use crate::alphabet::{
    alphabet_size, bound_indices_at_len, bounds_are_valid, candidate_to_index, lookup_predefined_alphabet, max_supported_len, PREDEFINED_ALPHABETS,
};
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
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<HeartbeatResponse>, AppError> {
    let outcome = ranges::heartbeat_range(&state.pool, &user, range_id, req.last_hash_a_match_filename).await?;
    Ok(Json(HeartbeatResponse { lease_seconds: outcome.lease_seconds, target_solved: outcome.target_solved }))
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

pub async fn alphabets() -> Json<AlphabetsResponse> {
    let alphabets = PREDEFINED_ALPHABETS
        .iter()
        .map(|&(name, characters)| AlphabetInfo { name: name.to_string(), characters: characters.to_string(), size: alphabet_size(characters) })
        .collect();
    Json(AlphabetsResponse { alphabets })
}

pub async fn admin_create_target(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Json(req): Json<AdminCreateTargetRequest>,
) -> Result<Json<AdminCreateTargetResponse>, AppError> {
    if req.name.trim().is_empty() {
        return Err(AppError::BadRequest("name is required".into()));
    }
    if req.max_backslash_count < 0 {
        return Err(AppError::BadRequest("max_backslash_count must be >= 0 (0 means unlimited)".into()));
    }
    let alphabet_name = req.alphabet_name.as_deref().unwrap_or("size49");
    let Some(alphabet) = lookup_predefined_alphabet(alphabet_name) else {
        let valid: Vec<&str> = PREDEFINED_ALPHABETS.iter().map(|&(name, _)| name).collect();
        return Err(AppError::BadRequest(format!("unknown alphabet_name '{alphabet_name}' - valid names: {}", valid.join(", "))));
    };

    // Only the first `cap` characters of a bound are ever consulted (carving never
    // searches past this length, and bound_indices_at_len truncates to whatever
    // length it's asked about) - so a bound can be longer than this without needing
    // to fit as a literal candidate itself. That's the point: bounds are often a
    // neighboring *known* filename from elsewhere (a listfile, an adjacent hash-table
    // entry) used purely for its alphabetical position, with no relation at all to
    // this target's own prefix/suffix/length - e.g. lower_bound "GLUE\PALCS\DLG.GRP"
    // and upper_bound "MUSIC\MENGSKVICTORY.WAV" are both valid even though the actual
    // target has a completely different (and unknown) prefix and a suffix of ".WAV".
    // The *stored* bound keeps everything the operator gave it, though (truncation
    // here is just to keep this character check from overflowing on an oversized
    // string) - bound_indices_at_len truncates lazily wherever it actually matters,
    // so pre-truncating what gets stored would only maim it on the dashboard for no
    // behavioral gain.
    let cap = max_supported_len(alphabet) as usize;
    let lower_bound = req.lower_bound.as_str();
    let upper_bound = req.upper_bound.as_str();
    let lower_for_validation: String = lower_bound.chars().take(cap).collect();
    let upper_for_validation: String = upper_bound.chars().take(cap).collect();

    if candidate_to_index(alphabet, &lower_for_validation).is_none() {
        return Err(AppError::BadRequest("lower_bound contains a character outside the chosen alphabet".into()));
    }
    if candidate_to_index(alphabet, &upper_for_validation).is_none() {
        return Err(AppError::BadRequest("upper_bound contains a character outside the chosen alphabet".into()));
    }
    if !bounds_are_valid(alphabet, lower_bound, upper_bound) {
        return Err(AppError::BadRequest("lower_bound must be alphabetically before upper_bound".into()));
    }
    let hash_a = parse_hash_hex(&req.hash_a_hex).map_err(|_| AppError::BadRequest("invalid hash_a_hex".into()))?;
    let hash_b = parse_hash_hex(&req.hash_b_hex).map_err(|_| AppError::BadRequest("invalid hash_b_hex".into()))?;

    let mut tx = state.pool.begin().await?;
    let now = now_unix();
    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, max_backslash_count, alphabet_name, alphabet, status, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&req.prefix)
    .bind(&req.suffix)
    .bind(u32_to_i64(hash_a))
    .bind(u32_to_i64(hash_b))
    .bind(lower_bound)
    .bind(upper_bound)
    .bind(req.prune_symbol_runs as i64)
    .bind(req.max_backslash_count)
    .bind(alphabet_name)
    .bind(alphabet)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;

    // Always start at the shortest possible candidate length, symmetric with always
    // searching up to max_supported_len at the top end - a bound doesn't get to skip
    // short lengths just because it's itself longer (see the comment above: at length
    // 1, bound_indices_at_len simply truncates each bound down to its first
    // character, which is exactly the right constraint there too).
    let start_len = 1i64;
    let start_index = bound_indices_at_len(alphabet, lower_bound, upper_bound, start_len).0;
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_index) VALUES (?, ?, ?)")
        .bind(target_id)
        .bind(start_len)
        .bind(start_index)
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

pub async fn admin_delete_target(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(target_id): Path<i64>,
) -> Result<StatusCode, AppError> {
    if ranges::delete_target(&state.pool, target_id).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound)
    }
}
