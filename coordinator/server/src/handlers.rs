use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use namebreak_protocol::{
    AdminCreatePriorityRangeRequest, AdminCreatePriorityRangeResponse, AdminCreateTargetRequest, AdminCreateTargetResponse, AdminPatchTargetRequest,
    AlphabetInfo, AlphabetsResponse, CompleteRequest, HeartbeatRequest, HeartbeatResponse, RegisterRequest, RegisterResponse, StatusResponse, TargetStatus,
    Version, PROTOCOL_VERSION,
};

use crate::alphabet::{
    alphabet_size, bound_indices_at_len, bounds_are_valid, candidate_to_index, compile_skip_regex, expand_priority_pattern, lookup_predefined_alphabet,
    max_supported_len, PREDEFINED_ALPHABETS,
};
use crate::auth::{AdminAuth, AuthedUser};
use crate::error::AppError;
use crate::models::{parse_hash_hex, u32_to_i64, Target, TargetProgress, User};
use crate::ranges::{self, PriorityRangeRemoval};
use crate::state::{generate_token, now_unix, AppState};

/// Parses `req.protocol_version` and rejects a MAJOR-version mismatch
/// against this server's own `PROTOCOL_VERSION` outright - nothing about
/// compatibility can be assumed across that boundary, so there's no point
/// letting an incompatible client proceed only to fail confusingly later.
fn resolve_client_protocol_version(req: &RegisterRequest) -> Result<Version, AppError> {
    let client_version = req.protocol_version.parse::<Version>().map_err(AppError::BadRequest)?;
    if client_version.major != PROTOCOL_VERSION.major {
        return Err(AppError::BadRequest(format!(
            "client protocol version {client_version} is incompatible with this server's protocol version {PROTOCOL_VERSION} \
             (major version mismatch) - please upgrade the client"
        )));
    }
    Ok(client_version)
}

pub async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, AppError> {
    let username = req.username.trim();
    let hostname = req.hostname.trim();
    if username.is_empty() || hostname.is_empty() {
        return Err(AppError::BadRequest("username and hostname are required".into()));
    }
    let client_version = resolve_client_protocol_version(&req)?.to_string();

    let now = now_unix();

    if let Some(existing) = sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = ? AND hostname = ?")
        .bind(username)
        .bind(hostname)
        .fetch_optional(&state.pool)
        .await?
    {
        // Refreshed on every registration, not just created once: a
        // returning client may have been upgraded (or downgraded) since it
        // last registered, and claim_range always wants the current picture.
        sqlx::query("UPDATE users SET last_seen_at = ?, protocol_version = ? WHERE id = ?")
            .bind(now)
            .bind(&client_version)
            .bind(existing.id)
            .execute(&state.pool)
            .await?;
        return Ok(Json(RegisterResponse { user_id: existing.id, token: existing.token, server_protocol_version: PROTOCOL_VERSION.to_string() }));
    }

    let token = generate_token();
    let user_id: i64 = sqlx::query_scalar(
        "INSERT INTO users (username, hostname, token, created_at, last_seen_at, protocol_version) VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(username)
    .bind(hostname)
    .bind(&token)
    .bind(now)
    .bind(now)
    .bind(&client_version)
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(RegisterResponse { user_id, token, server_protocol_version: PROTOCOL_VERSION.to_string() }))
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
        .map(|&(name, characters, (since_major, since_minor))| AlphabetInfo {
            name: name.to_string(),
            characters: characters.to_string(),
            size: alphabet_size(characters),
            since: format!("{since_major}.{since_minor}"),
        })
        .collect();
    Json(AlphabetsResponse { alphabets })
}

/// Rejects a skip_regex that doesn't compile as a regex (see
/// `alphabet::compile_skip_regex`). `None` or an empty/blank string is valid -
/// both mean "no skipping".
fn validate_skip_regex(skip_regex: &Option<String>) -> Result<(), AppError> {
    if let Some(pattern) = skip_regex {
        if !pattern.trim().is_empty() && compile_skip_regex(pattern).is_none() {
            return Err(AppError::BadRequest("skip_regex is not a valid regex".into()));
        }
    }
    Ok(())
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
        let valid: Vec<&str> = PREDEFINED_ALPHABETS.iter().map(|&(name, _, _)| name).collect();
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
    validate_skip_regex(&req.skip_regex)?;

    let mut tx = state.pool.begin().await?;
    let now = now_unix();
    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, max_backslash_count, alphabet_name, alphabet, status, priority, description, skip_regex, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?) RETURNING id",
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
    .bind(req.priority)
    .bind(&req.description)
    .bind(&req.skip_regex)
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
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_index, alphabet_name, alphabet) VALUES (?, ?, ?, ?, ?)")
        .bind(target_id)
        .bind(start_len)
        .bind(start_index)
        .bind(alphabet_name)
        .bind(alphabet)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(Json(AdminCreateTargetResponse { target_id }))
}

/// Resolves an `alphabet_name` patch into the `(alphabet_name, alphabet)` pair
/// to store, validating it the same way `admin_create_target` validates a
/// brand-new target: the name must be one of `PREDEFINED_ALPHABETS`, and the
/// target's own (immutable) bounds must still consist of characters in it.
/// Already-carved ranges and the carving cursor are untouched here - the
/// cursor is only translated onto the new alphabet lazily, the first time
/// `ranges::claim_range` next carves fresh work for this target (see
/// `alphabet::transition_alphabet_cursor`).
async fn resolve_alphabet_patch(pool: &sqlx::SqlitePool, target_id: i64, alphabet_name: &str) -> Result<(String, String), AppError> {
    let Some(alphabet) = lookup_predefined_alphabet(alphabet_name) else {
        let valid: Vec<&str> = PREDEFINED_ALPHABETS.iter().map(|&(name, _, _)| name).collect();
        return Err(AppError::BadRequest(format!("unknown alphabet_name '{alphabet_name}' - valid names: {}", valid.join(", "))));
    };
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_optional(pool).await?.ok_or(AppError::NotFound)?;

    let cap = max_supported_len(alphabet) as usize;
    let lower_for_validation: String = target.lower_bound.chars().take(cap).collect();
    let upper_for_validation: String = target.upper_bound.chars().take(cap).collect();
    if candidate_to_index(alphabet, &lower_for_validation).is_none() {
        return Err(AppError::BadRequest("target's lower_bound contains a character outside the new alphabet".into()));
    }
    if candidate_to_index(alphabet, &upper_for_validation).is_none() {
        return Err(AppError::BadRequest("target's upper_bound contains a character outside the new alphabet".into()));
    }

    Ok((alphabet_name.to_string(), alphabet.to_string()))
}

pub async fn admin_patch_target(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(target_id): Path<i64>,
    Json(req): Json<AdminPatchTargetRequest>,
) -> Result<StatusCode, AppError> {
    if let Some(status) = &req.status {
        if status != "active" && status != "paused" {
            return Err(AppError::BadRequest("status must be 'active' or 'paused'".into()));
        }
    }
    if req.status.is_none() && req.priority.is_none() && req.description.is_none() && req.skip_regex.is_none() && req.alphabet_name.is_none() {
        return Err(AppError::BadRequest("at least one of status, priority, description, skip_regex or alphabet_name must be provided".into()));
    }
    validate_skip_regex(&req.skip_regex)?;

    let (alphabet_name, alphabet) = match &req.alphabet_name {
        Some(name) => {
            let (name, chars) = resolve_alphabet_patch(&state.pool, target_id, name).await?;
            (Some(name), Some(chars))
        }
        None => (None, None),
    };

    let mut tx = state.pool.begin().await?;

    // Checked up front, in the same transaction as the UPDATE below, so we
    // can tell "no such target" (404) apart from "target exists but is
    // solved, and thus immutable by design" (409) - the UPDATE's own
    // `status != 'solved'` guard can't distinguish the two on its own, since
    // both leave `rows_affected() == 0`.
    let current_status: Option<(String,)> =
        sqlx::query_as("SELECT status FROM targets WHERE id = ?").bind(target_id).fetch_optional(&mut *tx).await?;
    match current_status {
        None => return Err(AppError::NotFound),
        Some((status,)) if status == "solved" => {
            return Err(AppError::Conflict("target is already solved and can no longer be modified".into()));
        }
        Some(_) => {}
    }

    let result = sqlx::query(
        "UPDATE targets SET status = COALESCE(?, status), priority = COALESCE(?, priority), \
         description = COALESCE(?, description), skip_regex = COALESCE(?, skip_regex), \
         alphabet_name = COALESCE(?, alphabet_name), alphabet = COALESCE(?, alphabet) \
         WHERE id = ? AND status != 'solved'",
    )
    .bind(&req.status)
    .bind(req.priority)
    .bind(&req.description)
    .bind(&req.skip_regex)
    .bind(&alphabet_name)
    .bind(&alphabet)
    .bind(target_id)
    .execute(&mut *tx)
    .await?;
    debug_assert!(result.rows_affected() > 0, "target existed and wasn't solved per the check above");

    // Priority ranges are translated eagerly, in the same transaction as
    // the alphabet change itself - see
    // ranges::migrate_priority_ranges_to_new_alphabet's doc comment for why
    // this can't be deferred the way the target's own cursor transition is.
    if let (Some(name), Some(chars)) = (&alphabet_name, &alphabet) {
        ranges::migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, name, chars, now_unix()).await?;
    }

    tx.commit().await?;
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

/// Fast-tracks a specific, bounded slice of a target's search space ahead of
/// its normal sequential sweep - see `ranges::claim_range`,
/// `ranges::claim_priority_range_chunk`. `req.pattern` may expand (via
/// `alphabet::expand_priority_pattern`) into several concrete prefixes, each
/// becoming its own `priority_ranges` row sharing `req.priority` and
/// `req.length`.
pub async fn admin_create_priority_range(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(target_id): Path<i64>,
    Json(req): Json<AdminCreatePriorityRangeRequest>,
) -> Result<Json<AdminCreatePriorityRangeResponse>, AppError> {
    if req.length <= 0 {
        return Err(AppError::BadRequest("length must be positive".into()));
    }

    // Everything from here on reads and writes inside one transaction - the
    // clamp below depends on the target's cursor not moving out from under
    // it, which a concurrent claim_range (its own separate transaction)
    // would otherwise be free to do between a read and this function's own
    // INSERT.
    let mut tx = state.pool.begin().await?;

    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?")
        .bind(target_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(AppError::NotFound)?;

    let cap = max_supported_len(&target.alphabet);
    if req.length > cap {
        return Err(AppError::BadRequest(format!("length {} exceeds this target's alphabet's max supported length ({cap})", req.length)));
    }

    let prefixes = expand_priority_pattern(&target.alphabet, &req.pattern).map_err(AppError::BadRequest)?;
    // Every expansion shares the same atom count (see expand_priority_pattern), so any one of them tells us the pattern's length.
    let pattern_len = prefixes[0].chars().count() as i64;
    if pattern_len > req.length {
        return Err(AppError::BadRequest(format!(
            "priority pattern is {pattern_len} characters long, which is longer than the requested length ({})",
            req.length
        )));
    }

    // A priority range only makes sense ahead of where the target's own
    // cursor already reached - see alphabet::expand_priority_pattern's doc
    // comment. Everything before that has already been fully searched by
    // the main sweep, under its own bookkeeping; there's nothing left here
    // to fast-track.
    let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
        .bind(target_id)
        .fetch_one(&mut *tx)
        .await?;
    if req.length < progress.candidate_len {
        return Err(AppError::BadRequest(format!(
            "the target has already fully searched every {}-character candidate - nothing left to prioritize there",
            req.length
        )));
    }
    // The cursor's own next_index is only safe to compare against a
    // freshly-computed start_index (both below) when it's denominated in
    // the same alphabet the new priority range is being created under -
    // otherwise (a patch happened but no claim has resolved the lazy
    // transition yet - see alphabet::transition_alphabet_cursor) next_index
    // is still an old-alphabet index, not comparable to a new-alphabet one.
    // Skipping the clamp in that rare window just means the new range
    // starts at its literal declared beginning, exactly as it already does
    // whenever req.length is ahead of the cursor's length.
    let clamp_to_cursor = req.length == progress.candidate_len && progress.alphabet_name == target.alphabet_name;

    let now = now_unix();
    let mut priority_range_ids = Vec::with_capacity(prefixes.len());

    for prefix in &prefixes {
        let (start_index, end_index_inclusive) = bound_indices_at_len(&target.alphabet, prefix, prefix, req.length);
        let end_index = end_index_inclusive + 1;
        let next_index = if clamp_to_cursor { start_index.max(progress.next_index) } else { start_index };
        if next_index >= end_index {
            return Err(AppError::BadRequest(format!(
                "prefix '{prefix}' at length {} has already been fully searched by the main sweep - nothing left to prioritize",
                req.length
            )));
        }

        // Each priority range permanently owns its declared span (see
        // ranges::find_priority_boundary) - two overlapping ones at the same
        // length *and alphabet* would double-book the same addresses. One
        // frozen under a different alphabet isn't index-comparable at all
        // (same reasoning as find_priority_boundary), so it's excluded here
        // rather than risking a wrong comparison.
        let overlap: Option<(i64,)> = sqlx::query_as(
            "SELECT id FROM priority_ranges WHERE target_id = ? AND candidate_len = ? AND alphabet_name = ? AND start_index < ? AND end_index > ?",
        )
        .bind(target_id)
        .bind(req.length)
        .bind(&target.alphabet_name)
        .bind(end_index)
        .bind(start_index)
        .fetch_optional(&mut *tx)
        .await?;
        if let Some((existing_id,)) = overlap {
            return Err(AppError::BadRequest(format!("prefix '{prefix}' at length {} overlaps existing priority range #{existing_id}", req.length)));
        }

        let id: i64 = sqlx::query_scalar(
            "INSERT INTO priority_ranges (target_id, priority, pattern, prefix, candidate_len, start_index, end_index, next_index, alphabet_name, alphabet, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(target_id)
        .bind(req.priority)
        .bind(&req.pattern)
        .bind(prefix)
        .bind(req.length)
        .bind(start_index)
        .bind(end_index)
        .bind(next_index)
        .bind(&target.alphabet_name)
        .bind(&target.alphabet)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;
        priority_range_ids.push(id);
    }

    tx.commit().await?;
    Ok(Json(AdminCreatePriorityRangeResponse { priority_range_ids }))
}

pub async fn admin_delete_priority_range(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(priority_range_id): Path<i64>,
) -> Result<StatusCode, AppError> {
    match ranges::retire_or_delete_priority_range(&state.pool, priority_range_id).await? {
        Some(PriorityRangeRemoval::Deleted) | Some(PriorityRangeRemoval::Retired) => Ok(StatusCode::NO_CONTENT),
        None => Err(AppError::NotFound),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register_request(protocol_version: &str) -> RegisterRequest {
        RegisterRequest { username: "u".into(), hostname: "h".into(), protocol_version: protocol_version.to_string() }
    }

    #[test]
    fn resolve_client_protocol_version_accepts_a_matching_major_version() {
        assert_eq!(resolve_client_protocol_version(&register_request("1.4.2")).unwrap(), Version::new(1, 4, 2));
    }

    #[test]
    fn resolve_client_protocol_version_rejects_a_different_major_version() {
        let err = resolve_client_protocol_version(&register_request("2.0.0")).unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[test]
    fn resolve_client_protocol_version_rejects_a_malformed_version_string() {
        let err = resolve_client_protocol_version(&register_request("not-a-version")).unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }
}
