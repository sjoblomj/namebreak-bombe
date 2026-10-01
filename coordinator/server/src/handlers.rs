use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use namebreak_protocol::{
    AdminCreatePriorityRangeRequest, AdminCreatePriorityRangeResponse, AdminCreateSkipRangeRequest, AdminCreateSkipRangeResponse, AdminCreateTargetRequest,
    AdminCreateTargetResponse, AdminDeletePriorityRangeResponse, AdminDeleteSkipRangeResponse, AdminPatchTargetRequest,
    AlphabetInfo, ClientReleases, AlphabetsResponse, CompleteRequest, HeartbeatRequest, HeartbeatResponse, RegisterRequest, RegisterResponse, StatusResponse, TargetStatus,
    Version, PROTOCOL_VERSION,
};

use crate::alphabet::{
    alphabet_size, custom_alphabet, bound_indices_at_len, bounds_are_valid, bounds_diverge_immediately, candidate_to_index, lookup_predefined_alphabet, max_supported_len,
    pattern_spans, split_end, split_pos, PREDEFINED_ALPHABETS,
};
use crate::auth::{AdminAuth, AuthedUser};
use crate::client_release;
use crate::error::AppError;
use crate::models::{parse_hash_hex, u32_to_i64, Target, TargetProgress, User};
use crate::ranges::{self, SegmentOwner};
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
    let backend = req.backend.trim();
    let client_release = req.client_release.as_deref().map(str::trim).filter(|r| !r.is_empty());
    let releases = client_release::load_client_releases(&state.pool).await?;
    client_release::check_minimum(&releases, client_release)?;

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
        sqlx::query("UPDATE users SET last_seen_at = ?, protocol_version = ?, backend = ?, client_release = ? WHERE id = ?")
            .bind(now)
            .bind(&client_version)
            .bind(backend)
            .bind(client_release)
            .bind(existing.id)
            .execute(&state.pool)
            .await?;
        return Ok(Json(RegisterResponse {
            user_id: existing.id,
            token: existing.token,
            server_protocol_version: PROTOCOL_VERSION.to_string(),
        }));
    }

    let token = generate_token();
    let user_id: i64 = sqlx::query_scalar(
        "INSERT INTO users (username, hostname, token, created_at, last_seen_at, protocol_version, backend, client_release) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(username)
    .bind(hostname)
    .bind(&token)
    .bind(now)
    .bind(now)
    .bind(&client_version)
    .bind(backend)
    .bind(client_release)
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(RegisterResponse { user_id, token, server_protocol_version: PROTOCOL_VERSION.to_string() }))
}

pub async fn claim(
    State(state): State<AppState>,
    AuthedUser(user): AuthedUser,
) -> Result<Response, AppError> {
    // Checked here too, not only at /register: a client that registered
    // before the minimum was raised is turned away once it finishes the
    // range it has, rather than carrying on for as long as it runs.
    let releases = client_release::load_client_releases(&state.pool).await?;
    client_release::check_minimum(&releases, user.client_release.as_deref())?;
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
    let outcome = ranges::heartbeat_range(&state.pool, &state.config, &user, range_id, req.last_hash_a_match_filename).await?;
    Ok(Json(HeartbeatResponse { lease_seconds: outcome.lease_seconds, range_released: outcome.range_released }))
}

pub async fn admin_get_client_releases(State(state): State<AppState>, _admin: AdminAuth) -> Result<Json<ClientReleases>, AppError> {
    Ok(Json(client_release::load_client_releases(&state.pool).await?))
}

/// Replaces the minimum client release - see `ClientReleases`. Left out (or
/// null), it's cleared.
pub async fn admin_set_client_releases(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Json(req): Json<ClientReleases>,
) -> Result<Json<ClientReleases>, AppError> {
    client_release::set_client_releases(&state.pool, &req).await?;
    Ok(Json(client_release::load_client_releases(&state.pool).await?))
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

/// Rejects a start_len the main sweep could never carve at under `alphabet`
/// - see `AdminCreateTargetRequest::start_len`.
fn validate_start_len(start_len: i64, alphabet: &str) -> Result<(), AppError> {
    let cap = max_supported_len(alphabet);
    if start_len < 1 || start_len > cap {
        return Err(AppError::BadRequest(format!("start_len must be between 1 and the alphabet's max supported length ({cap})")));
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
    let (alphabet_name, alphabet) = resolve_alphabet(req.alphabet_name.as_deref(), req.alphabet.as_deref())?;
    let (alphabet_name, alphabet) = (alphabet_name.as_str(), alphabet.as_str());
    validate_start_len(req.start_len, alphabet)?;

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
    if !bounds_diverge_immediately(alphabet, lower_bound, upper_bound) {
        return Err(AppError::BadRequest(
            "lower_bound and upper_bound must not share a leading character - move whatever they have in common into prefix/suffix instead".into(),
        ));
    }
    let hash_a = parse_hash_hex(&req.hash_a_hex).map_err(|_| AppError::BadRequest("invalid hash_a_hex".into()))?;
    let hash_b = parse_hash_hex(&req.hash_b_hex).map_err(|_| AppError::BadRequest("invalid hash_b_hex".into()))?;

    let mut tx = state.pool.begin().await?;
    let now = now_unix();
    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, prune_unopened_brackets, prune_whole_candidate, max_backslash_count, alphabet_name, alphabet, status, priority, description, start_len, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&req.prefix)
    .bind(&req.suffix)
    .bind(u32_to_i64(hash_a))
    .bind(u32_to_i64(hash_b))
    .bind(lower_bound)
    .bind(upper_bound)
    .bind(req.prune_symbol_runs as i64)
    .bind(req.prune_unopened_brackets as i64)
    .bind(req.prune_whole_candidate as i64)
    .bind(req.max_backslash_count)
    .bind(alphabet_name)
    .bind(alphabet)
    .bind(req.priority)
    .bind(&req.description)
    .bind(req.start_len)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;

    // Start at the operator's start_len (1 unless set) - a bound doesn't get to skip
    // short lengths just because it's itself longer (see the comment above: at length
    // 1, bound_indices_at_len simply truncates each bound down to its first
    // character, which is exactly the right constraint there too).
    let start_len = req.start_len;
    let start_index = bound_indices_at_len(alphabet, lower_bound, upper_bound, start_len).0;
    let (start_block, start_index) = split_end(alphabet, start_len, start_index);
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_block, next_index, alphabet_name, alphabet) VALUES (?, ?, ?, ?, ?, ?)")
        .bind(target_id)
        .bind(start_len)
        .bind(start_block)
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
/// A target's alphabet, as (name, characters): the predefined one named
/// `alphabet_name`, the custom one with the characters `alphabet` (see
/// `alphabet::custom_alphabet`), or `size49` if neither is given.
fn resolve_alphabet(alphabet_name: Option<&str>, alphabet: Option<&str>) -> Result<(String, String), AppError> {
    match (alphabet_name, alphabet) {
        (Some(_), Some(_)) => Err(AppError::BadRequest("give alphabet_name or alphabet, not both".into())),
        (None, Some(characters)) => custom_alphabet(characters).map_err(AppError::BadRequest),
        (name, None) => {
            let name = name.unwrap_or("size49");
            let Some(characters) = lookup_predefined_alphabet(name) else {
                let valid: Vec<&str> = PREDEFINED_ALPHABETS.iter().map(|&(name, _, _)| name).collect();
                return Err(AppError::BadRequest(format!("unknown alphabet_name '{name}' - valid names: {}", valid.join(", "))));
            };
            Ok((name.to_string(), characters.to_string()))
        }
    }
}

async fn resolve_alphabet_patch(
    pool: &sqlx::SqlitePool,
    target_id: i64,
    alphabet_name: Option<&str>,
    alphabet: Option<&str>,
) -> Result<(String, String), AppError> {
    let (alphabet_name, alphabet) = resolve_alphabet(alphabet_name, alphabet)?;
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_optional(pool).await?.ok_or(AppError::NotFound)?;

    let cap = max_supported_len(&alphabet) as usize;
    let lower_for_validation: String = target.lower_bound.chars().take(cap).collect();
    let upper_for_validation: String = target.upper_bound.chars().take(cap).collect();
    if candidate_to_index(&alphabet, &lower_for_validation).is_none() {
        return Err(AppError::BadRequest("target's lower_bound contains a character outside the new alphabet".into()));
    }
    if candidate_to_index(&alphabet, &upper_for_validation).is_none() {
        return Err(AppError::BadRequest("target's upper_bound contains a character outside the new alphabet".into()));
    }

    Ok((alphabet_name, alphabet))
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
    if req.status.is_none()
        && req.priority.is_none()
        && req.description.is_none()
        && req.alphabet_name.is_none()
        && req.alphabet.is_none()
        && req.prune_symbol_runs.is_none()
        && req.prune_unopened_brackets.is_none()
        && req.prune_whole_candidate.is_none()
        && req.max_backslash_count.is_none()
        && req.start_len.is_none()
    {
        return Err(AppError::BadRequest(
            "at least one of status, priority, description, alphabet_name, alphabet, prune_symbol_runs, prune_unopened_brackets, \
             prune_whole_candidate, max_backslash_count or start_len must be provided"
                .into(),
        ));
    }
    if req.max_backslash_count.is_some_and(|n| n < 0) {
        return Err(AppError::BadRequest("max_backslash_count must be >= 0 (0 means unlimited)".into()));
    }

    let (alphabet_name, alphabet) = if req.alphabet_name.is_some() || req.alphabet.is_some() {
        let (name, chars) = resolve_alphabet_patch(&state.pool, target_id, req.alphabet_name.as_deref(), req.alphabet.as_deref()).await?;
        (Some(name), Some(chars))
    } else {
        (None, None)
    };

    let mut tx = state.pool.begin().await?;

    // Checked up front, in the same transaction as the UPDATE below, so we
    // can tell "no such target" (404) apart from "target exists but is
    // solved, and thus immutable by design" (409) - the UPDATE's own
    // `status != 'solved'` guard can't distinguish the two on its own, since
    // both leave `rows_affected() == 0`.
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_optional(&mut *tx).await?.ok_or(AppError::NotFound)?;
    if target.status == "solved" {
        return Err(AppError::Conflict("target is already solved and can no longer be modified".into()));
    }

    // Checked against whatever the target ends up with, so a new alphabet
    // is checked against the existing start_len too, not just the other way
    // round. The cursor itself is only moved lazily, by ranges::claim_range.
    if req.start_len.is_some() || alphabet.is_some() {
        validate_start_len(req.start_len.unwrap_or(target.start_len), alphabet.as_deref().unwrap_or(&target.alphabet))?;
    }

    let result = sqlx::query(
        "UPDATE targets SET status = COALESCE(?, status), priority = COALESCE(?, priority), \
         description = COALESCE(?, description), \
         alphabet_name = COALESCE(?, alphabet_name), alphabet = COALESCE(?, alphabet), \
         prune_symbol_runs = COALESCE(?, prune_symbol_runs), prune_unopened_brackets = COALESCE(?, prune_unopened_brackets), \
         prune_whole_candidate = COALESCE(?, prune_whole_candidate), max_backslash_count = COALESCE(?, max_backslash_count), start_len = COALESCE(?, start_len) \
         WHERE id = ? AND status != 'solved'",
    )
    .bind(&req.status)
    .bind(req.priority)
    .bind(&req.description)
    .bind(&alphabet_name)
    .bind(&alphabet)
    .bind(req.prune_symbol_runs.map(i64::from))
    .bind(req.prune_unopened_brackets.map(i64::from))
    .bind(req.prune_whole_candidate.map(i64::from))
    .bind(req.max_backslash_count)
    .bind(req.start_len)
    .bind(target_id)
    .execute(&mut *tx)
    .await?;
    debug_assert!(result.rows_affected() > 0, "target existed and wasn't solved per the check above");

    // Priority and skip ranges are translated eagerly, in the same
    // transaction as the alphabet change itself - see
    // ranges::migrate_priority_ranges_to_new_alphabet's doc comment for why
    // this can't be deferred the way the target's own cursor transition is.
    // Skip ranges go first - see migrate_skip_ranges_to_new_alphabet.
    if let (Some(name), Some(chars)) = (&alphabet_name, &alphabet) {
        let now = now_unix();
        ranges::migrate_skip_ranges_to_new_alphabet(&mut tx, target_id, name, chars, now).await?;
        ranges::migrate_priority_ranges_to_new_alphabet(&mut tx, target_id, name, chars, now).await?;
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
/// `ranges::claim_priority_range_chunk`. However many separate stretches of
/// candidates `req.pattern` covers (see `alphabet::pattern_spans`), it
/// becomes one `priority_ranges` row, with one segment per stretch.
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

    let spans = pattern_spans(&target.alphabet, &req.pattern, req.length).map_err(AppError::BadRequest)?;
    let spans = ranges::clip_spans_to_bounds(&target.alphabet, &target.lower_bound, &target.upper_bound, req.length, &spans);
    if spans.is_empty() {
        return Err(AppError::BadRequest(format!(
            "'{}' at length {} matches nothing within the target's bounds ('{}' to '{}')",
            req.pattern, req.length, target.lower_bound, target.upper_bound
        )));
    }

    // A priority range only makes sense ahead of where the target's own
    // cursor already reached. Everything before that has already been fully
    // searched by the main sweep, under its own bookkeeping; there's
    // nothing left here to fast-track.
    let progress = sqlx::query_as::<_, TargetProgress>("SELECT * FROM target_progress WHERE target_id = ?")
        .bind(target_id)
        .fetch_one(&mut *tx)
        .await?;
    if req.length < progress.candidate_len {
        return Err(AppError::BadRequest(format!(
            "the main sweep has already passed every {}-character candidate - nothing left to prioritize there \
             (to get candidates a skip range left out searched, remove the skip range instead)",
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

    let (start_index, end_index) = (spans[0].0, spans[spans.len() - 1].1);
    let next_index = if clamp_to_cursor { start_index.max(progress.next()) } else { start_index };
    if !spans.iter().any(|&(_, end)| end > next_index) {
        return Err(AppError::BadRequest(format!(
            "'{}' at length {} has already been passed by the main sweep - nothing left to prioritize \
             (to get candidates a skip range left out searched, remove the skip range instead)",
            req.pattern, req.length
        )));
    }

    // Each priority range permanently owns its segments (see
    // ranges::priority_spans_at), so this one mustn't double-book another's
    // - beyond starting after work a finished one already handed out, see
    // ranges::priority_range_start. One frozen under a different alphabet
    // isn't index-comparable at all (same reasoning as priority_spans_at),
    // so it's excluded rather than risking a wrong comparison.
    let next_index = ranges::priority_range_start(&mut tx, target_id, req.length, &target.alphabet_name, &spans, next_index).await?;

    let (start_block, start_index) = split_pos(&target.alphabet, req.length, start_index);
    let (end_block, end_index) = split_end(&target.alphabet, req.length, end_index);
    let (next_block, next_index) = split_end(&target.alphabet, req.length, next_index);
    let priority_range_id: i64 = sqlx::query_scalar(
        "INSERT INTO priority_ranges (target_id, priority, pattern, candidate_len, start_block, start_index, end_block, end_index, \
         next_block, next_index, alphabet_name, alphabet, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(target_id)
    .bind(req.priority)
    .bind(&req.pattern)
    .bind(req.length)
    .bind(start_block)
    .bind(start_index)
    .bind(end_block)
    .bind(end_index)
    .bind(next_block)
    .bind(next_index)
    .bind(&target.alphabet_name)
    .bind(&target.alphabet)
    .bind(now_unix())
    .fetch_one(&mut *tx)
    .await?;
    ranges::insert_segments(&mut tx, SegmentOwner::Priority, priority_range_id, &target.alphabet, req.length, &spans).await?;

    tx.commit().await?;
    Ok(Json(AdminCreatePriorityRangeResponse { priority_range_id }))
}

pub async fn admin_delete_priority_range(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(priority_range_id): Path<i64>,
) -> Result<Json<AdminDeletePriorityRangeResponse>, AppError> {
    let removal = ranges::remove_priority_range(&state.pool, priority_range_id).await?.ok_or(AppError::NotFound)?;
    Ok(Json(AdminDeletePriorityRangeResponse {
        deleted: removal.deleted,
        returned_to_main_sweep: removal.returned_to_main_sweep,
        kept: removal.kept,
    }))
}

/// Leaves the candidates matching `req.pattern` at `req.length` out of the
/// search - see `ranges::create_skip_range`.
pub async fn admin_create_skip_range(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(target_id): Path<i64>,
    Json(req): Json<AdminCreateSkipRangeRequest>,
) -> Result<Json<AdminCreateSkipRangeResponse>, AppError> {
    let mut tx = state.pool.begin().await?;
    let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?")
        .bind(target_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(AppError::NotFound)?;
    let skip_range_id = ranges::create_skip_range(&mut tx, &target, &req.pattern, req.length, &req.reason, now_unix()).await?;
    tx.commit().await?;
    Ok(Json(AdminCreateSkipRangeResponse { skip_range_id }))
}

pub async fn admin_delete_skip_range(
    State(state): State<AppState>,
    _admin: AdminAuth,
    Path(skip_range_id): Path<i64>,
) -> Result<Json<AdminDeleteSkipRangeResponse>, AppError> {
    let requeued = ranges::remove_skip_range(&state.pool, skip_range_id).await?.ok_or(AppError::NotFound)?;
    Ok(Json(AdminDeleteSkipRangeResponse { requeued }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register_request(protocol_version: &str) -> RegisterRequest {
        RegisterRequest { username: "u".into(), hostname: "h".into(), backend: "cuda".into(), protocol_version: protocol_version.to_string(), client_release: None }
    }

    #[test]
    fn resolve_alphabet_takes_a_predefined_or_a_custom_one_but_not_both() {
        let (name, chars) = resolve_alphabet(None, None).unwrap();
        assert_eq!((name.as_str(), chars.as_str()), ("size49", lookup_predefined_alphabet("size49").unwrap()));
        let (name, chars) = resolve_alphabet(Some("size42"), None).unwrap();
        assert_eq!((name.as_str(), chars.as_str()), ("size42", lookup_predefined_alphabet("size42").unwrap()));
        let (name, chars) = resolve_alphabet(None, Some("ZYX")).unwrap();
        assert!(name.starts_with("custom-3-"), "{name}");
        assert_eq!(chars, "XYZ");
        assert!(matches!(resolve_alphabet(Some("size49"), Some("ABC")), Err(AppError::BadRequest(_))));
        assert!(matches!(resolve_alphabet(Some("size7"), None), Err(AppError::BadRequest(_))));
        assert!(matches!(resolve_alphabet(None, Some("abc")), Err(AppError::BadRequest(_))));
    }

    #[test]
    fn validate_start_len_accepts_exactly_1_through_the_alphabets_max_len() {
        let alphabet = lookup_predefined_alphabet("size49").unwrap();
        let cap = max_supported_len(alphabet);
        assert!(validate_start_len(1, alphabet).is_ok());
        assert!(validate_start_len(cap, alphabet).is_ok());
        assert!(matches!(validate_start_len(0, alphabet), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_start_len(cap + 1, alphabet), Err(AppError::BadRequest(_))));
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
