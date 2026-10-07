use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use namebreak_protocol::{
    AdminCreatePriorityRangeRequest, AdminCreatePriorityRangeResponse, AdminCreateSkipRangeRequest, AdminCreateSkipRangeResponse, AdminCreateTargetRequest,
    AdminCreateTargetResponse, AdminDeletePriorityRangeResponse, AdminDeleteSkipRangeResponse, AdminPatchTargetRequest,
    AlphabetInfo, ClientReleases, AlphabetsResponse, CompleteRequest, DictionarySettings, HeartbeatRequest, HeartbeatResponse, QuitRequest, RegisterRequest,
    RegisterResponse, StatusResponse, TargetStatus, Version, WordListInfo, PROTOCOL_VERSION,
};

use crate::alphabet::{
    alphabet_size, custom_alphabet, bound_indices_at_len, bounds_are_valid, bounds_diverge_immediately, candidate_to_index, lookup_predefined_alphabet, max_supported_len,
    pattern_spans, split_end, split_pos, MAX_CANDIDATE_LEN, PREDEFINED_ALPHABETS,
};
use crate::auth::{AdminAuth, AuthedUser};
use crate::client_release;
use crate::dictionary::{self, FilenameBounds};
use crate::error::AppError;
use crate::models::{parse_hash_hex, u32_to_i64, Target, TargetProgress, User, DICTIONARY_ALPHABET_NAME};
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
    ranges::record_basenames(&state.pool, &user, range_id, &req.basenames).await?;
    let outcome = ranges::heartbeat_range(&state.pool, &state.config, &user, range_id, req.last_hash_a_match_filename, req.next_candidate_number).await?;
    Ok(Json(HeartbeatResponse { lease_seconds: outcome.lease_seconds, range_released: outcome.range_released }))
}

pub async fn quit(
    State(state): State<AppState>,
    AuthedUser(user): AuthedUser,
    Path(range_id): Path<i64>,
    Json(req): Json<QuitRequest>,
) -> Result<StatusCode, AppError> {
    ranges::record_basenames(&state.pool, &user, range_id, &req.basenames).await?;
    ranges::quit_range(&state.pool, &user, range_id, req.last_hash_a_match_filename.as_deref(), req.next_candidate_number).await?;
    Ok(StatusCode::NO_CONTENT)
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
    ranges::record_basenames(&state.pool, &user, range_id, &req.basenames).await?;
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
        // The name found is a real filename too - see likely_prefixes.rs.
        // In the background, so the client isn't kept waiting for it.
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = state.config.likely_prefixes.refresh_matches(&state.pool).await {
                tracing::error!(%err, "failed to count the found names into the likely prefixes");
            }
        });
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn status(State(state): State<AppState>) -> Result<Json<StatusResponse>, AppError> {
    let rows: Vec<(i64, String, String, Option<String>)> =
        sqlx::query_as("SELECT id, name, status, found_filename FROM targets WHERE is_virtual = 0 ORDER BY created_at ASC")
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

/// Rejects backslash counts a client couldn't search with - see
/// `AdminCreateTargetRequest::max_backslash_count` and `min_backslash_count`.
fn validate_backslash_counts(max_backslash_count: i64, min_backslash_count: i64) -> Result<(), AppError> {
    if max_backslash_count < 0 {
        return Err(AppError::BadRequest("max_backslash_count must be >= 0 (0 means unlimited)".into()));
    }
    if min_backslash_count < 0 {
        return Err(AppError::BadRequest("min_backslash_count must be >= 0 (0 means none needed)".into()));
    }
    if max_backslash_count != 0 && min_backslash_count > max_backslash_count {
        return Err(AppError::BadRequest("min_backslash_count must not be more than max_backslash_count (unless that's 0, unlimited)".into()));
    }
    Ok(())
}

/// The longest text an insertion may have - the client's `kMaxInsertLen`
/// (client/src/engine/limits.h). Keep the two in sync.
const MAX_INSERT_LEN: usize = 16;

/// Rejects text inserted into candidates (see
/// `AdminCreateTargetRequest::insert_from_start`) a client couldn't search
/// with: 1 to MAX_INSERT_LEN printable ASCII characters, at a position from
/// 0 to MAX_CANDIDATE_LEN - and, together with the prefix and suffix, room
/// in the client's buffers (client/src/engine/limits.h: 64 characters for
/// the prefix extended by the candidate's leading characters, 64 for the
/// suffix, 128 for a whole filename).
fn validate_insertions(prefix: &str, suffix: &str, from_start: &Option<(String, i64)>, from_end: &Option<(String, i64)>) -> Result<(), AppError> {
    for (name, insertion) in [("insert_from_start", from_start), ("insert_from_end", from_end)] {
        if let Some((text, position)) = insertion {
            if text.is_empty() || text.len() > MAX_INSERT_LEN || !text.bytes().all(|b| (0x20..=0x7E).contains(&b)) {
                return Err(AppError::BadRequest(format!("{name} must insert 1 to {MAX_INSERT_LEN} printable ASCII characters")));
            }
            if !(0..=MAX_CANDIDATE_LEN).contains(position) {
                return Err(AppError::BadRequest(format!("{name}'s position must be between 0 and {MAX_CANDIDATE_LEN}")));
            }
        }
    }
    let inserted = [from_start, from_end].iter().filter_map(|i| i.as_ref()).map(|(text, _)| text.len()).sum::<usize>();
    let max_len = MAX_CANDIDATE_LEN as usize;
    if inserted > 0 && (prefix.len() + max_len + inserted >= 64 || suffix.len() + inserted >= 64 || prefix.len() + suffix.len() + max_len + inserted >= 128) {
        return Err(AppError::BadRequest("the prefix or suffix is too long for a client to fit the inserted text too".into()));
    }
    Ok(())
}

/// Parses an `encryption_key_hex` - see `AdminCreateTargetRequest::encryption_key_hex`.
fn parse_encryption_key(hex: &str) -> Result<u32, AppError> {
    parse_hash_hex(hex).map_err(|_| AppError::BadRequest("invalid encryption_key_hex".into()))
}

/// Rejects a `base_file_name` that can't be a file's name without its
/// directory - see `AdminCreateTargetRequest::base_file_name`.
fn validate_base_file_name(name: &str) -> Result<(), AppError> {
    if name.trim().is_empty() {
        return Err(AppError::BadRequest("base_file_name can't be blank".into()));
    }
    if name.contains(['\\', '/']) {
        return Err(AppError::BadRequest("base_file_name is a name without its directory, so it can't have '\\' or '/'".into()));
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
    if let Some(settings) = &req.dictionary {
        return create_dictionary_target(&state, &req, settings).await;
    }
    if req.send_basenames {
        return Err(AppError::BadRequest("send_basenames is for dictionary targets - an alphabet search doesn't compare basenames".into()));
    }
    let encryption_key = req.encryption_key_hex.as_deref().map(parse_encryption_key).transpose()?;
    if let Some(name) = &req.base_file_name {
        validate_base_file_name(name)?;
    }
    validate_backslash_counts(req.max_backslash_count, req.min_backslash_count)?;
    validate_insertions(&req.prefix, &req.suffix, &req.insert_from_start, &req.insert_from_end)?;
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
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, prune_symbol_runs, prune_unopened_brackets, prune_whole_candidate, max_backslash_count, min_backslash_count, prune_adjacent_backslashes, insert_from_start_text, insert_from_start_position, insert_from_end_text, insert_from_end_position, alphabet_name, alphabet, status, priority, description, start_len, auto_priority, encryption_key, base_file_name, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?) RETURNING id",
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
    .bind(req.min_backslash_count)
    .bind(req.prune_adjacent_backslashes as i64)
    .bind(req.insert_from_start.as_ref().map(|(text, _)| text))
    .bind(req.insert_from_start.as_ref().map_or(0, |&(_, position)| position))
    .bind(req.insert_from_end.as_ref().map(|(text, _)| text))
    .bind(req.insert_from_end.as_ref().map_or(0, |&(_, position)| position))
    .bind(alphabet_name)
    .bind(alphabet)
    .bind(req.priority)
    .bind(&req.description)
    .bind(req.start_len)
    .bind(req.auto_priority as i64)
    .bind(encryption_key.map(u32_to_i64))
    .bind(&req.base_file_name)
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

/// The settings of `req` that only an alphabet target has, by name, if
/// given - see `AdminCreateTargetRequest::dictionary`.
fn alphabet_settings_given(req: &AdminCreateTargetRequest) -> Vec<&'static str> {
    [
        ("alphabet_name", req.alphabet_name.is_some()),
        ("alphabet", req.alphabet.is_some()),
        ("prune_symbol_runs", req.prune_symbol_runs),
        ("prune_unopened_brackets", req.prune_unopened_brackets),
        ("prune_whole_candidate", req.prune_whole_candidate),
        ("max_backslash_count", req.max_backslash_count != 0),
        ("min_backslash_count", req.min_backslash_count != 0),
        ("prune_adjacent_backslashes", req.prune_adjacent_backslashes),
        ("insert_from_start", req.insert_from_start.is_some()),
        ("insert_from_end", req.insert_from_end.is_some()),
        ("start_len", req.start_len != 1),
        ("auto_priority", req.auto_priority),
    ]
    .into_iter()
    .filter_map(|(name, given)| given.then_some(name))
    .collect()
}

/// Rejects text a dictionary target's filenames can't have: anything but
/// printable ASCII, which is all a word list's words can be made of - or
/// more than 255 characters.
fn validate_dictionary_text(what: &str, text: &str) -> Result<(), AppError> {
    if !dictionary::is_printable_ascii(text) {
        return Err(AppError::BadRequest(format!("{what} can only have printable ASCII characters (' ' to '~')")));
    }
    if text.len() > 255 {
        return Err(AppError::BadRequest(format!("{what} can't be longer than 255 characters")));
    }
    Ok(())
}

/// `admin_create_target` for a dictionary target (`req.dictionary` - see
/// `AdminCreateTargetRequest::dictionary`): its prefix, suffix, separators
/// and bounds are normalized as the client normalizes them, its word lists
/// read, and its candidates numbered (`dictionary::load`), so that a target
/// whose candidates can't be numbered, or whose bounds leave none, is
/// refused here rather than handing out nothing. Its cursor starts at the
/// first window.
async fn create_dictionary_target(state: &AppState, req: &AdminCreateTargetRequest, settings: &DictionarySettings) -> Result<Json<AdminCreateTargetResponse>, AppError> {
    let alphabet_settings = alphabet_settings_given(req);
    if !alphabet_settings.is_empty() {
        return Err(AppError::BadRequest(format!("{} isn't for a dictionary target", alphabet_settings.join(", "))));
    }
    let encryption_key = req.encryption_key_hex.as_deref().map(parse_encryption_key).transpose()?;
    if let Some(name) = &req.base_file_name {
        validate_base_file_name(name)?;
    }
    if req.send_basenames && encryption_key.is_none() {
        return Err(AppError::BadRequest("send_basenames needs an encryption_key_hex to compare the basenames to".into()));
    }
    let hash_a = parse_hash_hex(&req.hash_a_hex).map_err(|_| AppError::BadRequest("invalid hash_a_hex".into()))?;
    let hash_b = parse_hash_hex(&req.hash_b_hex).map_err(|_| AppError::BadRequest("invalid hash_b_hex".into()))?;
    for (what, text) in [("prefix", &req.prefix), ("suffix", &req.suffix), ("lower_bound", &req.lower_bound), ("upper_bound", &req.upper_bound)] {
        validate_dictionary_text(what, text)?;
    }
    for separator in &settings.separators {
        validate_dictionary_text("a separator", separator)?;
    }
    let prefix = dictionary::normalize(&req.prefix);
    let suffix = dictionary::normalize(&req.suffix);
    let separators: Vec<String> = settings.separators.iter().map(|s| dictionary::normalize(s)).collect();
    let bound = |b: &str| (!b.is_empty()).then(|| dictionary::normalize(b));
    let bounds = FilenameBounds { lower: bound(&req.lower_bound), upper: bound(&req.upper_bound) };
    if let (Some(lower), Some(upper)) = (&bounds.lower, &bounds.upper) {
        if lower > upper {
            return Err(AppError::BadRequest(format!("lower_bound '{lower}' sorts after upper_bound '{upper}'")));
        }
    }

    let mut tx = state.pool.begin().await?;
    let loaded = dictionary::load(&mut tx, &settings.word_lists, &separators, settings.min_words, settings.max_words, &prefix, &suffix, bounds.clone()).await?;
    let first = loaded.windows[0];
    let now = now_unix();
    let as_json = |list: &[String]| serde_json::to_string(list).expect("a list of strings serializes");
    let target_id: i64 = sqlx::query_scalar(
        "INSERT INTO targets (name, prefix, suffix, hash_a, hash_b, lower_bound, upper_bound, alphabet_name, alphabet, status, priority, description, \
         start_len, encryption_key, base_file_name, kind, word_lists, separators, min_words, max_words, send_basenames, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', 'active', ?, ?, 1, ?, ?, 'dictionary', ?, ?, ?, ?, ?, ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&prefix)
    .bind(&suffix)
    .bind(u32_to_i64(hash_a))
    .bind(u32_to_i64(hash_b))
    .bind(bounds.lower.as_deref().unwrap_or(""))
    .bind(bounds.upper.as_deref().unwrap_or(""))
    .bind(DICTIONARY_ALPHABET_NAME)
    .bind(req.priority)
    .bind(&req.description)
    .bind(encryption_key.map(u32_to_i64))
    .bind(&req.base_file_name)
    .bind(as_json(&settings.word_lists))
    .bind(as_json(&separators))
    .bind(settings.min_words)
    .bind(settings.max_words)
    .bind(req.send_basenames as i64)
    .bind(now)
    .fetch_one(&mut *tx)
    .await?;
    sqlx::query("INSERT INTO target_progress (target_id, candidate_len, next_block, next_index, alphabet_name, alphabet) VALUES (?, ?, 0, ?, ?, '')")
        .bind(target_id)
        .bind(first.words)
        .bind(first.start)
        .bind(DICTIONARY_ALPHABET_NAME)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    tracing::info!(target_id, candidates = loaded.window_candidates(), words = loaded.space.words().len(), "created a dictionary target");
    Ok(Json(AdminCreateTargetResponse { target_id }))
}

/// Rejects a name a word list can't have: 1 to 64 letters, digits, '.',
/// '-' and '_', starting with a letter or digit - safe as a client's file
/// name, as it's cached under it.
fn validate_word_list_name(name: &str) -> Result<(), AppError> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-' || b == b'_')
        && name.as_bytes()[0].is_ascii_alphanumeric();
    if !valid {
        return Err(AppError::BadRequest("a word list's name is 1 to 64 letters, digits, '.', '-' and '_', starting with a letter or digit".into()));
    }
    Ok(())
}

/// Stores a word list under `name`, the request's body as it is: one word
/// per line, read as a client reads one (see `dictionary::parse_word_list`).
/// A list never changes once stored - a client caches it by its name - so
/// the same name again is only accepted with the very same file (and then
/// changes nothing); a different list needs a different name. The
/// `english-` names are the dictionaries built into the server and the
/// client (`english-1` - see `dictionary::store_built_in`), and can't be
/// uploaded.
pub async fn admin_put_word_list(State(state): State<AppState>, _admin: AdminAuth, Path(name): Path<String>, body: Bytes) -> Result<Json<WordListInfo>, AppError> {
    validate_word_list_name(&name)?;
    if dictionary::is_built_in_name(&name) {
        return Err(AppError::BadRequest(format!(
            "the english- names are the dictionaries built into the server and the client - {} is there already, so give this list another name",
            dictionary::ENGLISH_1
        )));
    }
    let (words, skipped_lines) = dictionary::parse_word_list(&body);
    let words = dictionary::sorted_unique(words);
    if words.is_empty() {
        return Err(AppError::BadRequest("the word list has no words".into()));
    }
    let checksum = dictionary::checksum(&words);

    let mut tx = state.pool.begin().await?;
    let existing: Option<(Vec<u8>, i64)> = sqlx::query_as("SELECT content, created_at FROM word_lists WHERE name = ?").bind(&name).fetch_optional(&mut *tx).await?;
    let created_at = match existing {
        Some((content, _)) if content != body.as_ref() => {
            return Err(AppError::Conflict(format!("there's already a word list named '{name}', with other contents - a word list never changes, so give this one another name")));
        }
        Some((_, created_at)) => created_at,
        None => {
            let now = now_unix();
            sqlx::query("INSERT INTO word_lists (name, content, word_count, checksum, created_at) VALUES (?, ?, ?, ?, ?)")
                .bind(&name)
                .bind(body.as_ref())
                .bind(words.len() as i64)
                .bind(dictionary::hex64(checksum))
                .bind(now)
                .execute(&mut *tx)
                .await?;
            now
        }
    };
    tx.commit().await?;
    Ok(Json(WordListInfo { name, word_count: words.len() as i64, checksum: dictionary::hex64(checksum), created_at, skipped_lines }))
}

/// Every stored word list, by name.
pub async fn admin_list_word_lists(State(state): State<AppState>, _admin: AdminAuth) -> Result<Json<Vec<WordListInfo>>, AppError> {
    let rows: Vec<(String, i64, String, i64)> =
        sqlx::query_as("SELECT name, word_count, checksum, created_at FROM word_lists ORDER BY name").fetch_all(&state.pool).await?;
    Ok(Json(
        rows.into_iter()
            .map(|(name, word_count, checksum, created_at)| WordListInfo { name, word_count, checksum, created_at, skipped_lines: Vec::new() })
            .collect(),
    ))
}

/// The dictionary targets (by id) whose candidates are made of word list `name`.
async fn targets_using_word_list(conn: &mut sqlx::SqliteConnection, name: &str) -> Result<Vec<i64>, AppError> {
    let rows: Vec<(i64, Option<String>)> = sqlx::query_as("SELECT id, word_lists FROM targets WHERE kind = 'dictionary'").fetch_all(&mut *conn).await?;
    let mut using = Vec::new();
    for (id, lists) in rows {
        if dictionary::parse_string_list(lists.as_deref())?.iter().any(|list| list == name) {
            using.push(id);
        }
    }
    Ok(using)
}

/// Deletes a word list no target uses - but not one built in.
pub async fn admin_delete_word_list(State(state): State<AppState>, _admin: AdminAuth, Path(name): Path<String>) -> Result<StatusCode, AppError> {
    if dictionary::is_built_in_name(&name) {
        return Err(AppError::BadRequest(format!("{name} is built into the server - it can't be deleted")));
    }
    let mut tx = state.pool.begin().await?;
    let using = targets_using_word_list(&mut tx, &name).await?;
    if !using.is_empty() {
        let ids: Vec<String> = using.iter().map(i64::to_string).collect();
        return Err(AppError::Conflict(format!("the word list '{name}' is used by target {} - delete those first", ids.join(", "))));
    }
    let deleted = sqlx::query("DELETE FROM word_lists WHERE name = ?").bind(&name).execute(&mut *tx).await?.rows_affected();
    tx.commit().await?;
    if deleted == 0 {
        return Err(AppError::NotFound);
    }
    Ok(StatusCode::NO_CONTENT)
}

/// A stored word list, exactly as uploaded - for a client to search a
/// dictionary target with (see `ClaimResponse::word_lists`).
pub async fn word_list(State(state): State<AppState>, AuthedUser(_user): AuthedUser, Path(name): Path<String>) -> Result<Response, AppError> {
    let content: Vec<u8> = sqlx::query_scalar("SELECT content FROM word_lists WHERE name = ?").bind(&name).fetch_optional(&state.pool).await?.ok_or(AppError::NotFound)?;
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], content).into_response())
}

/// How many basenames `target_basenames` reads at a time.
pub(crate) const BASENAMES_PAGE: i64 = 10_000;

/// Every basename clients found matching a target's encryption key (see
/// `AdminCreateTargetRequest::send_basenames`), one per line, in the order
/// they were first reported. Public, like the dashboard, which shows only
/// the latest of them and links here for the rest. There can be a million
/// of them (three words of english-1), so they're read and sent a page of
/// `BASENAMES_PAGE` at a time: the database - one connection, which every
/// claim and heartbeat needs too - is only held for a page's query, and
/// only a page is ever in memory, however slowly the list is downloaded.
pub async fn target_basenames(State(state): State<AppState>, Path(target_id): Path<i64>) -> Result<Response, AppError> {
    let exists: Option<i64> = sqlx::query_scalar("SELECT id FROM targets WHERE id = ? AND is_virtual = 0").bind(target_id).fetch_optional(&state.pool).await?;
    exists.ok_or(AppError::NotFound)?;
    let pool = state.pool.clone();
    // The pages, each after the last row of the one before - None once
    // there's none left.
    let pages = futures_util::stream::unfold(Some(0i64), move |after| {
        let pool = pool.clone();
        async move {
            let after = after?;
            let rows: Result<Vec<(i64, String)>, sqlx::Error> =
                sqlx::query_as("SELECT id, basename FROM basenames WHERE target_id = ? AND id > ? ORDER BY id LIMIT ?")
                    .bind(target_id)
                    .bind(after)
                    .bind(BASENAMES_PAGE)
                    .fetch_all(&pool)
                    .await;
            match rows {
                Ok(rows) if rows.is_empty() => None,
                Ok(rows) => {
                    let next = (rows.len() as i64 == BASENAMES_PAGE).then(|| rows.last().expect("not empty").0);
                    let text: String = rows.into_iter().map(|(_, basename)| basename + "\n").collect();
                    Some((Ok::<_, std::io::Error>(text), next))
                }
                Err(err) => Some((Err(std::io::Error::other(err)), None)),
            }
        }
    });
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], axum::body::Body::from_stream(pages)).into_response())
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
    if req.name.as_deref().is_some_and(|name| name.trim().is_empty()) {
        return Err(AppError::BadRequest("name can't be blank".into()));
    }
    if let Some(status) = &req.status {
        if status != "active" && status != "paused" {
            return Err(AppError::BadRequest("status must be 'active' or 'paused'".into()));
        }
    }
    if req.name.is_none()
        && req.status.is_none()
        && req.priority.is_none()
        && req.description.is_none()
        && req.alphabet_name.is_none()
        && req.alphabet.is_none()
        && req.prune_symbol_runs.is_none()
        && req.prune_unopened_brackets.is_none()
        && req.prune_whole_candidate.is_none()
        && req.max_backslash_count.is_none()
        && req.min_backslash_count.is_none()
        && req.prune_adjacent_backslashes.is_none()
        && req.insert_from_start.is_none()
        && req.insert_from_end.is_none()
        && req.start_len.is_none()
        && req.auto_priority.is_none()
        && req.encryption_key_hex.is_none()
        && req.base_file_name.is_none()
        && req.send_basenames.is_none()
    {
        return Err(AppError::BadRequest(
            "at least one of name, status, priority, description, alphabet_name, alphabet, prune_symbol_runs, prune_unopened_brackets, \
             prune_whole_candidate, max_backslash_count, min_backslash_count, prune_adjacent_backslashes, insert_from_start, \
             insert_from_end, start_len, auto_priority, encryption_key_hex, base_file_name or send_basenames must be provided"
                .into(),
        ));
    }
    // Some(None): remove it (`null`); None: leave it as it is.
    let encryption_key = req.encryption_key_hex.as_ref().map(|hex| hex.as_deref().map(parse_encryption_key).transpose()).transpose()?;
    if let Some(Some(name)) = &req.base_file_name {
        validate_base_file_name(name)?;
    }

    // A dictionary target has none of an alphabet target's search settings,
    // and only it compares basenames - see AdminCreateTargetRequest::dictionary.
    let existing = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_optional(&state.pool).await?.ok_or(AppError::NotFound)?;
    if existing.is_dictionary() {
        let alphabet_settings: Vec<&str> = [
            ("alphabet_name", req.alphabet_name.is_some()),
            ("alphabet", req.alphabet.is_some()),
            ("prune_symbol_runs", req.prune_symbol_runs.is_some()),
            ("prune_unopened_brackets", req.prune_unopened_brackets.is_some()),
            ("prune_whole_candidate", req.prune_whole_candidate.is_some()),
            ("max_backslash_count", req.max_backslash_count.is_some()),
            ("min_backslash_count", req.min_backslash_count.is_some()),
            ("prune_adjacent_backslashes", req.prune_adjacent_backslashes.is_some()),
            ("insert_from_start", req.insert_from_start.is_some()),
            ("insert_from_end", req.insert_from_end.is_some()),
            ("start_len", req.start_len.is_some()),
            ("auto_priority", req.auto_priority.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, given)| given.then_some(name))
        .collect();
        if !alphabet_settings.is_empty() {
            return Err(AppError::BadRequest(format!("{} isn't for a dictionary target", alphabet_settings.join(", "))));
        }
        // What the target ends up with.
        let has_key = match encryption_key {
            Some(key) => key.is_some(),
            None => existing.encryption_key.is_some(),
        };
        if req.send_basenames.unwrap_or(existing.send_basenames != 0) && !has_key {
            return Err(AppError::BadRequest("send_basenames needs an encryption_key_hex to compare the basenames to".into()));
        }
    } else if req.send_basenames == Some(true) {
        return Err(AppError::BadRequest("send_basenames is for dictionary targets - an alphabet search doesn't compare basenames".into()));
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

    // Like start_len below, against what the target ends up with: a new
    // max_backslash_count against the existing min_backslash_count too.
    if req.max_backslash_count.is_some() || req.min_backslash_count.is_some() {
        validate_backslash_counts(
            req.max_backslash_count.unwrap_or(target.max_backslash_count),
            req.min_backslash_count.unwrap_or(target.min_backslash_count),
        )?;
    }

    // The insertions the target ends up with, checked together.
    let stored = |text: &Option<String>, position: i64| text.clone().map(|text| (text, position));
    let insert_from_start = req.insert_from_start.clone().unwrap_or_else(|| stored(&target.insert_from_start_text, target.insert_from_start_position));
    let insert_from_end = req.insert_from_end.clone().unwrap_or_else(|| stored(&target.insert_from_end_text, target.insert_from_end_position));
    validate_insertions(&target.prefix, &target.suffix, &insert_from_start, &insert_from_end)?;

    // Checked against whatever the target ends up with, so a new alphabet
    // is checked against the existing start_len too, not just the other way
    // round. The cursor itself is only moved lazily, by ranges::claim_range.
    if req.start_len.is_some() || alphabet.is_some() {
        validate_start_len(req.start_len.unwrap_or(target.start_len), alphabet.as_deref().unwrap_or(&target.alphabet))?;
    }

    let result = sqlx::query(
        "UPDATE targets SET name = COALESCE(?, name), status = COALESCE(?, status), priority = COALESCE(?, priority), \
         description = COALESCE(?, description), \
         alphabet_name = COALESCE(?, alphabet_name), alphabet = COALESCE(?, alphabet), \
         prune_symbol_runs = COALESCE(?, prune_symbol_runs), prune_unopened_brackets = COALESCE(?, prune_unopened_brackets), \
         prune_whole_candidate = COALESCE(?, prune_whole_candidate), max_backslash_count = COALESCE(?, max_backslash_count), \
         min_backslash_count = COALESCE(?, min_backslash_count), prune_adjacent_backslashes = COALESCE(?, prune_adjacent_backslashes), \
         start_len = COALESCE(?, start_len), auto_priority = COALESCE(?, auto_priority), send_basenames = COALESCE(?, send_basenames) \
         WHERE id = ? AND status != 'solved'",
    )
    .bind(&req.name)
    .bind(&req.status)
    .bind(req.priority)
    .bind(&req.description)
    .bind(&alphabet_name)
    .bind(&alphabet)
    .bind(req.prune_symbol_runs.map(i64::from))
    .bind(req.prune_unopened_brackets.map(i64::from))
    .bind(req.prune_whole_candidate.map(i64::from))
    .bind(req.max_backslash_count)
    .bind(req.min_backslash_count)
    .bind(req.prune_adjacent_backslashes.map(i64::from))
    .bind(req.start_len)
    .bind(req.auto_priority.map(i64::from))
    .bind(req.send_basenames.map(i64::from))
    .bind(target_id)
    .execute(&mut *tx)
    .await?;
    debug_assert!(result.rows_affected() > 0, "target existed and wasn't solved per the check above");
    // Set outright rather than COALESCEd above, since `null` removes one.
    if req.insert_from_start.is_some() || req.insert_from_end.is_some() {
        sqlx::query(
            "UPDATE targets SET insert_from_start_text = ?, insert_from_start_position = ?, insert_from_end_text = ?, insert_from_end_position = ? \
             WHERE id = ?",
        )
        .bind(insert_from_start.as_ref().map(|(text, _)| text))
        .bind(insert_from_start.as_ref().map_or(0, |&(_, position)| position))
        .bind(insert_from_end.as_ref().map(|(text, _)| text))
        .bind(insert_from_end.as_ref().map_or(0, |&(_, position)| position))
        .bind(target_id)
        .execute(&mut *tx)
        .await?;
    }
    if let Some(encryption_key) = encryption_key {
        sqlx::query("UPDATE targets SET encryption_key = ? WHERE id = ?").bind(encryption_key.map(u32_to_i64)).bind(target_id).execute(&mut *tx).await?;
    }
    if let Some(base_file_name) = &req.base_file_name {
        sqlx::query("UPDATE targets SET base_file_name = ? WHERE id = ?").bind(base_file_name).bind(target_id).execute(&mut *tx).await?;
    }

    // Turned on again, or in an alphabet that can spell other prefixes: look
    // for likely prefixes afresh - see ranges::create_next_auto_priority_range.
    if req.auto_priority.is_some() || alphabet.is_some() {
        sqlx::query("UPDATE targets SET auto_priority_exhausted_len = NULL WHERE id = ?").bind(target_id).execute(&mut *tx).await?;
    }

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
    if target.is_dictionary() {
        return Err(AppError::BadRequest("a dictionary target has no priority ranges - its candidates aren't strings of an alphabet".into()));
    }

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
    if target.is_dictionary() {
        return Err(AppError::BadRequest("a dictionary target has no skip ranges - its candidates aren't strings of an alphabet".into()));
    }
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
    fn validate_insertions_wants_short_printable_text_at_a_position_a_candidate_can_have() {
        let some = |text: &str, position: i64| Some((text.to_string(), position));
        assert!(validate_insertions("REZ\\", ".WAV", &None, &None).is_ok());
        assert!(validate_insertions("REZ\\", ".WAV", &some("\\", 3), &some("_X", 0)).is_ok());
        assert!(validate_insertions("REZ\\", ".WAV", &some("\\", MAX_CANDIDATE_LEN), &None).is_ok());
        assert!(matches!(validate_insertions("REZ\\", ".WAV", &some("", 3), &None), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_insertions("REZ\\", ".WAV", &some("\t", 3), &None), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_insertions("REZ\\", ".WAV", &some(&"A".repeat(17), 3), &None), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_insertions("REZ\\", ".WAV", &None, &some("A", -1)), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_insertions("REZ\\", ".WAV", &None, &some("A", MAX_CANDIDATE_LEN + 1)), Err(AppError::BadRequest(_))));
        // No room left in the client's buffers for it.
        assert!(matches!(validate_insertions(&"P".repeat(40), ".WAV", &some(&"A".repeat(10), 1), &None), Err(AppError::BadRequest(_))));
        assert!(validate_insertions(&"P".repeat(40), ".WAV", &None, &None).is_ok(), "only with inserted text");
    }

    #[test]
    fn validate_backslash_counts_wants_a_min_no_more_than_a_max() {
        assert!(validate_backslash_counts(0, 0).is_ok());
        assert!(validate_backslash_counts(0, 5).is_ok(), "no max: any min");
        assert!(validate_backslash_counts(2, 2).is_ok());
        assert!(validate_backslash_counts(3, 1).is_ok());
        assert!(matches!(validate_backslash_counts(2, 3), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_backslash_counts(-1, 0), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_backslash_counts(0, -1), Err(AppError::BadRequest(_))));
    }

    #[test]
    fn validate_base_file_name_wants_a_name_without_a_directory() {
        assert!(validate_base_file_name("DF.Diablo II").is_ok());
        assert!(validate_base_file_name("patch.txt").is_ok());
        assert!(matches!(validate_base_file_name(""), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_base_file_name("  "), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_base_file_name("103c\\DF.Diablo II"), Err(AppError::BadRequest(_))));
        assert!(matches!(validate_base_file_name("rez/x.wav"), Err(AppError::BadRequest(_))));
    }

    async fn state_with_empty_database() -> AppState {
        let pool = crate::db::connect("sqlite::memory:").await.unwrap();
        let config = crate::state::RangeConfig {
            target_chunk_seconds: 1.0,
            default_rate_per_sec: 1.0,
            default_dictionary_rate_per_sec: 1.0,
            min_chunk_candidates: 1,
            max_chunk_candidates: 1,
            lease_seconds: 60,
            reclaim_interval_secs: 30,
            ema_alpha: 0.3,
            canary_probability: 0.0,
            canary_seconds: 5.0,
            stall_release_seconds: 60,
            likely_prefixes: Default::default(),
            dictionaries: Default::default(),
        };
        AppState(std::sync::Arc::new(crate::state::Inner { pool, admin_token: "t".into(), config }))
    }

    async fn create_target(state: &AppState, extra: serde_json::Value) -> Result<i64, AppError> {
        let mut body = serde_json::json!({
            "name": "t", "prefix": "REZ\\", "suffix": ".WAV",
            "hash_a_hex": "0xF60F5D90", "hash_b_hex": "0xCE0A9BDB",
            "lower_bound": "FINZ09BX", "upper_bound": "GLUCMPGN",
        });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let req = serde_json::from_value(body).unwrap();
        admin_create_target(State(state.clone()), AdminAuth, Json(req)).await.map(|response| response.target_id)
    }

    async fn patch_target(state: &AppState, target_id: i64, body: serde_json::Value) -> Result<StatusCode, AppError> {
        admin_patch_target(State(state.clone()), AdminAuth, Path(target_id), Json(serde_json::from_value(body).unwrap())).await
    }

    async fn key_and_base_file_name(state: &AppState, target_id: i64) -> (Option<i64>, Option<String>) {
        let target = sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(target_id).fetch_one(&state.pool).await.unwrap();
        (target.encryption_key, target.base_file_name)
    }

    #[tokio::test]
    async fn encryption_key_and_base_file_name_are_optional_when_creating_a_target() {
        let state = state_with_empty_database().await;
        let without = create_target(&state, serde_json::json!({})).await.unwrap();
        assert_eq!(key_and_base_file_name(&state, without).await, (None, None));

        let with = create_target(&state, serde_json::json!({"encryption_key_hex": "0xD9AC2EFF", "base_file_name": "X.WAV"})).await.unwrap();
        assert_eq!(key_and_base_file_name(&state, with).await, (Some(0xD9AC2EFF), Some("X.WAV".into())));

        assert!(matches!(create_target(&state, serde_json::json!({"encryption_key_hex": "xyz"})).await, Err(AppError::BadRequest(_))));
        assert!(matches!(create_target(&state, serde_json::json!({"base_file_name": "rez\\X.WAV"})).await, Err(AppError::BadRequest(_))));
    }

    #[tokio::test]
    async fn patching_encryption_key_and_base_file_name_sets_leaves_or_removes_them() {
        let state = state_with_empty_database().await;
        let id = create_target(&state, serde_json::json!({"encryption_key_hex": "D9AC2EFF", "base_file_name": "X.WAV"})).await.unwrap();

        patch_target(&state, id, serde_json::json!({"priority": 1})).await.unwrap();
        assert_eq!(key_and_base_file_name(&state, id).await, (Some(0xD9AC2EFF), Some("X.WAV".into())), "left out: unchanged");

        patch_target(&state, id, serde_json::json!({"encryption_key_hex": null})).await.unwrap();
        assert_eq!(key_and_base_file_name(&state, id).await, (None, Some("X.WAV".into())), "null: removed");

        patch_target(&state, id, serde_json::json!({"encryption_key_hex": "0x0000002a", "base_file_name": "Y.WAV"})).await.unwrap();
        assert_eq!(key_and_base_file_name(&state, id).await, (Some(42), Some("Y.WAV".into())));

        patch_target(&state, id, serde_json::json!({"base_file_name": null})).await.unwrap();
        assert_eq!(key_and_base_file_name(&state, id).await, (Some(42), None), "on its own, null still counts as a change");

        assert!(matches!(patch_target(&state, id, serde_json::json!({"encryption_key_hex": "0x1FFFFFFFF"})).await, Err(AppError::BadRequest(_))));
        assert!(matches!(patch_target(&state, id, serde_json::json!({"base_file_name": ""})).await, Err(AppError::BadRequest(_))));
        assert_eq!(key_and_base_file_name(&state, id).await, (Some(42), None), "rejected patches change nothing");
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
