//! Meet-in-the-middle targets end to end, through the handlers: targets made
//! and patched, their tails claimed, checkpointed, quit and completed, the
//! memory a client has to give before it gets any of them, the canaries they
//! hand out, and what the dashboard shows of it all. `meet_in_the_middle.rs`
//! tests the numbering and the table's size themselves.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use namebreak_protocol::{ClaimResponse, CompleteRequest, HeartbeatRequest, QuitRequest};

use crate::auth::{AdminAuth, AuthedUser};
use crate::error::AppError;
use crate::handlers;
use crate::meet_in_the_middle;
use crate::models::{Range, Target, User};
use crate::mpq::{continue_hash, mpq_hash, FILE_KEY_OFFSET, HASH_A_OFFSET, HASH_B_OFFSET, INITIAL_STATE};
use crate::ranges;
use crate::state::{now_unix, AppState, Inner, RangeConfig};

/// Each claim gets `chunk` tails (rate 1 a second for `chunk` seconds),
/// whatever the client's measured rate - unless a test measures one.
fn config(chunk: f64) -> RangeConfig {
    RangeConfig {
        target_chunk_seconds: chunk,
        default_rate_per_sec: 1.0,
        default_dictionary_rate_per_sec: 1.0,
        default_meet_in_the_middle_rate_per_sec: 1.0,
        min_chunk_candidates: 1,
        max_chunk_candidates: 1_000_000_000_000,
        lease_seconds: 60,
        reclaim_interval_secs: 30,
        ema_alpha: 0.5,
        canary_probability: 0.0,
        canary_seconds: 5.0,
        stall_release_seconds: 600,
        likely_prefixes: Default::default(),
        dictionaries: Default::default(),
    }
}

async fn state_with(config: RangeConfig) -> AppState {
    let pool = crate::db::connect("sqlite::memory:").await.unwrap();
    AppState(std::sync::Arc::new(Inner { pool, admin_token: "t".into(), config, disk: Default::default() }))
}

async fn state(chunk: f64) -> AppState {
    state_with(config(chunk)).await
}

/// A client of `protocol_version` that will give `table_bytes` to a
/// meet-in-the-middle search's table - `None` for one that says nothing (a
/// client older than protocol 1.6).
async fn user(state: &AppState, name: &str, protocol_version: &str, table_bytes: Option<i64>) -> User {
    let now = now_unix();
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO users (username, hostname, token, created_at, last_seen_at, protocol_version, backend, meet_in_the_middle_table_bytes) \
         VALUES (?, 'host', ?, ?, ?, ?, 'cuda', ?) RETURNING id",
    )
    .bind(name)
    .bind(format!("{name}-token"))
    .bind(now)
    .bind(now)
    .bind(protocol_version)
    .bind(table_bytes)
    .fetch_one(&state.pool)
    .await
    .unwrap();
    sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

/// A client that will give a gibibyte - enough for every table these tests
/// make.
async fn client(state: &AppState, name: &str) -> User {
    user(state, name, "1.6.0", Some(1 << 30)).await
}

const PREFIX: &str = "SETUPDAT\\";
const SUFFIX: &str = "\\BWUNIN.PIF";
/// The alphabet these tests search a directory of: three characters, so a
/// length's tails can be listed in full - in ascending order, as a custom
/// alphabet is stored (see `alphabet::custom_alphabet`).
const ALPHABET: &str = "-AB";

/// The meeting point and hashes of `prefix` + `dir` + `suffix`, as an operator
/// recovering them would give them: the state after the directory, and the
/// whole name's two hashes.
fn meet_in_the_middle_of(prefix: &str, dir: &str, suffix: &str) -> serde_json::Value {
    let directory = format!("{prefix}{dir}");
    let state = |offset| continue_hash(INITIAL_STATE, &directory, offset);
    let (a, b) = (state(HASH_A_OFFSET), state(HASH_B_OFFSET));
    let filename = format!("{directory}{suffix}");
    serde_json::json!({
        "hash_a_hex": format!("0x{:08X}", mpq_hash(&filename, HASH_A_OFFSET)),
        "hash_b_hex": format!("0x{:08X}", mpq_hash(&filename, HASH_B_OFFSET)),
        "meet_in_the_middle": {
            "meeting_point_a_seed1_hex": format!("0x{:08X}", a.0),
            "meeting_point_a_seed2_hex": format!("0x{:08X}", a.1),
            "meeting_point_b_seed1_hex": format!("0x{:08X}", b.0),
            "meeting_point_b_seed2_hex": format!("0x{:08X}", b.1),
            "min_length": 2, "max_length": 3, "table_length": 1,
        },
    })
}

/// A meet-in-the-middle target for the directory "B-A" under `PREFIX`:
/// directories of 2 to 3 characters of `ALPHABET`, the first of which the
/// table holds - so 3 tails at length 2 and 9 at length 3. `extra` is merged
/// into the request (its `meet_in_the_middle` object replaces the one above
/// wholesale).
async fn create(state: &AppState, extra: serde_json::Value) -> Result<i64, AppError> {
    let mut body = serde_json::json!({"name": "setupdat", "prefix": PREFIX, "suffix": SUFFIX, "alphabet": ALPHABET});
    for part in [meet_in_the_middle_of(PREFIX, "B-A", SUFFIX), extra] {
        body.as_object_mut().unwrap().extend(part.as_object().unwrap().clone());
    }
    let req = serde_json::from_value(body).unwrap();
    handlers::admin_create_target(State(state.clone()), AdminAuth, Json(req)).await.map(|json| json.0.target_id)
}

async fn target(state: &AppState, id: i64) -> Target {
    sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

async fn range(state: &AppState, id: i64) -> Range {
    sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

async fn claim_work(state: &AppState, user: &User) -> Option<ClaimResponse> {
    ranges::claim_range(&state.pool, &state.config, user).await.unwrap()
}

/// A claim as (directory length, first tail, end tail).
fn claimed(claim: &ClaimResponse) -> (i64, i64, i64) {
    (claim.directory_length.unwrap(), claim.first_tail_index.unwrap(), claim.end_tail_index.unwrap())
}

async fn heartbeat(state: &AppState, user: &User, range_id: i64, next_tail_index: Option<i64>) -> Result<bool, AppError> {
    let req = HeartbeatRequest { last_hash_a_match_filename: None, next_candidate_number: None, next_tail_index, basenames: Vec::new(), false_matches: Vec::new() };
    handlers::heartbeat(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await.map(|json| json.0.range_released)
}

async fn quit(state: &AppState, user: &User, range_id: i64, next_tail_index: Option<i64>) -> Result<StatusCode, AppError> {
    let req = QuitRequest { last_hash_a_match_filename: None, next_candidate_number: None, next_tail_index, basenames: Vec::new(), false_matches: Vec::new() };
    handlers::quit(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await
}

async fn complete(state: &AppState, user: &User, range_id: i64, found: Option<&str>, elapsed: f64, tails: i64) -> Result<StatusCode, AppError> {
    let req = CompleteRequest {
        found: found.is_some(),
        filename: found.map(str::to_string),
        elapsed_seconds: elapsed,
        candidates_processed: tails,
        basenames: Vec::new(),
        false_matches: Vec::new(),
    };
    handlers::complete(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await
}

#[tokio::test]
async fn a_target_is_made_from_a_meet_in_the_middle_that_fits_its_hashes() {
    let state = state(1.0).await;
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let target = target(&state, id).await;
    assert!(target.is_meet_in_the_middle());
    assert_eq!((target.start_len, target.meet_in_the_middle_max_length, target.meet_in_the_middle_table_length), (2, Some(3), Some(1)));
    assert_eq!(target.alphabet, ALPHABET);
    // The meeting point is the state after PREFIX + the directory.
    let settings = meet_in_the_middle::of_target(&target).unwrap();
    assert_eq!(settings.meeting_point_a, continue_hash(INITIAL_STATE, "SETUPDAT\\B-A", HASH_A_OFFSET));
    assert_eq!(settings.meeting_point_b, continue_hash(INITIAL_STATE, "SETUPDAT\\B-A", HASH_B_OFFSET));
    // Its cursor starts at the first tail of the shortest directory.
    let (len, index): (i64, i64) = sqlx::query_as("SELECT candidate_len, next_index FROM target_progress WHERE target_id = ?")
        .bind(id)
        .fetch_one(&state.pool)
        .await
        .unwrap();
    assert_eq!((len, index), (2, 0));

    // A meeting point that doesn't fit the file's hashes is refused, saying which.
    let mut wrong = meet_in_the_middle_of(PREFIX, "B-A", SUFFIX);
    wrong["meet_in_the_middle"]["meeting_point_b_seed1_hex"] = serde_json::json!("0xDEADBEEF");
    let err = create(&state, wrong).await.unwrap_err();
    assert!(matches!(&err, AppError::BadRequest(m) if m.contains("hash_b") && m.contains("right where the suffix begins")), "{err:?}");
    // And so is one for the wrong place in the name - the state after the
    // whole filename, say, rather than after the directory.
    let mut misplaced = meet_in_the_middle_of(PREFIX, "B-A", SUFFIX);
    misplaced["meet_in_the_middle"]["meeting_point_a_seed1_hex"] = serde_json::json!("0x00000000");
    assert!(create(&state, misplaced).await.is_err());
}

#[tokio::test]
async fn a_target_refuses_what_isnt_a_meet_in_the_middle_targets() {
    let state = state(1.0).await;
    // The settings an alphabet or dictionary target has that this doesn't.
    for extra in [
        serde_json::json!({"lower_bound": "AB"}),
        serde_json::json!({"upper_bound": "BB"}),
        serde_json::json!({"prune_whole_candidate": true}),
        serde_json::json!({"insert_from_start": ["\\", 2]}),
        serde_json::json!({"start_len": 3}),
        serde_json::json!({"auto_priority": true}),
        serde_json::json!({"send_basenames": true}),
    ] {
        let err = create(&state, extra.clone()).await.unwrap_err();
        assert!(matches!(&err, AppError::BadRequest(m) if m.contains("isn't for a meet-in-the-middle target")), "{extra} gave {err:?}");
    }
    // Words or a meeting point, not both.
    let both = serde_json::json!({"dictionary": {"word_lists": ["english-1"], "separators": [""], "max_words": 1}});
    assert!(matches!(create(&state, both).await, Err(AppError::BadRequest(m)) if m.contains("not both")));

    // Lengths and a table that can't make a search.
    let lengths = |min: i64, max: i64, table: i64| {
        let mut extra = meet_in_the_middle_of(PREFIX, "B-A", SUFFIX);
        extra["meet_in_the_middle"]["min_length"] = serde_json::json!(min);
        extra["meet_in_the_middle"]["max_length"] = serde_json::json!(max);
        extra["meet_in_the_middle"]["table_length"] = serde_json::json!(table);
        extra
    };
    assert!(create(&state, lengths(3, 2, 1)).await.is_err(), "the shortest past the longest");
    assert!(create(&state, lengths(0, 3, 0)).await.is_err(), "no shortest directory");
    assert!(create(&state, lengths(2, 17, 1)).await.is_err(), "a directory longer than a client searches");
    let table = create(&state, lengths(2, 3, 3)).await.unwrap_err();
    assert!(matches!(&table, AppError::BadRequest(m) if m.contains("shortest directory")), "a table longer than the directories: {table:?}");
    assert!(create(&state, lengths(2, 3, 0)).await.is_ok(), "no table at all is allowed - the walk does everything");

    // A filename longer than a client searches.
    let mut long = meet_in_the_middle_of(PREFIX, "B-A", SUFFIX);
    long["prefix"] = serde_json::json!("P".repeat(130));
    // Its meeting point has to fit that prefix's own hashes to get that far.
    let fitting = meet_in_the_middle_of(&"P".repeat(130), "B-A", SUFFIX);
    long["hash_a_hex"] = fitting["hash_a_hex"].clone();
    long["hash_b_hex"] = fitting["hash_b_hex"].clone();
    long["meet_in_the_middle"] = fitting["meet_in_the_middle"].clone();
    let err = create(&state, long).await.unwrap_err();
    assert!(matches!(&err, AppError::BadRequest(m) if m.contains("longest filename")), "{err:?}");
}

#[tokio::test]
async fn only_a_client_with_room_for_the_table_gets_the_work() {
    let state = state(100.0).await;
    create(&state, serde_json::json!({})).await.unwrap();
    // A table of one character of three is 8 bytes of keys and 8 of filter.
    let table_bytes = meet_in_the_middle::table_bytes(3, 1);
    assert_eq!(table_bytes, 3 * 8 + 8);

    let old = user(&state, "old", "1.5.0", Some(1 << 30)).await;
    assert!(claim_work(&state, &old).await.is_none(), "a client from before meet-in-the-middle targets gets none");
    let silent = user(&state, "silent", "1.6.0", None).await;
    assert!(claim_work(&state, &silent).await.is_none(), "nor one that didn't say what it will give");
    let stingy = user(&state, "stingy", "1.6.0", Some(table_bytes as i64 - 1)).await;
    assert!(claim_work(&state, &stingy).await.is_none(), "nor one that won't give the table's memory");
    let exact = user(&state, "exact", "1.6.0", Some(table_bytes as i64)).await;
    assert!(claim_work(&state, &exact).await.is_some(), "exactly enough is enough");
}

#[tokio::test]
async fn a_claim_says_everything_the_client_needs() {
    let state = state(2.0).await;
    let id = create(&state, serde_json::json!({"prune_symbol_runs": true, "max_backslash_count": 2})).await.unwrap();
    let client = client(&state, "u").await;
    let claim = claim_work(&state, &client).await.unwrap();

    assert!(claim.meet_in_the_middle, "a meet-in-the-middle claim says so");
    assert!(!claim.dictionary);
    assert_eq!(claim.target_id, id);
    assert_eq!((claim.prefix.as_str(), claim.suffix.as_str()), (PREFIX, SUFFIX));
    assert_eq!(claim.alphabet, ALPHABET);
    assert_eq!(claimed(&claim), (2, 0, 2), "the first two tails of the shortest directory");
    assert_eq!(claim.meet_in_the_middle_table_length, Some(1));
    assert_eq!(claim.candidate_count, 2, "a meet-in-the-middle range counts tails");
    // The meeting point itself, and the hashes to check a find against.
    let settings = meet_in_the_middle::of_target(&target(&state, id).await).unwrap();
    assert_eq!(claim.meeting_point_a_seed1_hex, Some(format!("0x{:08X}", settings.meeting_point_a.0)));
    assert_eq!(claim.meeting_point_a_seed2_hex, Some(format!("0x{:08X}", settings.meeting_point_a.1)));
    assert_eq!(claim.meeting_point_b_seed1_hex, Some(format!("0x{:08X}", settings.meeting_point_b.0)));
    assert_eq!(claim.meeting_point_b_seed2_hex, Some(format!("0x{:08X}", settings.meeting_point_b.1)));
    assert_eq!(claim.hash_a_hex, format!("0x{:08X}", mpq_hash("SETUPDAT\\B-A\\BWUNIN.PIF", HASH_A_OFFSET)));
    assert_eq!(claim.hash_b_hex, format!("0x{:08X}", mpq_hash("SETUPDAT\\B-A\\BWUNIN.PIF", HASH_B_OFFSET)));
    // The pruning rules are the target's; the whole-candidate switch isn't a
    // meet-in-the-middle search's (it checks every character anyway).
    assert!(claim.prune_symbol_runs && !claim.prune_whole_candidate);
    assert_eq!(claim.max_backslash_count, 2);
    // The bounds are for a human: the table's characters aren't searched one
    // by one, so they show as '?'.
    assert_eq!(claim.lower_bound_filename, "SETUPDAT\\?-\\BWUNIN.PIF");
    assert_eq!(claim.upper_bound_filename, "SETUPDAT\\?A\\BWUNIN.PIF");
    // None of a dictionary target's fields.
    assert!(claim.word_lists.is_none() && claim.first_candidate_number.is_none() && !claim.send_basenames);
}

#[tokio::test]
async fn claims_hand_out_every_tail_of_every_length_once() {
    let state = state(2.0).await;
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let client = client(&state, "u").await;

    // Two tails a claim, through 3 tails at length 2 and 9 at length 3.
    let mut handed: Vec<(i64, i64, i64)> = Vec::new();
    while let Some(claim) = claim_work(&state, &client).await {
        let (length, start, end) = claimed(&claim);
        complete(&state, &client, claim.range_id, None, 1.0, end - start).await.unwrap();
        handed.push((length, start, end));
        assert!(handed.len() < 20, "a claim should run out");
    }
    assert_eq!(
        handed,
        vec![(2, 0, 2), (2, 2, 3), (3, 0, 2), (3, 2, 4), (3, 4, 6), (3, 6, 8), (3, 8, 9)],
        "each length's tails in turn, a chunk at a time, never across two lengths"
    );
    // Nothing left, and the cursor says so.
    assert!(claim_work(&state, &client).await.is_none());
    let (len, index): (i64, i64) = sqlx::query_as("SELECT candidate_len, next_index FROM target_progress WHERE target_id = ?")
        .bind(id)
        .fetch_one(&state.pool)
        .await
        .unwrap();
    assert_eq!((len, index), (3, 9), "past the last tail of the last length");
}

#[tokio::test]
async fn a_heartbeat_records_how_far_the_walk_has_got() {
    let state = state(9.0).await;
    create(&state, serde_json::json!({})).await.unwrap();
    let first = client(&state, "first").await;
    let claim = claim_work(&state, &first).await.unwrap();
    assert_eq!(claimed(&claim), (2, 0, 3), "the whole of the shortest length");

    assert!(!heartbeat(&state, &first, claim.range_id, Some(2)).await.unwrap());
    assert_eq!(range(&state, claim.range_id).await.progress(), Some(1), "every tail below 2 searched: the last is 1");
    // Backwards, and past the end, are both ignored.
    heartbeat(&state, &first, claim.range_id, Some(1)).await.unwrap();
    assert_eq!(range(&state, claim.range_id).await.progress(), Some(1), "progress never moves backwards");
    heartbeat(&state, &first, claim.range_id, Some(4)).await.unwrap();
    assert_eq!(range(&state, claim.range_id).await.progress(), Some(1), "nor past the range's end");

    // Reassigned after the lease expires: the searched part is kept and
    // credited, and the rest goes to whoever claims next.
    sqlx::query("UPDATE ranges SET lease_expires_at = 1 WHERE id = ?").bind(claim.range_id).execute(&state.pool).await.unwrap();
    ranges::reclaim_expired(&state.pool).await.unwrap();
    let second = client(&state, "second").await;
    let next = claim_work(&state, &second).await.unwrap();
    assert_ne!(next.range_id, claim.range_id, "the remainder is a range of its own");
    assert_eq!(claimed(&next), (2, 2, 3), "from where the first client's heartbeat got to");
    let searched = range(&state, claim.range_id).await;
    assert_eq!((searched.status.as_str(), searched.end(), searched.last_assigned_user_id), ("completed", 2, Some(first.id)));
}

#[tokio::test]
async fn quitting_splits_the_range_at_the_tail_reached() {
    let state = state(9.0).await;
    create(&state, serde_json::json!({})).await.unwrap();
    let client = client(&state, "u").await;
    let claim = claim_work(&state, &client).await.unwrap();
    quit(&state, &client, claim.range_id, Some(2)).await.unwrap();

    let searched = range(&state, claim.range_id).await;
    assert_eq!((searched.status.as_str(), searched.start(), searched.end()), ("completed", 0, 2));
    let rest = claim_work(&state, &client).await.unwrap();
    assert_eq!(claimed(&rest), (2, 2, 3), "the rest is handed out straight away");
}

#[tokio::test]
async fn a_completed_range_measures_the_clients_tails_a_second() {
    let state = state(4.0).await;
    create(&state, serde_json::json!({})).await.unwrap();
    let client = client(&state, "u").await;
    let claim = claim_work(&state, &client).await.unwrap();
    complete(&state, &client, claim.range_id, None, 2.0, 100).await.unwrap();

    let client = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?").bind(client.id).fetch_one(&state.pool).await.unwrap();
    assert_eq!(client.ema_meet_in_the_middle_rate_per_sec, Some(50.0), "100 tails in 2 seconds");
    assert_eq!((client.ema_rate_per_sec, client.ema_dictionary_rate_per_sec), (None, None), "measured apart from the other searches");
    // And the next claim is sized from it: 50 a second for 4 seconds is more
    // than the next length has at all, so it gets the whole of it.
    let next = claim_work(&state, &client).await.unwrap();
    assert_eq!(claimed(&next), (3, 0, 9));
}

#[tokio::test]
async fn a_find_solves_the_target() {
    let state = state(100.0).await;
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let client = client(&state, "u").await;
    let claim = claim_work(&state, &client).await.unwrap();
    assert_eq!(claimed(&claim), (2, 0, 3), "every tail of the shortest length");
    // The directory is three characters, so it isn't in this range - the
    // next one is where it turns up.
    complete(&state, &client, claim.range_id, None, 1.0, 3).await.unwrap();
    let claim = claim_work(&state, &client).await.unwrap();
    assert_eq!(claim.directory_length, Some(3));
    complete(&state, &client, claim.range_id, Some("SETUPDAT\\B-A\\BWUNIN.PIF"), 1.0, 9).await.unwrap();

    let target = target(&state, id).await;
    assert_eq!((target.status.as_str(), target.found_filename.as_deref()), ("solved", Some("SETUPDAT\\B-A\\BWUNIN.PIF")));
    assert!(claim_work(&state, &client).await.is_none(), "a solved target hands out nothing more");
}

#[tokio::test]
async fn a_find_is_checked_against_the_key() {
    let state = state(100.0).await;
    // The file's own key, from its basename: what an archive lists for it.
    let key = format!("0x{:08X}", mpq_hash("BWUNIN.PIF", FILE_KEY_OFFSET));
    let id = create(&state, serde_json::json!({"encryption_key_hex": key})).await.unwrap();
    assert_eq!(target(&state, id).await.encryption_key.map(crate::models::i64_to_u32), Some(mpq_hash("BWUNIN.PIF", FILE_KEY_OFFSET)));

    // A client that checks matches is given the key; one older than
    // protocol 1.7 isn't, as for any target.
    let older = client(&state, "older").await;
    assert_eq!(claim_work(&state, &older).await.unwrap().encryption_key_hex, None);
    let newer = user(&state, "newer", "1.7.0", Some(1 << 30)).await;
    let claim = claim_work(&state, &newer).await.unwrap();
    assert_eq!(claim.encryption_key_hex, Some(key));
    complete(&state, &newer, claim.range_id, Some("SETUPDAT\\B-A\\BWUNIN.PIF"), 1.0, 9).await.unwrap();
    assert_eq!(target(&state, id).await.status, "solved", "the right name passes the key");

    // A key the name doesn't have makes the same find a false match.
    let other = create(&state, serde_json::json!({"name": "wrong-key", "encryption_key_hex": "0x12345678"})).await.unwrap();
    let claim = claim_work(&state, &newer).await.unwrap();
    assert_eq!(claim.target_id, other);
    complete(&state, &newer, claim.range_id, Some("SETUPDAT\\B-A\\BWUNIN.PIF"), 1.0, 9).await.unwrap();
    assert_ne!(target(&state, other).await.status, "solved");
}

#[tokio::test]
async fn a_canary_copies_the_meet_in_the_middle_target_and_plants_a_directory_in_it() {
    let mut config = config(100.0);
    config.canary_probability = 1.0;
    config.canary_seconds = 4.0;
    let state = state_with(config).await;
    create(&state, serde_json::json!({})).await.unwrap();
    let client = client(&state, "u").await;
    // Without a measured meet-in-the-middle rate there's no canary - a
    // guessed one could make it take far too long.
    assert_eq!(claim_work(&state, &client).await.unwrap().target_name, "setupdat");
    sqlx::query("UPDATE users SET ema_meet_in_the_middle_rate_per_sec = 1.0 WHERE id = ?").bind(client.id).execute(&state.pool).await.unwrap();
    let client = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?").bind(client.id).fetch_one(&state.pool).await.unwrap();

    let canary = claim_work(&state, &client).await.unwrap();
    assert_eq!(canary.target_name, crate::canary::CANARY_TARGET_NAME);
    assert!(canary.meet_in_the_middle, "a meet-in-the-middle target's canary is meet-in-the-middle work too");
    // It copies the template: the same alphabet, prefix, suffix and - so
    // that it costs the client what the real work does - table length.
    assert_eq!((canary.prefix.as_str(), canary.suffix.as_str(), canary.alphabet.as_str()), (PREFIX, SUFFIX, ALPHABET));
    assert_eq!(canary.meet_in_the_middle_table_length, Some(1));
    let (length, start, end) = claimed(&canary);
    assert_eq!(end - start, 4.min(3i64.pow((length - 1) as u32)), "four seconds at a tail a second");

    // The planted directory: in the range, of letters and digits, and the
    // claim's meeting point and hashes are its own.
    let planted: String = sqlx::query_scalar("SELECT filename FROM canaries").fetch_one(&state.pool).await.unwrap();
    let directory = planted.strip_prefix(PREFIX).unwrap().strip_suffix(SUFFIX).unwrap();
    assert_eq!(directory.len() as i64, length);
    assert!(directory.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()), "{directory} - no pruning rule skips it");
    let space = meet_in_the_middle::MeetInTheMiddleSpace::new(ALPHABET, length, length, 1).unwrap();
    let index = space.tail_index(&directory[1..]).unwrap();
    assert!(start <= index && index < end, "the planted directory's tail {index} is in [{start}, {end})");
    let after = continue_hash(INITIAL_STATE, &format!("{PREFIX}{directory}"), HASH_A_OFFSET);
    assert_eq!(canary.meeting_point_a_seed1_hex, Some(format!("0x{:08X}", after.0)));
    assert_eq!(canary.hash_a_hex, format!("0x{:08X}", mpq_hash(&planted, HASH_A_OFFSET)));

    // Finding it is passing; the canary's target is then done with.
    complete(&state, &client, canary.range_id, Some(&planted), 1.0, end - start).await.unwrap();
    let result: (String, Option<String>) = sqlx::query_as("SELECT result, NULL FROM canaries").fetch_one(&state.pool).await.unwrap();
    assert_eq!(result.0, "found");
}

#[tokio::test]
async fn a_canary_isnt_made_for_a_client_without_room_for_the_table() {
    let mut config = config(100.0);
    config.canary_probability = 1.0;
    let state = state_with(config).await;
    create(&state, serde_json::json!({})).await.unwrap();
    // Measured, so the only thing standing in the way is the memory.
    let stingy = user(&state, "stingy", "1.6.0", Some(1)).await;
    sqlx::query("UPDATE users SET ema_meet_in_the_middle_rate_per_sec = 1.0 WHERE id = ?").bind(stingy.id).execute(&state.pool).await.unwrap();
    let stingy = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?").bind(stingy.id).fetch_one(&state.pool).await.unwrap();
    assert!(claim_work(&state, &stingy).await.is_none(), "no canary, and no real work either");
    assert_eq!(sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM canaries").fetch_one(&state.pool).await.unwrap(), 0);
}

#[tokio::test]
async fn patching_leaves_what_the_tails_are_numbered_by_alone() {
    let state = state(1.0).await;
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let patch = |body: serde_json::Value| {
        let state = state.clone();
        async move {
            let req = serde_json::from_value(body).unwrap();
            handlers::admin_patch_target(State(state), AdminAuth, Path(id), Json(req)).await
        }
    };
    for body in [
        serde_json::json!({"alphabet": "ABC"}),
        serde_json::json!({"alphabet_name": "size43"}),
        serde_json::json!({"start_len": 3}),
        serde_json::json!({"prune_whole_candidate": true}),
        serde_json::json!({"insert_from_start": ["\\", 1]}),
        serde_json::json!({"auto_priority": true}),
    ] {
        let err = patch(body.clone()).await.unwrap_err();
        assert!(matches!(&err, AppError::BadRequest(m) if m.contains("isn't for a meet-in-the-middle target")), "{body} gave {err:?}");
    }
    // What can be changed, can.
    patch(serde_json::json!({"priority": 5, "description": "the setupdat scripts", "prune_symbol_runs": true})).await.unwrap();
    let target = target(&state, id).await;
    assert_eq!((target.priority, target.prune_symbol_runs), (5, 1));
}

#[tokio::test]
async fn a_meet_in_the_middle_target_has_no_priority_or_skip_ranges() {
    let state = state(1.0).await;
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let priority = serde_json::from_value(serde_json::json!({"pattern": "A", "length": 3, "priority": 1})).unwrap();
    let err = handlers::admin_create_priority_range(State(state.clone()), AdminAuth, Path(id), Json(priority)).await.unwrap_err();
    assert!(matches!(&err, AppError::BadRequest(m) if m.contains("no priority ranges")), "{err:?}");
    let skip = serde_json::from_value(serde_json::json!({"pattern": "A", "length": 3, "reason": "no"})).unwrap();
    let err = handlers::admin_create_skip_range(State(state.clone()), AdminAuth, Path(id), Json(skip)).await.unwrap_err();
    assert!(matches!(&err, AppError::BadRequest(m) if m.contains("no skip ranges")), "{err:?}");
}

#[tokio::test]
async fn the_dashboard_shows_the_meet_in_the_middle_the_table_and_the_tails() {
    let state = state(2.0).await;
    create(&state, serde_json::json!({})).await.unwrap();
    let client = client(&state, "u").await;
    let first = claim_work(&state, &client).await.unwrap();
    heartbeat(&state, &client, first.range_id, Some(1)).await.unwrap();

    let data = crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().0;
    let target = data.targets.iter().find(|t| t.name == "setupdat").unwrap();
    assert_eq!(target.kind, "meet-in-the-middle");
    assert!(target.dictionary.is_none());
    let meet_in_the_middle = target.meet_in_the_middle.as_ref().unwrap();
    assert_eq!((meet_in_the_middle.min_length, meet_in_the_middle.max_length, meet_in_the_middle.table_length), (2, 3, 1));
    assert_eq!((meet_in_the_middle.table_directories, meet_in_the_middle.table_bytes), (3, meet_in_the_middle::table_bytes(3, 1)));
    assert_eq!(meet_in_the_middle.meeting_point_a, format!("{:08X} {:08X}", continue_hash(INITIAL_STATE, "SETUPDAT\\B-A", HASH_A_OFFSET).0, continue_hash(INITIAL_STATE, "SETUPDAT\\B-A", HASH_A_OFFSET).1));
    // 3 tails at length 2 and 9 at length 3, standing for 3 directories each.
    assert_eq!(meet_in_the_middle.tail_count, 12);
    assert_eq!(meet_in_the_middle.directory_count, 36);

    // Its range shows the directories it covers, the table's characters as '?'.
    assert_eq!(target.ranges.len(), 1);
    let range = &target.ranges[0];
    assert_eq!((range.candidate_len, range.candidate_count), (2, 2));
    assert_eq!((range.first_candidate.as_str(), range.last_candidate.as_str()), ("?-", "?A"));
    assert_eq!(range.progress_candidate.as_deref(), Some("?-"));
    // And where the next range starts.
    assert_eq!((target.cursor_candidate_len, target.cursor_candidate.as_str()), (2, "?B"));

    // A gap within a length shows up; one "between" two lengths never does.
    let second = claim_work(&state, &client).await.unwrap();
    assert_eq!(claimed(&second), (2, 2, 3));
    complete(&state, &client, second.range_id, None, 1.0, 1).await.unwrap();
    let third = claim_work(&state, &client).await.unwrap();
    assert_eq!(third.directory_length, Some(3), "on to the next length");
    let data = crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().0;
    let target = data.targets.iter().find(|t| t.name == "setupdat").unwrap();
    assert!(target.gaps.is_empty(), "nothing is uncovered yet: {:?}", target.gaps.len());
}
