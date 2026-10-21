//! False matches end to end, through the handlers: what's known about a
//! target's file beyond its hashes given to the clients that check it and
//! kept from those that don't, the false matches clients report, the finds
//! an older client makes that are false matches - and the rest of its range,
//! handed out again - and what the dashboard shows of them.
//! `false_matches.rs` tests the checks themselves.

use axum::extract::{Path, State};
use axum::Json;
use namebreak_protocol::{ClaimResponse, CompleteRequest, HeartbeatRequest, MAX_FALSE_MATCHES_PER_REPORT};

use crate::auth::{AdminAuth, AuthedUser};
use crate::error::AppError;
use crate::false_matches::start_slot;
use crate::handlers;
use crate::models::{Range, Target, User};
use crate::mpq::{mpq_hash, FILE_KEY_OFFSET, HASH_A_OFFSET, HASH_B_OFFSET};
use crate::ranges;
use crate::state::{now_unix, AppState, Inner, RangeConfig};

/// Each claim gets `chunk` candidates, whatever the client's measured rate.
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

async fn state(chunk: f64) -> AppState {
    let pool = crate::db::connect("sqlite::memory:").await.unwrap();
    AppState(std::sync::Arc::new(Inner { pool, admin_token: "t".into(), config: config(chunk), disk: Default::default() }))
}

async fn user(state: &AppState, name: &str, protocol_version: &str) -> User {
    let now = now_unix();
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO users (username, hostname, token, created_at, last_seen_at, protocol_version, backend) VALUES (?, 'host', ?, ?, ?, ?, 'cuda') RETURNING id",
    )
    .bind(name)
    .bind(format!("{name}-token"))
    .bind(now)
    .bind(now)
    .bind(protocol_version)
    .fetch_one(&state.pool)
    .await
    .unwrap();
    sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

fn hex(value: u32) -> String {
    format!("0x{value:08X}")
}

/// The name every target here is for - candidate "ABA" of the alphabet "AB",
/// number 2 of the eight three-character ones.
const NAME: &str = "REZ\\ABA.WAV";

/// The key of `NAME`'s basename.
fn key() -> u32 {
    mpq_hash("ABA.WAV", FILE_KEY_OFFSET)
}

/// A hash table position whose slots `NAME`'s lookup starts at - or, with
/// `reaches` false, doesn't.
fn position(reaches: bool) -> serde_json::Value {
    let slot = start_slot(NAME, 2048);
    let first = if reaches { slot } else { (slot + 10) % 2048 };
    serde_json::json!({"table_size": 2048, "first_start_slot": first, "last_start_slot": (first + 2) % 2048})
}

/// An alphabet target with `NAME`'s hashes: three characters of "AB", the
/// eight of them one range of `state(8.0)` - with `extra` merged into the
/// request.
async fn create(state: &AppState, extra: serde_json::Value) -> Result<i64, AppError> {
    let mut body = serde_json::json!({
        "name": "rez", "prefix": "REZ\\", "suffix": ".WAV",
        "hash_a_hex": hex(mpq_hash(NAME, HASH_A_OFFSET)), "hash_b_hex": hex(mpq_hash(NAME, HASH_B_OFFSET)),
        "alphabet": "AB", "lower_bound": "AAA", "upper_bound": "BBB", "start_len": 3,
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    let req = serde_json::from_value(body).unwrap();
    handlers::admin_create_target(State(state.clone()), AdminAuth, Json(req)).await.map(|json| json.0.target_id)
}

async fn patch(state: &AppState, id: i64, body: serde_json::Value) -> Result<(), AppError> {
    let req = serde_json::from_value(body).unwrap();
    handlers::admin_patch_target(State(state.clone()), AdminAuth, Path(id), Json(req)).await.map(|_| ())
}

async fn target(state: &AppState, id: i64) -> Target {
    sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

async fn ranges_of(state: &AppState, target_id: i64) -> Vec<Range> {
    sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE target_id = ? ORDER BY start_block, start_index").bind(target_id).fetch_all(&state.pool).await.unwrap()
}

async fn claim(state: &AppState, user: &User) -> Option<ClaimResponse> {
    ranges::claim_range(&state.pool, &state.config, user).await.unwrap()
}

async fn heartbeat(state: &AppState, user: &User, range_id: i64, false_matches: &[&str]) -> Result<bool, AppError> {
    let req = HeartbeatRequest { false_matches: false_matches.iter().map(|f| f.to_string()).collect(), ..Default::default() };
    handlers::heartbeat(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await.map(|json| json.0.range_released)
}

async fn complete(state: &AppState, user: &User, range_id: i64, found: Option<&str>, false_matches: &[&str]) -> Result<(), AppError> {
    let req = CompleteRequest {
        found: found.is_some(),
        filename: found.map(str::to_string),
        elapsed_seconds: 1.0,
        candidates_processed: 8,
        basenames: Vec::new(),
        false_matches: false_matches.iter().map(|f| f.to_string()).collect(),
    };
    handlers::complete(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await.map(|_| ())
}

/// A target's false matches, as (filename, reason, reporter's id).
async fn false_matches(state: &AppState, target_id: i64) -> Vec<(String, String, i64)> {
    sqlx::query_as("SELECT filename, reason, user_id FROM false_matches WHERE target_id = ? ORDER BY id").bind(target_id).fetch_all(&state.pool).await.unwrap()
}

#[tokio::test]
async fn the_key_and_hash_table_position_go_only_to_clients_that_check_them() {
    let state = state(1.0).await;
    let id = create(&state, serde_json::json!({"encryption_key_hex": hex(key()), "hash_table_position": position(true)})).await.unwrap();
    let old = user(&state, "old", "1.6.0").await;
    let new = user(&state, "new", "1.7.0").await;

    let c = claim(&state, &old).await.unwrap();
    assert_eq!(c.target_id, id);
    assert_eq!((c.encryption_key_hex.as_deref(), c.hash_table_size), (None, None), "an older client's claim is as it was");
    let json = serde_json::to_value(&c).unwrap();
    for field in ["encryption_key_hex", "hash_table_size", "hash_table_first_start_slot", "hash_table_last_start_slot"] {
        assert!(json.get(field).is_none(), "{field} isn't even in an older client's claim");
    }

    let c = claim(&state, &new).await.unwrap();
    let slot = start_slot(NAME, 2048);
    assert_eq!(c.encryption_key_hex, Some(hex(key())));
    assert_eq!((c.hash_table_size, c.hash_table_first_start_slot, c.hash_table_last_start_slot), (Some(2048), Some(slot), Some((slot + 2) % 2048)));
}

#[tokio::test]
async fn a_dictionary_targets_key_still_goes_to_the_clients_that_search_by_it() {
    let state = state(1.0).await;
    let list = b"alpha\nbravo\n";
    let _ = handlers::admin_put_word_list(State(state.clone()), AdminAuth, Path("nato".to_string()), axum::body::Bytes::copy_from_slice(list)).await.unwrap();
    let req = serde_json::from_value(serde_json::json!({
        "name": "dict", "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x1", "hash_b_hex": "0x2",
        "encryption_key_hex": "0x0000ABCD", "hash_table_position": {"table_size": 16, "first_start_slot": 3, "last_start_slot": 4},
        "dictionary": {"word_lists": ["nato"], "separators": [""], "max_words": 1},
    }))
    .unwrap();
    let _ = handlers::admin_create_target(State(state.clone()), AdminAuth, Json(req)).await.unwrap();

    let c = claim(&state, &user(&state, "old", "1.5.0").await).await.unwrap();
    assert_eq!((c.encryption_key_hex.as_deref(), c.hash_table_size), (Some("0x0000ABCD"), None), "the key as before, and no more");
    let c = claim(&state, &user(&state, "new", "1.7.0").await).await.unwrap();
    assert_eq!((c.encryption_key_hex.as_deref(), c.hash_table_size, c.hash_table_first_start_slot), (Some("0x0000ABCD"), Some(16), Some(3)));
}

#[tokio::test]
async fn hash_table_positions_are_checked_patched_and_removed() {
    let state = state(1.0).await;
    for bad in [
        serde_json::json!({"table_size": 2000, "first_start_slot": 1, "last_start_slot": 2}),
        serde_json::json!({"table_size": 2048, "first_start_slot": 2048, "last_start_slot": 2}),
        serde_json::json!({"table_size": 2048, "first_start_slot": 1, "last_start_slot": -1}),
    ] {
        assert!(matches!(create(&state, serde_json::json!({"hash_table_position": bad})).await, Err(AppError::BadRequest(_))), "{bad}");
    }
    let id = create(&state, serde_json::json!({})).await.unwrap();
    assert_eq!(target(&state, id).await.hash_table_position(), None);

    patch(&state, id, serde_json::json!({"hash_table_position": {"table_size": 16, "first_start_slot": 14, "last_start_slot": 1}})).await.unwrap();
    let position = target(&state, id).await.hash_table_position().unwrap();
    assert_eq!((position.table_size, position.first_start_slot, position.last_start_slot), (16, 14, 1), "wrapping past the end");
    assert!(matches!(patch(&state, id, serde_json::json!({"hash_table_position": {"table_size": 16, "first_start_slot": 16, "last_start_slot": 1}})).await, Err(AppError::BadRequest(_))));
    patch(&state, id, serde_json::json!({"priority": 3})).await.unwrap();
    assert!(target(&state, id).await.hash_table_position().is_some(), "left as it is when not given");
    patch(&state, id, serde_json::json!({"hash_table_position": null})).await.unwrap();
    assert_eq!(target(&state, id).await.hash_table_position(), None);
}

#[tokio::test]
async fn a_newer_clients_false_matches_are_kept_once_and_the_range_completes() {
    let state = state(8.0).await;
    // The key of another basename: NAME is a false match.
    let wrong_key = key() ^ 1;
    let id = create(&state, serde_json::json!({"encryption_key_hex": hex(wrong_key), "hash_table_position": position(false)})).await.unwrap();
    let client = user(&state, "new", "1.7.0").await;
    let c = claim(&state, &client).await.unwrap();

    assert!(!heartbeat(&state, &client, c.range_id, &[NAME]).await.unwrap());
    let kept = false_matches(&state, id).await;
    assert_eq!(kept.len(), 1);
    assert_eq!((kept[0].0.as_str(), kept[0].2), (NAME, client.id));
    assert!(kept[0].1.contains(&format!("its basename's key is {}, not the file's {}", hex(key()), hex(wrong_key))), "{}", kept[0].1);
    assert!(kept[0].1.contains(", and its lookup starts at hash table slot"), "both of what it fails: {}", kept[0].1);

    // Again with the next report (one that got no answer, say), and a name
    // that doesn't even match the hashes: nothing more.
    heartbeat(&state, &client, c.range_id, &[NAME, "REZ\\BBB.WAV"]).await.unwrap();
    assert_eq!(false_matches(&state, id).await.len(), 1);
    let too_many = vec![NAME; MAX_FALSE_MATCHES_PER_REPORT + 1];
    assert!(matches!(heartbeat(&state, &client, c.range_id, &too_many).await, Err(AppError::BadRequest(_))));

    complete(&state, &client, c.range_id, None, &[NAME]).await.unwrap();
    assert_eq!(target(&state, id).await.status, "active");
    let ranges = ranges_of(&state, id).await;
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].status, "completed", "searched past it, to the end");
}

#[tokio::test]
async fn a_reported_false_match_that_passes_the_checks_as_they_are_now_is_the_find() {
    let state = state(8.0).await;
    let id = create(&state, serde_json::json!({"encryption_key_hex": hex(key() ^ 1)})).await.unwrap();
    let client = user(&state, "new", "1.7.0").await;
    let c = claim(&state, &client).await.unwrap();
    // The operator puts the key right while the client searches with the
    // wrong one.
    patch(&state, id, serde_json::json!({"encryption_key_hex": hex(key())})).await.unwrap();

    assert!(heartbeat(&state, &client, c.range_id, &[NAME]).await.unwrap(), "the report that solves it releases it - the search is over");
    let solved = target(&state, id).await;
    assert_eq!((solved.status.as_str(), solved.found_filename.as_deref(), solved.found_by_user_id), ("solved", Some(NAME), Some(client.id)));
    assert!(false_matches(&state, id).await.is_empty());
}

#[tokio::test]
async fn an_older_clients_false_find_leaves_the_target_unsolved_and_hands_out_the_rest_of_the_range() {
    let state = state(8.0).await;
    let id = create(&state, serde_json::json!({"hash_table_position": position(false)})).await.unwrap();
    let old = user(&state, "old", "1.6.0").await;
    let c = claim(&state, &old).await.unwrap();
    assert_eq!((c.lower_bound_filename.as_str(), c.upper_bound_filename.as_str()), ("REZ\\AAA.WAV", "REZ\\BBB.WAV"));

    // It stopped at NAME, taking it for the file.
    complete(&state, &old, c.range_id, Some(NAME), &[]).await.unwrap();
    let t = target(&state, id).await;
    assert_eq!((t.status.as_str(), t.found_filename), ("active", None));
    let kept = false_matches(&state, id).await;
    assert_eq!(kept.len(), 1);
    assert!(kept[0].1.starts_with("its lookup starts at hash table slot"), "{}", kept[0].1);

    // Everything up to and including it was searched; the rest - which
    // hasn't got it in it - is anyone's, an older client's too.
    let c = claim(&state, &old).await.unwrap();
    assert_eq!((c.lower_bound_filename.as_str(), c.upper_bound_filename.as_str()), ("REZ\\ABB.WAV", "REZ\\BBB.WAV"));
    let ranges = ranges_of(&state, id).await;
    let statuses: Vec<(&str, i64)> = ranges.iter().map(|r| (r.status.as_str(), r.false_match_inside)).collect();
    assert_eq!(statuses, [("completed", 0), ("in_progress", 0)]);
    assert_eq!((ranges[0].start_index, ranges[0].end_index, ranges[0].last_assigned_user_id), (0, 3, Some(old.id)));

    // And a real find is still one.
    let name = "REZ\\BAB.WAV";
    sqlx::query("UPDATE targets SET hash_a = ?, hash_b = ?, hash_table_size = NULL, hash_table_first_start_slot = NULL, hash_table_last_start_slot = NULL WHERE id = ?")
        .bind(mpq_hash(name, HASH_A_OFFSET) as i64)
        .bind(mpq_hash(name, HASH_B_OFFSET) as i64)
        .bind(id)
        .execute(&state.pool)
        .await
        .unwrap();
    complete(&state, &old, c.range_id, Some(name), &[]).await.unwrap();
    assert_eq!(target(&state, id).await.found_filename.as_deref(), Some(name));
}

#[tokio::test]
async fn the_rest_of_a_dictionary_range_an_older_client_stopped_in_goes_only_to_newer_clients() {
    let state = state(2.0).await;
    let list = b"alpha\nbravo\ncharlie\ndelta\n";
    let _ = handlers::admin_put_word_list(State(state.clone()), AdminAuth, Path("nato".to_string()), axum::body::Bytes::copy_from_slice(list)).await.unwrap();
    // Candidate 1 of four one-word ones, two a range.
    let name = "MUSIC\\BRAVO.WAV";
    let req = serde_json::from_value(serde_json::json!({
        "name": "dict", "prefix": "music\\", "suffix": ".wav",
        "hash_a_hex": hex(mpq_hash(name, HASH_A_OFFSET)), "hash_b_hex": hex(mpq_hash(name, HASH_B_OFFSET)),
        "hash_table_position": {"table_size": 1 << 20, "first_start_slot": (start_slot(name, 1 << 20) + 5) % (1 << 20), "last_start_slot": (start_slot(name, 1 << 20) + 5) % (1 << 20)},
        "dictionary": {"word_lists": ["nato"], "separators": [""], "max_words": 1},
    }))
    .unwrap();
    let id = handlers::admin_create_target(State(state.clone()), AdminAuth, Json(req)).await.unwrap().0.target_id;
    let old = user(&state, "old", "1.6.0").await;
    let new = user(&state, "new", "1.7.0").await;

    let c = claim(&state, &old).await.unwrap();
    assert_eq!((c.first_candidate_number, c.end_candidate_number), (Some(0), Some(2)));
    complete(&state, &old, c.range_id, Some(name), &[]).await.unwrap();
    assert_eq!(target(&state, id).await.status, "active");
    assert_eq!(false_matches(&state, id).await.len(), 1);

    // A dictionary range's progress is a number a completion doesn't carry:
    // the whole of it goes back, false match and all - but not to a client
    // that would stop at it again.
    let c = claim(&state, &old).await.unwrap();
    assert_eq!((c.first_candidate_number, c.end_candidate_number), (Some(2), Some(4)), "an older client gets fresh work instead");
    let c = claim(&state, &new).await.unwrap();
    assert_eq!((c.first_candidate_number, c.end_candidate_number), (Some(0), Some(2)), "a newer one gets it");
    assert_eq!(c.hash_table_size, Some(1 << 20));
}

#[tokio::test]
async fn the_dashboard_shows_the_hash_table_position_and_the_false_matches() {
    let state = state(8.0).await;
    let id = create(&state, serde_json::json!({"hash_table_position": position(false)})).await.unwrap();
    let other = create(&state, serde_json::json!({"name": "other", "priority": -1})).await.unwrap();
    let client = user(&state, "new", "1.7.0").await;
    let c = claim(&state, &client).await.unwrap();
    heartbeat(&state, &client, c.range_id, &[NAME]).await.unwrap();

    let dashboard = serde_json::to_value(crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().0).unwrap();
    let card = |target_id: i64| dashboard["targets"].as_array().unwrap().iter().find(|t| t["id"] == target_id).unwrap().clone();
    let t = card(id);
    let slot = (start_slot(NAME, 2048) + 10) % 2048;
    assert_eq!(t["hash_table_position"], serde_json::json!({"table_size": 2048, "first_start_slot": slot, "last_start_slot": (slot + 2) % 2048}));
    let shown = t["false_matches"].as_array().unwrap();
    assert_eq!(shown.len(), 1);
    assert_eq!(shown[0]["filename"], NAME);
    assert_eq!(shown[0]["reported_by"], "new@host");
    assert_eq!(shown[0]["reported_by_backend"], "cuda");
    assert!(shown[0]["reason"].as_str().unwrap().starts_with("its lookup starts at hash table slot"));
    let o = card(other);
    assert_eq!((o["hash_table_position"].clone(), o["false_matches"].clone()), (serde_json::Value::Null, serde_json::json!([])));
}
