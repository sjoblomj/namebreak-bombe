//! Dictionary targets end to end, through the handlers: word lists stored
//! and downloaded, targets made and patched, their numbers claimed,
//! checkpointed, quit and completed, the basenames clients send, and what
//! the dashboard shows of it all. `dictionary.rs` tests the numbering itself.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use namebreak_protocol::{ClaimResponse, CompleteRequest, HeartbeatRequest, QuitRequest, WordListInfo, MAX_BASENAMES_PER_REPORT};

use crate::auth::{AdminAuth, AuthedUser};
use crate::dictionary;
use crate::error::AppError;
use crate::handlers;
use crate::models::{Range, Target, User};
use crate::ranges;
use crate::state::{now_unix, AppState, Inner, RangeConfig};

/// Each claim gets `chunk` candidates (rate 1 a second for `chunk` seconds),
/// whatever the client's measured rate - unless a test measures one.
fn config(chunk: f64) -> RangeConfig {
    RangeConfig {
        target_chunk_seconds: chunk,
        default_rate_per_sec: 1.0,
        default_dictionary_rate_per_sec: 1.0,
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
    AppState(std::sync::Arc::new(Inner { pool, admin_token: "t".into(), config }))
}

async fn state(chunk: f64) -> AppState {
    state_with(config(chunk)).await
}

async fn put_list(state: &AppState, name: &str, text: &[u8]) -> Result<WordListInfo, AppError> {
    handlers::admin_put_word_list(State(state.clone()), AdminAuth, Path(name.to_string()), axum::body::Bytes::copy_from_slice(text)).await.map(|json| json.0)
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
    reload_user(state, id).await
}

async fn reload_user(state: &AppState, id: i64) -> User {
    sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

/// Words A to E (and the words of `create`'s default target).
const WORDS: &[u8] = b"alpha\nbravo\ncharlie\ndelta\necho\n";

/// A dictionary target of `WORDS` - 5 one-word and 5 * 2 * 5 = 50 two-word
/// candidates, numbered 0 to 54 - with `extra` merged into the request.
async fn create(state: &AppState, extra: serde_json::Value) -> Result<i64, AppError> {
    let mut body = serde_json::json!({
        "name": "dict", "prefix": "music/", "suffix": ".wav",
        "hash_a_hex": "0x11111111", "hash_b_hex": "0x22222222",
        "dictionary": {"word_lists": ["nato"], "separators": ["", "_"], "max_words": 2},
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    let req = serde_json::from_value(body).unwrap();
    handlers::admin_create_target(State(state.clone()), AdminAuth, Json(req)).await.map(|json| json.0.target_id)
}

async fn target(state: &AppState, id: i64) -> Target {
    sqlx::query_as::<_, Target>("SELECT * FROM targets WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

async fn range(state: &AppState, id: i64) -> Range {
    sqlx::query_as::<_, Range>("SELECT * FROM ranges WHERE id = ?").bind(id).fetch_one(&state.pool).await.unwrap()
}

async fn claim(state: &AppState, user: &User) -> Option<ClaimResponse> {
    ranges::claim_range(&state.pool, &state.config, user).await.unwrap()
}

fn report(next: Option<i64>, basenames: &[&str]) -> HeartbeatRequest {
    HeartbeatRequest { last_hash_a_match_filename: None, next_candidate_number: next, basenames: basenames.iter().map(|b| b.to_string()).collect() }
}

async fn heartbeat(state: &AppState, user: &User, range_id: i64, req: HeartbeatRequest) -> Result<bool, AppError> {
    handlers::heartbeat(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await.map(|json| json.0.range_released)
}

async fn quit(state: &AppState, user: &User, range_id: i64, next: Option<i64>, basenames: &[&str]) -> Result<StatusCode, AppError> {
    let req = QuitRequest { last_hash_a_match_filename: None, next_candidate_number: next, basenames: basenames.iter().map(|b| b.to_string()).collect() };
    handlers::quit(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await
}

async fn complete(state: &AppState, user: &User, range_id: i64, found: Option<&str>, elapsed: f64, candidates: i64, basenames: &[&str]) -> Result<StatusCode, AppError> {
    let req = CompleteRequest {
        found: found.is_some(),
        filename: found.map(str::to_string),
        elapsed_seconds: elapsed,
        candidates_processed: candidates,
        basenames: basenames.iter().map(|b| b.to_string()).collect(),
    };
    handlers::complete(State(state.clone()), AuthedUser(user.clone()), Path(range_id), Json(req)).await
}

async fn basenames(state: &AppState, target_id: i64) -> Vec<(String, i64, i64)> {
    sqlx::query_as("SELECT basename, user_id, range_id FROM basenames WHERE target_id = ? ORDER BY id").bind(target_id).fetch_all(&state.pool).await.unwrap()
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn a_word_list_is_stored_once_and_never_changes() {
    let state = state(1.0).await;
    let info = put_list(&state, "nato", b"alpha\r\nBravo\n# comment\n\nalpha\ncaf\xC3\xA9\n").await.unwrap();
    assert_eq!(info.word_count, 2, "ALPHA twice, BRAVO; the comment, the blank line and the line past ASCII aren't words");
    assert_eq!(info.skipped_lines, vec![6]);
    assert_eq!(info.checksum, dictionary::hex64(dictionary::checksum(&["ALPHA".to_string(), "BRAVO".to_string()])));

    let again = put_list(&state, "nato", b"alpha\r\nBravo\n# comment\n\nalpha\ncaf\xC3\xA9\n").await.unwrap();
    assert_eq!((again.word_count, again.created_at), (2, info.created_at), "the same file again changes nothing");
    let other = put_list(&state, "nato", b"alpha\nbravo\n").await;
    assert!(matches!(other, Err(AppError::Conflict(_))), "the same words in another file are still another file");

    for bad in ["", ".hidden", "-x", "a/b", "a b", &"x".repeat(65)] {
        assert!(matches!(put_list(&state, bad, b"word\n").await, Err(AppError::BadRequest(_))), "{bad:?}");
    }
    assert!(matches!(put_list(&state, "empty", b"# nothing\n\n").await, Err(AppError::BadRequest(_))));

    let listed = handlers::admin_list_word_lists(State(state.clone()), AdminAuth).await.unwrap().0;
    assert_eq!(listed.iter().map(|l| (l.name.as_str(), l.word_count)).collect::<Vec<_>>(), vec![("english-1", 63_875), ("nato", 2)]);
    assert!(listed.iter().all(|l| l.skipped_lines.is_empty()), "only PUT says which lines were skipped");
}

#[tokio::test]
async fn english_1_is_built_in() {
    let state = state(1.0).await;
    let (word_count, checksum): (i64, String) =
        sqlx::query_as("SELECT word_count, checksum FROM word_lists WHERE name = 'english-1'").fetch_one(&state.pool).await.unwrap();
    assert_eq!((word_count, checksum.as_str()), (63_875, "63b352823c6059b0"), "there from the start, with the client's words");

    // A client downloads it as the client's own file - though it uses its own.
    let client = user(&state, "u", "1.5.0").await;
    let response = handlers::word_list(State(state.clone()), AuthedUser(client), Path("english-1".into())).await.unwrap();
    assert!(body_text(response).await.as_bytes() == include_bytes!("../../../client/data/english-1.txt"));

    // A target uses it without uploading anything.
    let id = create(&state, serde_json::json!({"dictionary": {"word_lists": ["english-1"], "separators": [""], "max_words": 1}})).await.unwrap();
    assert!(target(&state, id).await.is_dictionary());

    // The english- names can't be uploaded or deleted - not even the same file.
    let english_1 = include_bytes!("../../../client/data/english-1.txt");
    assert!(matches!(put_list(&state, "english-1", english_1).await, Err(AppError::BadRequest(_))));
    assert!(matches!(put_list(&state, "english-2", b"other\nwords\n").await, Err(AppError::BadRequest(_))), "the english- names are kept for built-in ones");
    assert!(put_list(&state, "english", b"other\nwords\n").await.is_ok(), "but 'english' alone is anyone's");
    ranges::delete_target(&state.pool, id).await.unwrap();
    let delete = handlers::admin_delete_word_list(State(state.clone()), AdminAuth, Path("english-1".into())).await;
    assert!(matches!(delete, Err(AppError::BadRequest(_))), "unused, but built in");

    // Connecting again - a restart - leaves it as it is.
    crate::dictionary::store_built_in(&state.pool).await.unwrap();
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM word_lists WHERE name = 'english-1'").fetch_one(&state.pool).await.unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn a_client_downloads_a_word_list_as_it_was_uploaded() {
    let state = state(1.0).await;
    put_list(&state, "nato", b"alpha\r\nbravo\n").await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let response = handlers::word_list(State(state.clone()), AuthedUser(client.clone()), Path("nato".into())).await.unwrap();
    assert_eq!(body_text(response).await, "alpha\r\nbravo\n");
    assert!(matches!(handlers::word_list(State(state.clone()), AuthedUser(client), Path("other".into())).await, Err(AppError::NotFound)));
}

#[tokio::test]
async fn a_word_list_a_target_uses_cant_be_deleted() {
    let state = state(1.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    put_list(&state, "spare", b"x\n").await.unwrap();
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let delete = |name: &str| handlers::admin_delete_word_list(State(state.clone()), AdminAuth, Path(name.to_string()));
    let err = delete("nato").await.unwrap_err();
    assert!(matches!(&err, AppError::Conflict(msg) if msg.contains(&id.to_string())), "{err:?}");
    assert_eq!(delete("spare").await.unwrap(), StatusCode::NO_CONTENT);
    assert!(matches!(delete("spare").await, Err(AppError::NotFound)));

    ranges::delete_target(&state.pool, id).await.unwrap();
    assert_eq!(delete("nato").await.unwrap(), StatusCode::NO_CONTENT, "once the target is gone");
}

#[tokio::test]
async fn a_dictionary_target_is_stored_normalized_and_starts_at_its_first_window() {
    let state = state(1.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(
        &state,
        serde_json::json!({"lower_bound": "music/charlie", "upper_bound": "MUSIC\\ECHO.WAV", "encryption_key_hex": "0xABCDEF01", "send_basenames": true}),
    )
    .await
    .unwrap();
    let t = target(&state, id).await;
    assert!(t.is_dictionary());
    assert_eq!((t.prefix.as_str(), t.suffix.as_str()), ("MUSIC\\", ".WAV"));
    assert_eq!((t.lower_bound.as_str(), t.upper_bound.as_str()), ("MUSIC\\CHARLIE", "MUSIC\\ECHO.WAV"));
    assert_eq!(dictionary::parse_string_list(t.separators.as_deref()).unwrap(), vec!["".to_string(), "_".to_string()]);
    assert_eq!((t.min_words, t.max_words, t.send_basenames), (Some(1), Some(2), 1));
    assert_eq!((t.alphabet_name.as_str(), t.alphabet.as_str()), ("dictionary", ""));

    // CHARLIE is word 2: one-word candidates 2 to 4.
    let (len, next): (i64, i64) = sqlx::query_as("SELECT candidate_len, next_index FROM target_progress WHERE target_id = ?").bind(id).fetch_one(&state.pool).await.unwrap();
    assert_eq!((len, next), (1, 2));
}

#[tokio::test]
async fn a_dictionary_target_has_to_make_sense() {
    let state = state(1.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let refused = |extra: serde_json::Value| {
        let state = state.clone();
        async move {
            match create(&state, extra.clone()).await {
                Err(AppError::BadRequest(msg)) => msg,
                other => panic!("{extra} wasn't refused: {other:?}"),
            }
        }
    };
    assert!(refused(serde_json::json!({"alphabet_name": "size42", "prune_symbol_runs": true})).await.contains("alphabet_name, prune_symbol_runs isn't for a dictionary target"));
    assert!(refused(serde_json::json!({"start_len": 3})).await.contains("start_len"));
    assert!(refused(serde_json::json!({"insert_from_end": ["\\", 2]})).await.contains("insert_from_end"));
    assert!(refused(serde_json::json!({"send_basenames": true})).await.contains("encryption_key_hex"));
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": ["nope"], "separators": [""], "max_words": 1}})).await.contains("no word list named 'nope'"));
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": ["nato", "nato"], "separators": [""], "max_words": 1}})).await.contains("more than once"));
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": [], "separators": [""], "max_words": 1}})).await.contains("at least one word list"));
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": ["nato"], "separators": ["x", "X"], "max_words": 1}})).await.contains("more than once"), "the same once normalized");
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": ["nato"], "separators": [], "max_words": 1}})).await.contains("no separators"));
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": ["nato"], "separators": [""], "max_words": 9}})).await.contains("min_words"));
    assert!(refused(serde_json::json!({"dictionary": {"word_lists": ["nato"], "separators": ["\t"], "max_words": 1}})).await.contains("printable ASCII"));
    assert!(refused(serde_json::json!({"prefix": "caf\u{e9}\\"})).await.contains("printable ASCII"));
    assert!(refused(serde_json::json!({"lower_bound": "MUSIC\\Z", "upper_bound": "MUSIC\\A"})).await.contains("sorts after"));
    assert!(refused(serde_json::json!({"lower_bound": "SOUND\\"})).await.contains("no candidate is within the bounds"));
    assert!(refused(serde_json::json!({"hash_a_hex": "xyz"})).await.contains("hash_a_hex"));

    // An alphabet target doesn't compare basenames at all.
    let alphabet_target = serde_json::json!({
        "name": "a", "prefix": "REZ\\", "suffix": ".WAV", "hash_a_hex": "1", "hash_b_hex": "2",
        "lower_bound": "A", "upper_bound": "B", "encryption_key_hex": "3", "send_basenames": true,
    });
    let result = handlers::admin_create_target(State(state.clone()), AdminAuth, Json(serde_json::from_value(alphabet_target).unwrap())).await;
    assert!(matches!(result, Err(AppError::BadRequest(msg)) if msg.contains("dictionary targets")));
}

#[tokio::test]
async fn too_many_candidates_are_refused() {
    let state = state(1.0).await;
    let words: String = (0..1000).map(|i| format!("w{i}\n")).collect();
    put_list(&state, "thousand", words.as_bytes()).await.unwrap();
    let result = create(&state, serde_json::json!({"dictionary": {"word_lists": ["thousand"], "separators": ["", "_", "-", " "], "max_words": 7}})).await;
    assert!(matches!(result, Err(AppError::BadRequest(msg)) if msg.contains("more candidates than the server can number")));
}

#[tokio::test]
async fn only_clients_that_know_dictionary_targets_get_their_work() {
    let state = state(1000.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({})).await.unwrap();
    let old = user(&state, "old", "1.4.0").await;
    assert!(claim(&state, &old).await.is_none());
    let new = user(&state, "new", "1.5.0").await;
    let c = claim(&state, &new).await.expect("a 1.5 client gets it");
    assert!(c.dictionary);
}

#[tokio::test]
async fn a_claim_says_everything_the_client_needs() {
    let state = state(3.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    put_list(&state, "extra", b"bravo\nfoxtrot\n").await.unwrap();
    let id = create(
        &state,
        serde_json::json!({
            "dictionary": {"word_lists": ["nato", "extra"], "separators": ["", "-"], "min_words": 1, "max_words": 2},
            "lower_bound": "MUSIC\\B", "encryption_key_hex": "0x0000ABCD", "send_basenames": true,
        }),
    )
    .await
    .unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    assert_eq!(c.target_id, id);
    let merged: Vec<String> = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"].iter().map(|s| s.to_string()).collect();
    assert_eq!(c.word_lists, Some(vec!["nato".to_string(), "extra".to_string()]));
    let list_checksum = |words: &[&str]| dictionary::hex64(dictionary::checksum(&words.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
    assert_eq!(c.word_list_checksums, Some(vec![list_checksum(&["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]), list_checksum(&["BRAVO", "FOXTROT"])]));
    assert_eq!(c.words_checksum, Some(dictionary::hex64(dictionary::checksum(&merged))));
    assert_eq!(c.separators, Some(vec!["".to_string(), "-".to_string()]));
    assert_eq!((c.min_words, c.max_words), (Some(1), Some(2)));
    // BRAVO (word 1) is the first word the lower bound lets in.
    assert_eq!((c.first_candidate_number, c.end_candidate_number, c.candidate_count), (Some(1), Some(4), 3));
    assert_eq!((c.lower_bound_filename.as_str(), c.upper_bound_filename.as_str()), ("MUSIC\\BRAVO.WAV", "MUSIC\\DELTA.WAV"));
    assert_eq!((c.filename_lower_bound.as_deref(), c.filename_upper_bound.as_deref()), (Some("MUSIC\\B"), None));
    assert!(c.send_basenames);
    assert_eq!(c.encryption_key_hex.as_deref(), Some("0x0000ABCD"));
    assert_eq!((c.prefix.as_str(), c.suffix.as_str(), c.alphabet.as_str()), ("MUSIC\\", ".WAV", ""));

    // Without send_basenames, the key isn't sent - nor compared.
    handlers::admin_patch_target(State(state.clone()), AdminAuth, Path(id), Json(serde_json::from_value(serde_json::json!({"send_basenames": false})).unwrap()))
        .await
        .unwrap();
    let c = claim(&state, &client).await.unwrap();
    assert!(!c.send_basenames);
    assert_eq!(c.encryption_key_hex, None);
}

#[test]
fn an_alphabet_targets_claim_has_nothing_an_older_client_cant_read() {
    // Clients before 1.5 read a claim with a parser that takes no arrays,
    // nor needs any of the dictionary fields.
    let claim = serde_json::json!({
        "range_id": 1, "target_id": 1, "target_name": "t", "prefix": "", "suffix": "", "hash_a_hex": "0x1", "hash_b_hex": "0x2",
        "prune_symbol_runs": false, "max_backslash_count": 0, "lower_bound_filename": "A", "upper_bound_filename": "B",
        "alphabet": "AB", "candidate_count": 2, "lease_seconds": 60,
    });
    let parsed: ClaimResponse = serde_json::from_value(claim).unwrap();
    assert!(!parsed.dictionary);
    let text = serde_json::to_string(&parsed).unwrap();
    for key in ["dictionary", "word_lists", "word_list_checksums", "words_checksum", "separators", "min_words", "max_words", "first_candidate_number",
        "end_candidate_number", "filename_lower_bound", "filename_upper_bound", "send_basenames", "encryption_key_hex"]
    {
        assert!(!text.contains(&format!("\"{key}\"")), "{key} in {text}");
    }
    assert!(!text.contains('['), "no arrays: {text}");
}

#[tokio::test]
async fn claims_cover_every_window_once_and_never_span_two() {
    let state = state(7.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    // DELTA to ECHO_B...: one-word 3..5, two-word children (DELTA, "")
    // to (ECHO, "_") - 10 + 3 * 5 = 25 + ... see below.
    let id = create(&state, serde_json::json!({"lower_bound": "MUSIC\\DELTA", "upper_bound": "MUSIC\\ECHO_BRAVO.WAV"})).await.unwrap();
    let windows = {
        let t = target(&state, id).await;
        let mut conn = state.pool.acquire().await.unwrap();
        state.config.dictionaries.get(&mut conn, &t).await.unwrap().windows.clone()
    };
    assert_eq!(windows, vec![dictionary::Window { words: 1, start: 3, end: 5 }, dictionary::Window { words: 2, start: 5 + 3 * 2 * 5, end: 5 + 5 * 2 * 5 }]);

    let client = user(&state, "u", "1.5.0").await;
    let mut covered = Vec::new();
    while let Some(c) = claim(&state, &client).await {
        let (start, end) = (c.first_candidate_number.unwrap(), c.end_candidate_number.unwrap());
        assert!(end - start <= 7);
        let window = windows.iter().find(|w| w.start <= start && end <= w.end).unwrap_or_else(|| panic!("[{start}, {end}) isn't within one window"));
        assert_eq!(range(&state, c.range_id).await.candidate_len, window.words, "a range's candidate_len is its words");
        covered.extend(start..end);
        complete(&state, &client, c.range_id, None, 0.0, 0, &[]).await.unwrap();
    }
    let expected: Vec<i64> = windows.iter().flat_map(|w| w.start..w.end).collect();
    assert_eq!(covered, expected);
}

#[tokio::test]
async fn heartbeats_checkpoint_by_candidate_number() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({"dictionary": {"word_lists": ["nato"], "separators": ["", "_"], "min_words": 2, "max_words": 2}})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    assert_eq!((c.first_candidate_number, c.end_candidate_number), (Some(0), Some(20)));
    let progress = |r: Range| r.progress().map(|p| p as i64);

    heartbeat(&state, &client, c.range_id, report(Some(0), &[])).await.unwrap();
    assert_eq!(progress(range(&state, c.range_id).await), None, "nothing searched yet");
    heartbeat(&state, &client, c.range_id, report(Some(7), &[])).await.unwrap();
    assert_eq!(progress(range(&state, c.range_id).await), Some(6), "the last candidate searched");
    heartbeat(&state, &client, c.range_id, report(Some(3), &[])).await.unwrap();
    assert_eq!(progress(range(&state, c.range_id).await), Some(6), "never backwards");
    heartbeat(&state, &client, c.range_id, report(Some(21), &[])).await.unwrap();
    assert_eq!(progress(range(&state, c.range_id).await), Some(6), "past the end: ignored");
    let filename_only = HeartbeatRequest { last_hash_a_match_filename: Some("MUSIC\\ECHO_ECHO.WAV".into()), next_candidate_number: None, basenames: vec![] };
    heartbeat(&state, &client, c.range_id, filename_only).await.unwrap();
    assert_eq!(progress(range(&state, c.range_id).await), Some(6), "a dictionary range goes by numbers only");
    heartbeat(&state, &client, c.range_id, report(Some(20), &[])).await.unwrap();
    assert_eq!(progress(range(&state, c.range_id).await), Some(19), "all of it");
}

#[tokio::test]
async fn a_range_with_no_progress_for_too_long_is_released() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    let long_ago = now_unix() - 601;
    sqlx::query("UPDATE ranges SET last_progress_at = ? WHERE id = ?").bind(long_ago).bind(c.range_id).execute(&state.pool).await.unwrap();
    assert!(!heartbeat(&state, &client, c.range_id, report(Some(2), &[])).await.unwrap(), "progress keeps it");
    sqlx::query("UPDATE ranges SET last_progress_at = ? WHERE id = ?").bind(long_ago).bind(c.range_id).execute(&state.pool).await.unwrap();
    assert!(heartbeat(&state, &client, c.range_id, report(Some(2), &[])).await.unwrap(), "the same number again is no progress");
}

#[tokio::test]
async fn quitting_hands_the_rest_out_again() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({})).await.unwrap();
    let first = user(&state, "first", "1.5.0").await;
    let c = claim(&state, &first).await.unwrap();
    assert_eq!((c.first_candidate_number, c.end_candidate_number), (Some(0), Some(5)), "the one-word window");
    let c = claim(&state, &first).await.unwrap();
    assert_eq!((c.first_candidate_number, c.end_candidate_number), (Some(5), Some(25)));
    assert_eq!(quit(&state, &first, c.range_id, Some(12), &[]).await.unwrap(), StatusCode::NO_CONTENT);

    let second = user(&state, "second", "1.5.0").await;
    let rest = claim(&state, &second).await.unwrap();
    assert_eq!((rest.first_candidate_number, rest.end_candidate_number), (Some(12), Some(25)), "from the first candidate not searched");
    let searched = range(&state, c.range_id).await;
    assert_eq!((searched.status.as_str(), searched.start() as i64, searched.end() as i64), ("completed", 5, 12));
    assert_eq!(searched.last_assigned_user_id, Some(first.id), "credited to whoever searched it");
    assert_eq!(rest.lower_bound_filename, "MUSIC\\ALPHA_CHARLIE.WAV", "number 12: two words, the 8th - A, '_', C");

    // Without a checkpoint, the whole range goes back.
    assert_eq!(quit(&state, &second, rest.range_id, None, &[]).await.unwrap(), StatusCode::NO_CONTENT);
    let again = claim(&state, &first).await.unwrap();
    assert_eq!((again.range_id, again.first_candidate_number, again.end_candidate_number), (rest.range_id, Some(12), Some(25)));
}

#[tokio::test]
async fn a_dictionary_search_has_its_own_rate() {
    let mut config = config(10.0);
    config.default_rate_per_sec = 1000.0;
    let state = state_with(config).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({"dictionary": {"word_lists": ["nato"], "separators": ["", "_"], "min_words": 2, "max_words": 2}})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    assert_eq!(c.candidate_count, 10, "the default dictionary rate (1 a second), not the alphabet one");
    complete(&state, &client, c.range_id, None, 4.0, 10, &[]).await.unwrap();
    let client = reload_user(&state, client.id).await;
    assert_eq!((client.ema_dictionary_rate_per_sec, client.ema_rate_per_sec), (Some(2.5), None));
    let c = claim(&state, &client).await.unwrap();
    assert_eq!(c.candidate_count, 25, "2.5 a second for 10 seconds");
    complete(&state, &client, c.range_id, None, 5.0, 25, &[]).await.unwrap();
    let client = reload_user(&state, client.id).await;
    assert_eq!(client.ema_dictionary_rate_per_sec, Some(0.5 * 5.0 + 0.5 * 2.5));
}

#[tokio::test]
async fn a_find_solves_the_target() {
    let state = state(100.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    complete(&state, &client, c.range_id, Some("MUSIC\\CHARLIE.WAV"), 1.0, 5, &[]).await.unwrap();
    let t = target(&state, id).await;
    assert_eq!((t.status.as_str(), t.found_filename.as_deref()), ("solved", Some("MUSIC\\CHARLIE.WAV")));
    assert!(claim(&state, &client).await.is_none());
}

#[tokio::test]
async fn basenames_are_kept_once_per_target_from_every_kind_of_report() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({"encryption_key_hex": "0x1234", "send_basenames": true})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let one = claim(&state, &client).await.unwrap();
    let two = claim(&state, &client).await.unwrap();

    heartbeat(&state, &client, one.range_id, report(Some(1), &["A.WAV", "B.WAV"])).await.unwrap();
    heartbeat(&state, &client, one.range_id, report(Some(2), &["B.WAV", "C.WAV", "", "D\\E.WAV", "caf\u{e9}.wav", &"X".repeat(256)])).await.unwrap();
    complete(&state, &client, one.range_id, None, 1.0, 5, &["D.WAV"]).await.unwrap();
    quit(&state, &client, two.range_id, None, &["E.WAV", "A.WAV"]).await.unwrap();
    assert_eq!(
        basenames(&state, id).await,
        vec![
            ("A.WAV".into(), client.id, one.range_id),
            ("B.WAV".into(), client.id, one.range_id),
            ("C.WAV".into(), client.id, one.range_id),
            ("D.WAV".into(), client.id, one.range_id),
            ("E.WAV".into(), client.id, two.range_id),
        ],
        "each once, with its first report; what can't be a basename is left out"
    );

    let response = handlers::target_basenames(State(state.clone()), Path(id)).await.unwrap();
    assert_eq!(body_text(response).await, "A.WAV\nB.WAV\nC.WAV\nD.WAV\nE.WAV\n");
    assert!(matches!(handlers::target_basenames(State(state.clone()), Path(id + 1)).await, Err(AppError::NotFound)));

    ranges::delete_target(&state.pool, id).await.unwrap();
    assert!(basenames(&state, id).await.is_empty(), "deleted with their target");
}

#[tokio::test]
async fn basenames_are_kept_even_when_the_report_is_refused() {
    // The client counts them as delivered on a 409, so the server must
    // have kept them.
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({"encryption_key_hex": "0x1234", "send_basenames": true})).await.unwrap();
    let first = user(&state, "first", "1.5.0").await;
    let c = claim(&state, &first).await.unwrap();
    // The lease runs out and someone else gets the range.
    sqlx::query("UPDATE ranges SET lease_expires_at = 0 WHERE id = ?").bind(c.range_id).execute(&state.pool).await.unwrap();
    ranges::reclaim_expired(&state.pool).await.unwrap();
    let second = user(&state, "second", "1.5.0").await;
    assert_eq!(claim(&state, &second).await.unwrap().range_id, c.range_id);

    assert!(matches!(heartbeat(&state, &first, c.range_id, report(Some(1), &["LATE.WAV"])).await, Err(AppError::Conflict(_))));
    assert!(matches!(quit(&state, &first, c.range_id, None, &["LATER.WAV"]).await, Err(AppError::Conflict(_))));
    assert!(matches!(complete(&state, &first, c.range_id, None, 1.0, 5, &["LATEST.WAV"]).await, Err(AppError::Conflict(_))));
    let kept: Vec<String> = basenames(&state, id).await.into_iter().map(|b| b.0).collect();
    assert_eq!(kept, vec!["LATE.WAV", "LATER.WAV", "LATEST.WAV"]);

    // But not for a range there's no such thing as.
    assert!(matches!(heartbeat(&state, &first, 9999, report(None, &["X.WAV"])).await, Err(AppError::NotFound)));
}

#[tokio::test]
async fn a_targets_basenames_are_counted_as_they_come() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let one = create(&state, serde_json::json!({"encryption_key_hex": "0x1234", "send_basenames": true})).await.unwrap();
    let two = create(&state, serde_json::json!({"name": "two", "encryption_key_hex": "0x5678", "send_basenames": true, "priority": -1})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    assert_eq!(c.target_id, one);
    heartbeat(&state, &client, c.range_id, report(None, &["A.WAV", "B.WAV", "A.WAV"])).await.unwrap();
    heartbeat(&state, &client, c.range_id, report(None, &["B.WAV", "C.WAV", "D\\E.WAV"])).await.unwrap();
    assert_eq!(target(&state, one).await.basename_count, 3, "each once, and only those kept");
    assert_eq!(target(&state, two).await.basename_count, 0, "a target's own");
    let dashboard = crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().0;
    let shown = |id: i64| {
        let d = dashboard.targets.iter().find(|t| t.id == id).unwrap().dictionary.as_ref().unwrap();
        (d.basename_count, d.basenames.iter().map(|b| b.basename.clone()).collect::<Vec<_>>())
    };
    assert_eq!(shown(one), (3, vec!["C.WAV".to_string(), "B.WAV".to_string(), "A.WAV".to_string()]));
    assert_eq!(shown(two), (0, vec![]));
}

#[tokio::test]
async fn every_basename_is_listed_a_page_at_a_time() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({"encryption_key_hex": "0x1234", "send_basenames": true})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let listed = |state: AppState| async move { body_text(handlers::target_basenames(State(state), Path(id)).await.unwrap()).await };
    assert_eq!(listed(state.clone()).await, "", "none yet");

    // Two pages and a bit, then exactly three: the last page full, and then
    // none after it.
    let page = handlers::BASENAMES_PAGE as usize;
    for (count, label) in [(2 * page + 7, "two pages and a bit"), (3 * page, "exactly three pages")] {
        sqlx::query("DELETE FROM basenames").execute(&state.pool).await.unwrap();
        let mut tx = state.pool.begin().await.unwrap();
        for i in 0..count {
            // Not in name order, so that it's the order they came in that shows.
            sqlx::query("INSERT INTO basenames (target_id, basename, user_id, range_id, reported_at) VALUES (?, ?, ?, 0, 0)")
                .bind(id)
                .bind(format!("N{:06}.WAV", (i * 7919) % count))
                .bind(client.id)
                .execute(&mut *tx)
                .await
                .unwrap();
        }
        tx.commit().await.unwrap();
        let expected: String = (0..count).map(|i| format!("N{:06}.WAV\n", (i * 7919) % count)).collect();
        assert!(listed(state.clone()).await == expected, "{label}: every one, once, in the order they came");
    }
}

#[tokio::test]
async fn too_many_basenames_at_once_are_refused() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({"encryption_key_hex": "0x1234", "send_basenames": true})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    let many: Vec<String> = (0..=MAX_BASENAMES_PER_REPORT).map(|i| format!("N{i}.WAV")).collect();
    let req = HeartbeatRequest { last_hash_a_match_filename: None, next_candidate_number: None, basenames: many.clone() };
    assert!(matches!(heartbeat(&state, &client, c.range_id, req).await, Err(AppError::BadRequest(_))));
    assert!(basenames(&state, id).await.is_empty());
    let req = HeartbeatRequest { last_hash_a_match_filename: None, next_candidate_number: None, basenames: many[1..].to_vec() };
    heartbeat(&state, &client, c.range_id, req).await.unwrap();
    assert_eq!(basenames(&state, id).await.len(), MAX_BASENAMES_PER_REPORT, "as many as are allowed");
}

#[tokio::test]
async fn an_alphabet_target_keeps_no_basenames() {
    let state = state(20.0).await;
    let alphabet_target = serde_json::json!({
        "name": "a", "prefix": "REZ\\", "suffix": ".WAV", "hash_a_hex": "1", "hash_b_hex": "2", "lower_bound": "A", "upper_bound": "B", "alphabet_name": "size42",
    });
    let id = handlers::admin_create_target(State(state.clone()), AdminAuth, Json(serde_json::from_value(alphabet_target).unwrap())).await.unwrap().0.target_id;
    let client = user(&state, "u", "1.5.0").await;
    let c = claim(&state, &client).await.unwrap();
    assert!(!c.dictionary);
    heartbeat(&state, &client, c.range_id, report(None, &["A.WAV"])).await.unwrap();
    assert!(basenames(&state, id).await.is_empty());
}

#[tokio::test]
async fn patching_a_dictionary_target() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({})).await.unwrap();
    let patch = |body: serde_json::Value| {
        let state = state.clone();
        async move { handlers::admin_patch_target(State(state), AdminAuth, Path(id), Json(serde_json::from_value(body).unwrap())).await }
    };
    assert!(matches!(patch(serde_json::json!({"send_basenames": true})).await, Err(AppError::BadRequest(msg)) if msg.contains("encryption_key_hex")));
    patch(serde_json::json!({"send_basenames": true, "encryption_key_hex": "0x99"})).await.unwrap();
    assert_eq!(target(&state, id).await.send_basenames, 1);
    assert!(matches!(patch(serde_json::json!({"encryption_key_hex": null})).await, Err(AppError::BadRequest(_))), "not without the key it needs");
    patch(serde_json::json!({"send_basenames": false, "encryption_key_hex": null})).await.unwrap();
    let t = target(&state, id).await;
    assert_eq!((t.send_basenames, t.encryption_key), (0, None));
    patch(serde_json::json!({"name": "renamed", "priority": 3, "status": "paused", "description": "<b>words</b>"})).await.unwrap();
    let t = target(&state, id).await;
    assert_eq!((t.name.as_str(), t.priority, t.status.as_str()), ("renamed", 3, "paused"));

    for field in [
        serde_json::json!({"alphabet_name": "size42"}),
        serde_json::json!({"prune_whole_candidate": true}),
        serde_json::json!({"start_len": 2}),
        serde_json::json!({"insert_from_start": null}),
        serde_json::json!({"auto_priority": false}),
    ] {
        assert!(matches!(patch(field.clone()).await, Err(AppError::BadRequest(msg)) if msg.contains("isn't for a dictionary target")), "{field}");
    }

    let priority = serde_json::json!({"pattern": "A", "length": 3, "priority": 1});
    let result = handlers::admin_create_priority_range(State(state.clone()), AdminAuth, Path(id), Json(serde_json::from_value(priority).unwrap())).await;
    assert!(matches!(result, Err(AppError::BadRequest(_))));
    let skip = serde_json::json!({"pattern": "A", "length": 3, "reason": "no"});
    let result = handlers::admin_create_skip_range(State(state.clone()), AdminAuth, Path(id), Json(serde_json::from_value(skip).unwrap())).await;
    assert!(matches!(result, Err(AppError::BadRequest(_))));
}

#[tokio::test]
async fn an_alphabet_target_cant_send_basenames() {
    let state = state(20.0).await;
    let alphabet_target = serde_json::json!({"name": "a", "prefix": "", "suffix": "", "hash_a_hex": "1", "hash_b_hex": "2", "lower_bound": "A", "upper_bound": "B"});
    let id = handlers::admin_create_target(State(state.clone()), AdminAuth, Json(serde_json::from_value(alphabet_target).unwrap())).await.unwrap().0.target_id;
    let body = serde_json::json!({"send_basenames": true, "encryption_key_hex": "0x1"});
    let result = handlers::admin_patch_target(State(state.clone()), AdminAuth, Path(id), Json(serde_json::from_value(body).unwrap())).await;
    assert!(matches!(result, Err(AppError::BadRequest(msg)) if msg.contains("dictionary targets")));
    let body = serde_json::json!({"send_basenames": false});
    assert!(handlers::admin_patch_target(State(state.clone()), AdminAuth, Path(id), Json(serde_json::from_value(body).unwrap())).await.is_ok());
}

#[tokio::test]
async fn no_canary_is_made_from_a_dictionary_target() {
    let mut config = config(20.0);
    config.canary_probability = 1.0;
    let state = state_with(config).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({"priority": 10})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    sqlx::query("UPDATE users SET ema_rate_per_sec = 1000 WHERE id = ?").bind(client.id).execute(&state.pool).await.unwrap();
    let client = reload_user(&state, client.id).await;
    let c = claim(&state, &client).await.unwrap();
    assert!(c.dictionary, "with no alphabet target to copy, no canary: the dictionary target's work");

    // With one, a canary copies it - however much higher the dictionary
    // target's priority.
    let alphabet_target = serde_json::json!({
        "name": "a", "prefix": "REZ\\", "suffix": ".WAV", "hash_a_hex": "1", "hash_b_hex": "2", "lower_bound": "A", "upper_bound": "B", "alphabet_name": "size42",
    });
    let _ = handlers::admin_create_target(State(state.clone()), AdminAuth, Json(serde_json::from_value(alphabet_target).unwrap())).await.unwrap();
    let c = claim(&state, &client).await.unwrap();
    assert_eq!((c.target_name.as_str(), c.dictionary, c.prefix.as_str()), (crate::canary::CANARY_TARGET_NAME, false, "REZ\\"));
}

#[tokio::test]
async fn the_dashboard_shows_a_dictionary_target() {
    let state = state(10.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    let id = create(&state, serde_json::json!({"encryption_key_hex": "0x1234", "send_basenames": true, "upper_bound": "MUSIC\\DELTA_ECHO.WAV"})).await.unwrap();
    let client = user(&state, "u", "1.5.0").await;
    let one = claim(&state, &client).await.unwrap();
    let two = claim(&state, &client).await.unwrap();
    let three = claim(&state, &client).await.unwrap();
    heartbeat(&state, &client, two.range_id, report(Some(10), &[])).await.unwrap();
    complete(&state, &client, one.range_id, None, 1.0, 5, &(0..25).map(|i| format!("B{i:02}.WAV")).collect::<Vec<_>>().iter().map(String::as_str).collect::<Vec<_>>())
        .await
        .unwrap();
    // A range lost for good leaves a gap the dashboard shows.
    sqlx::query("DELETE FROM ranges WHERE id = ?").bind(two.range_id).execute(&state.pool).await.unwrap();

    let dashboard = crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().0;
    let t = dashboard.targets.iter().find(|t| t.id == id).unwrap();
    assert_eq!(t.kind, "dictionary");
    assert_eq!((t.alphabet_name.as_str(), t.lower_bound.as_str(), t.upper_bound.as_str()), ("", "", "MUSIC\\DELTA_ECHO.WAV"));
    let d = t.dictionary.as_ref().unwrap();
    assert_eq!(d.word_lists, vec!["nato".to_string()]);
    assert_eq!((d.word_count, d.min_words, d.max_words, d.send_basenames), (5, 1, 2, true));
    assert_eq!(d.separators, vec!["".to_string(), "_".to_string()]);
    // One-word: all 5 (DELTA, ECHO... ECHO.WAV sorts after DELTA_ECHO.WAV,
    // but the one-word window runs to its last word within, DELTA).
    assert_eq!((d.total_count, d.candidate_count), (55, 4 + 4 * 2 * 5));
    assert_eq!(d.basename_count, 25);
    assert_eq!(d.basenames.len(), 20, "only the latest");
    assert_eq!(d.basenames[0].basename, "B24.WAV", "newest first");
    assert_eq!(d.basenames[0].reported_by.as_deref(), Some("u@host"));

    assert_eq!(t.ranges.len(), 2);
    assert_eq!((t.ranges[0].first_candidate.as_str(), t.ranges[0].last_candidate.as_str(), t.ranges[0].candidate_len), ("ALPHA", "DELTA", 1));
    let last = &t.ranges[1];
    assert_eq!((last.id, last.candidate_len, last.status.as_str()), (three.range_id, 2, "in_progress"));
    assert_eq!(last.first_candidate, "BRAVOALPHA", "number 15: two words, the 11th");
    assert_eq!(t.gaps.len(), 1);
    let gap = &t.gaps[0];
    assert_eq!((gap.after_range_id, gap.first_candidate.as_str(), gap.last_candidate.as_str(), gap.count), (one.range_id, "ALPHAALPHA", "ALPHA_ECHO", 10));
    assert_eq!((t.cursor_candidate_len, t.cursor_candidate.as_str(), t.sweep_after_range_id), (2, "CHARLIEALPHA", Some(three.range_id)), "number 25");

    // Once everything has been handed out, the cursor is "".
    while claim(&state, &client).await.is_some() {}
    let dashboard = crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().0;
    let t = dashboard.targets.iter().find(|t| t.id == id).unwrap();
    assert_eq!(t.cursor_candidate, "");
    assert_eq!(t.sweep_after_range_id, t.ranges.last().map(|r| r.id));

    // And the volunteers' counts include the dictionary ranges.
    let volunteer = dashboard.volunteers.iter().find(|v| v.username == "u").unwrap();
    assert_eq!((volunteer.ranges_completed, volunteer.candidates), (1, 4));
}

#[tokio::test]
async fn the_dashboard_response_is_json_the_page_can_read() {
    let state = state(20.0).await;
    put_list(&state, "nato", WORDS).await.unwrap();
    create(&state, serde_json::json!({})).await.unwrap();
    let response = crate::dashboard::dashboard_data(State(state.clone())).await.unwrap().into_response();
    let json: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
    let t = &json["targets"][0];
    assert_eq!(t["kind"], "dictionary");
    assert_eq!(t["dictionary"]["total_count"], "55", "counts as decimal strings, as every count is");
    assert_eq!(t["dictionary"]["basenames"], serde_json::json!([]));
}
