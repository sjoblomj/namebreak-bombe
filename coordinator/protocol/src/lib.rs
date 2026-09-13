//! Wire types used by the coordinator server - an identical shape needs to be
//! implemented by the client.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub hostname: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub user_id: i64,
    pub token: String,
}

/// A contiguous, ready-to-run slice of one target's search space, handed to a client.
/// `lower_bound_filename`/`upper_bound_filename` are both inclusive and can be passed
/// directly as the `namebreak bounded` CLI's lowerBound/upperBound arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResponse {
    pub range_id: i64,
    pub target_id: i64,
    pub target_name: String,
    pub prefix: String,
    pub suffix: String,
    pub hash_a_hex: String,
    pub hash_b_hex: String,
    pub prune_symbol_runs: bool,
    /// Max '\' occurrences namebreak will allow in a candidate; 0 means unlimited.
    pub max_backslash_count: i64,
    pub lower_bound_filename: String,
    pub upper_bound_filename: String,
    /// The literal alphabet characters for this range's target.
    pub alphabet: String,
    /// Number of candidates covered by this range - lets the client report
    /// throughput on completion without doing any index math itself.
    pub candidate_count: i64,
    pub lease_seconds: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HeartbeatRequest {
    /// Full filename (prefix+candidate+suffix) of the most recent partial (Hash A
    /// only) match `namebreak` has printed for this range so far, if any. The
    /// server uses this as a progress checkpoint: everything up to and including
    /// this candidate is known to have been searched (namebreak only logs a match
    /// after the CUDA batch containing it has finished), so if this range is later
    /// reassigned, the new client resumes just past it instead of from the start.
    #[serde(default)]
    pub last_hash_a_match_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub lease_seconds: i64,
    /// True if this range's target has already been solved - by someone
    /// else via a different range, or because a *different target* sharing
    /// the same hash_a/hash_b was solved instead (see `ranges::complete_range`).
    /// The client should move on to a new range rather than let it keep
    /// searching a target that's already found - it won't be reporting
    /// completion for this range either way.
    pub target_solved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteRequest {
    pub found: bool,
    pub filename: Option<String>,
    pub elapsed_seconds: f64,
    pub candidates_processed: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub targets: Vec<TargetStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetStatus {
    pub id: i64,
    pub name: String,
    pub status: String,
    pub found_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreateTargetRequest {
    pub name: String,
    pub prefix: String,
    pub suffix: String,
    pub hash_a_hex: String,
    pub hash_b_hex: String,
    /// The candidate portion only - no prefix/suffix (unlike `ClaimResponse`'s
    /// bound fields, which are full filenames). The server always starts
    /// searching at the very beginning `lower_bound` implies and always
    /// searches as long as the chosen alphabet supports, tightened to exactly
    /// this alphabetical range at every candidate length in between (the two
    /// bounds don't need to be the same length as each other - e.g.
    /// "BLACKSMITH" to "CATAPULT" is valid).
    pub lower_bound: String,
    pub upper_bound: String,
    #[serde(default)]
    pub prune_symbol_runs: bool,
    /// One of the names from `GET /api/v1/alphabets`. Defaults to `"size49"`.
    #[serde(default)]
    pub alphabet_name: Option<String>,
    /// Max '\' occurrences allowed in a candidate; 0 (the default when omitted)
    /// means unlimited.
    #[serde(default)]
    pub max_backslash_count: i64,
    /// Higher-priority active targets have their claimable work handed out
    /// first, ahead of any lower-priority target's. Defaults to 0 when
    /// omitted, so an unset target just competes on creation order as before.
    #[serde(default)]
    pub priority: i64,
    /// Optional operator note shown on the target's dashboard card. Rendered
    /// there as raw HTML, not escaped - e.g. `<b>` tags come out bold - so
    /// only ever set this from trusted, operator-supplied text.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional regex excluding part of the search space from ever being
    /// carved out and handed to a worker. Matched only against a candidate's
    /// *leading character* (independent of candidate length) - e.g. "[M-Q]"
    /// skips every candidate starting with M through Q. Must compile as a
    /// regex or target creation is rejected; empty/omitted means no skipping.
    #[serde(default)]
    pub skip_regex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreateTargetResponse {
    pub target_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminPatchTargetRequest {
    /// "active" or "paused". Leave unset to change only `priority`.
    #[serde(default)]
    pub status: Option<String>,
    /// See `AdminCreateTargetRequest::priority`. Leave unset to leave priority unchanged.
    #[serde(default)]
    pub priority: Option<i64>,
    /// See `AdminCreateTargetRequest::description`. Leave unset to leave the
    /// description unchanged; pass `""` to clear it.
    #[serde(default)]
    pub description: Option<String>,
    /// See `AdminCreateTargetRequest::skip_regex`. Leave unset to leave it
    /// unchanged; pass `""` to clear it. A new value only takes effect for
    /// ranges carved after the patch - see `ranges::claim_range`.
    #[serde(default)]
    pub skip_regex: Option<String>,
    /// See `AdminCreateTargetRequest::alphabet_name`. Leave unset to leave
    /// the alphabet unchanged. Rejected if the name is unknown, or if the
    /// target's own (immutable) bounds contain a character outside the new
    /// alphabet. Ranges already carved - in progress, pending, completed or
    /// skipped - keep whatever alphabet they were carved with; only the
    /// carving cursor is affected, and only lazily, the next time fresh work
    /// is carved for this target (see `ranges::claim_range`).
    #[serde(default)]
    pub alphabet_name: Option<String>,
}

/// Fast-tracks a specific, bounded slice of a target's search space ahead of
/// its normal sequential sweep - see `ranges::claim_range`. Unlike
/// `AdminCreateTargetRequest::skip_regex`, this never removes anything from
/// the search, it only reorders when it happens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreatePriorityRangeRequest {
    /// A short, regex-*style* pattern describing the leading characters to
    /// prioritize, one position per "atom" - a literal character, `.`, a
    /// backslash escape, or a full `[...]` bracket class (ranges and
    /// negation both work). E.g. `"[ _-]S"` means "space, underscore or
    /// hyphen, followed by S". No quantifiers, alternation, groups or
    /// anchors - each pattern has one single, unambiguous length (its atom
    /// count), which must not exceed `length` below. A pattern with more
    /// than one matching character at some position (e.g. the bracket class
    /// above) expands into that many separate priority ranges, one per
    /// concrete prefix, all sharing this request's `priority`.
    pub pattern: String,
    /// The exact candidate length this priority range applies to - not a
    /// range of lengths. Wanting several lengths (e.g. both 9 and 10
    /// characters) means sending this request once per length.
    pub length: i64,
    /// Higher claims first, same convention as `AdminCreateTargetRequest::priority`.
    /// Multiple priority ranges may share a priority value.
    #[serde(default)]
    pub priority: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreatePriorityRangeResponse {
    /// One id per concrete prefix `pattern` expanded into - see
    /// `AdminCreatePriorityRangeRequest::pattern`.
    pub priority_range_ids: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphabetInfo {
    pub name: String,
    pub characters: String,
    pub size: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphabetsResponse {
    pub alphabets: Vec<AlphabetInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}
