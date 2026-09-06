//! Wire types shared between the coordinator server and the client, so the two
//! can't silently drift out of sync on the JSON shape.

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
    /// Max '\' occurrences namebreak will allow in a candidate before discarding
    /// it unhashed; 0 means unlimited. Passed straight through as `namebreak`'s
    /// `<maxBackslashCount>` CLI argument.
    pub max_backslash_count: i64,
    pub lower_bound_filename: String,
    pub upper_bound_filename: String,
    /// The literal alphabet characters for this range's target - passed straight
    /// through as `namebreak`'s `<alphabet>` CLI argument. The client never needs
    /// to know this by name; only the server resolves profile names.
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
    /// True if this range's target has already been solved (by someone else,
    /// via a different range). The client should kill its running `namebreak`
    /// subprocess rather than let it keep searching a target that's already
    /// found - it won't be reporting completion for this range either way.
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
    /// Full filenames (like `ClaimResponse`'s bound fields) - the server
    /// always starts searching at the very beginning `lower_bound_filename`
    /// implies and always searches as long as the chosen alphabet supports,
    /// tightened to exactly this alphabetical range at every candidate length
    /// in between (the two bounds don't need to be the same length as each
    /// other - e.g. "ART\BLACKSMITH.GRP" to "ART\CATAPULT.GRP" is valid).
    pub lower_bound_filename: String,
    pub upper_bound_filename: String,
    #[serde(default)]
    pub prune_symbol_runs: bool,
    /// One of the names from `GET /api/v1/alphabets`. Defaults to `"size49"`
    /// (the original default alphabet) when omitted, for backward compatibility.
    #[serde(default)]
    pub alphabet_name: Option<String>,
    /// Max '\' occurrences allowed in a candidate; 0 (the default when omitted)
    /// means unlimited.
    #[serde(default)]
    pub max_backslash_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreateTargetResponse {
    pub target_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminPatchTargetRequest {
    /// "active" or "paused"
    pub status: String,
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
