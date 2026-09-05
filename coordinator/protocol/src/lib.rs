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
    pub lower_bound_filename: String,
    pub upper_bound_filename: String,
    /// Number of candidates covered by this range - lets the client report
    /// throughput on completion without doing any index math itself.
    pub candidate_count: i64,
    pub lease_seconds: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub lease_seconds: i64,
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
    pub min_len: i64,
    pub max_len: i64,
    #[serde(default)]
    pub prune_symbol_runs: bool,
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
pub struct ErrorResponse {
    pub error: String,
}
