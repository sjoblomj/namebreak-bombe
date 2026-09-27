//! Wire types used by the coordinator server - an identical shape needs to be
//! implemented by the client.

use serde::{Deserialize, Serialize};

/// A `MAJOR.MINOR.PATCH` version, semver-style, comparable field-by-field in
/// that order (the derived `Ord` already does the right thing, since Rust
/// compares struct fields in declaration order). Deliberately doesn't
/// support pre-release/build-metadata suffixes or version *ranges* - this
/// client/server pair only ever needs to compare two exact versions, never
/// parse an arbitrary semver range expression.
///
/// The wire format is always a plain `"X.Y.Z"` string (see
/// `RegisterRequest::protocol_version`/`RegisterResponse::server_protocol_version`),
/// not a nested JSON object - the C++ client's hand-rolled JSON parser only
/// handles flat objects, and a version number doesn't need anything richer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl Version {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Version { major, minor, patch }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::str::FromStr for Version {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.trim().split('.');
        let mut next = |what: &str| -> Result<u64, String> {
            let raw = parts.next().ok_or_else(|| format!("version '{s}' is missing its {what} component"))?;
            raw.parse::<u64>().map_err(|_| format!("version '{s}' has an invalid {what} component"))
        };
        let major = next("major")?;
        let minor = next("minor")?;
        let patch = next("patch")?;
        if parts.next().is_some() {
            return Err(format!("version '{s}' has more than three components"));
        }
        Ok(Version { major, minor, patch })
    }
}

/// This server's own wire-protocol version - see `Version`. Bump it (and
/// update this comment's changelog) whenever the request/response shapes
/// below, or what `alphabet::PREDEFINED_ALPHABETS` offers, change in a way a
/// client might care about:
/// - MAJOR: a shape changed in a way an older client's parser can't tolerate
///   (a field removed, renamed, or given a different type/meaning). The
///   server refuses to register a client whose declared MAJOR doesn't match
///   its own (see `handlers::register`) - nothing about compatibility can be
///   assumed across that boundary, so there's no point letting it proceed.
/// - MINOR: something purely additive was introduced - a new optional
///   field, or a new entry in `alphabet::PREDEFINED_ALPHABETS`. An older
///   client keeps working exactly as before; `ranges::claim_range` never
///   hands it an alphabet introduced in a MINOR version newer than what
///   that client declared. It searches such a target in the smallest
///   alphabet it does know that contains every character of the target's,
///   or gets no work from it if there's none (see
///   `alphabet::client_alphabet_for`) - so the server can keep an old,
///   un-upgraded client fed with work it can actually make sense of, rather
///   than crashing it with a target it has no idea how to search.
/// - PATCH: anything else (bug fixes, doc changes) - never gates anything.
///
/// Changelog: 1.0.0 - initial versioned release; every alphabet in
/// `PREDEFINED_ALPHABETS` as of this version is tagged `since: (1, 0)`.
pub const PROTOCOL_VERSION: Version = Version::new(1, 0, 0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub hostname: String,
    /// The backend this client searches with (e.g. `"cuda"`, `"cpu"`) -
    /// informational only: stored on the user, never consulted. Empty for a
    /// client that doesn't send it.
    #[serde(default)]
    pub backend: String,
    /// This client's own protocol version (`"X.Y.Z"`, see `PROTOCOL_VERSION`).
    pub protocol_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub user_id: i64,
    pub token: String,
    /// This server's own protocol version - see `PROTOCOL_VERSION`. Purely
    /// informational: the server has already decided whether to accept this
    /// registration at all (see `handlers::register`), so the client has no
    /// decision left to make from this - it's just useful to log if it
    /// differs from what the client expected.
    pub server_protocol_version: String,
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
    /// Skip candidates that close a bracket never opened - see
    /// `AdminCreateTargetRequest::prune_unopened_brackets`.
    #[serde(default)]
    pub prune_unopened_brackets: bool,
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
    /// Also doubles as this client's only liveness-of-*progress* signal (as
    /// opposed to liveness of the heartbeat itself, which every call already
    /// proves): reporting the same filename heartbeat after heartbeat - as a
    /// paused client necessarily would, having nothing new to report - is
    /// what lets the server notice and eventually reclaim a stalled range.
    /// See `HeartbeatResponse::range_released`.
    #[serde(default)]
    pub last_hash_a_match_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub lease_seconds: i64,
    /// True once this range should be abandoned - either its target was
    /// solved (by this range or a different one, see
    /// `ranges::complete_range`), or the range went STALL_RELEASE_SECONDS
    /// without any reported progress and was released back to pending for
    /// someone else (see `ranges::heartbeat_range`). Either way the client
    /// should abort its current search and won't be reporting completion for
    /// this range; the two reasons need no further distinguishing on the
    /// wire; a client that's locally paused already knows to stay paused and
    /// idle rather than claim a new range regardless of which one this was.
    pub range_released: bool,
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
    /// Skip candidates that close a bracket never opened: reading left to
    /// right, a ')' at a point where more ')' than '(' have been seen, or
    /// likewise a ']' with '['. The two kinds are counted separately.
    /// Brackets the prefix leaves open count as opened. Defaults to false.
    #[serde(default)]
    pub prune_unopened_brackets: bool,
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
    /// The shortest candidate length the main sweep carves - every shorter
    /// candidate is left out entirely. Must be between 1 and the alphabet's
    /// max supported length. Defaults to 1, i.e. every length.
    #[serde(default = "default_start_len")]
    pub start_len: i64,
}

fn default_start_len() -> i64 {
    1
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
    /// See `AdminCreateTargetRequest::alphabet_name`. Leave unset to leave
    /// the alphabet unchanged. Rejected if the name is unknown, or if the
    /// target's own (immutable) bounds contain a character outside the new
    /// alphabet. Ranges already carved - in progress, pending, completed or
    /// skipped - keep whatever alphabet they were carved with; only the
    /// carving cursor is affected, and only lazily, the next time fresh work
    /// is carved for this target (see `ranges::claim_range`).
    #[serde(default)]
    pub alphabet_name: Option<String>,
    /// See `AdminCreateTargetRequest::prune_symbol_runs`. Leave unset to
    /// leave it unchanged. Like `prune_unopened_brackets`, it's sent to
    /// clients with each claim, so a change applies to every range claimed
    /// after the patch - ranges already in progress keep the old setting.
    #[serde(default)]
    pub prune_symbol_runs: Option<bool>,
    /// See `AdminCreateTargetRequest::prune_unopened_brackets`. Leave unset
    /// to leave it unchanged.
    #[serde(default)]
    pub prune_unopened_brackets: Option<bool>,
    /// See `AdminCreateTargetRequest::max_backslash_count`. Leave unset to
    /// leave it unchanged. Applies to ranges claimed after the patch, like
    /// `prune_symbol_runs`.
    #[serde(default)]
    pub max_backslash_count: Option<i64>,
    /// See `AdminCreateTargetRequest::start_len`. Leave unset to leave it
    /// unchanged. Raising it past where the main sweep has reached makes the
    /// sweep jump straight to the start of the new length the next time it
    /// carves; lowering it never moves the sweep back, so lengths it already
    /// passed (or jumped over) stay as they are. Ranges already carved are
    /// untouched either way, and so are priority ranges.
    #[serde(default)]
    pub start_len: Option<i64>,
}

/// Fast-tracks a specific, bounded slice of a target's search space ahead of
/// its normal sequential sweep - see `ranges::claim_range`. Unlike
/// `AdminCreateSkipRangeRequest`, this never removes anything from the
/// search, it only reorders when it happens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreatePriorityRangeRequest {
    /// A short, regex-*style* pattern describing the leading characters to
    /// prioritize, one position per "atom" - a literal character, `.`, a
    /// backslash escape, or a full `[...]` bracket class (ranges and
    /// negation both work). E.g. `"[ _-]S"` means "space, underscore or
    /// hyphen, followed by S". No quantifiers, alternation, groups or
    /// anchors - each pattern has one single, unambiguous length (its atom
    /// count), which must not exceed `length` below. However many separate
    /// stretches of candidates it matches, it's one priority range - but a
    /// pattern matching more than 1000 of them is rejected (see
    /// `alphabet::pattern_spans` in the server).
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
    pub priority_range_id: i64,
}

/// Leaves every candidate of one length matching a pattern out of the
/// search, recorded as `skipped` ranges once carving reaches them. Ranges
/// already carved (even pending ones) are left as they are.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreateSkipRangeRequest {
    /// Same syntax as `AdminCreatePriorityRangeRequest::pattern`.
    pub pattern: String,
    /// The exact candidate length to skip at.
    pub length: i64,
    /// Plain text explaining why, shown on the dashboard. Required.
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCreateSkipRangeResponse {
    pub skip_range_id: i64,
}

/// What `DELETE /api/v1/admin/skip-ranges/{id}` did. Carving stops skipping
/// what the skip range matches either way; what it already skipped stays
/// skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminDeleteSkipRangeResponse {
    /// True if the skip range is gone entirely. False if it had already
    /// skipped something: it's then kept, marked removed, so the dashboard
    /// can still show why those candidates were skipped.
    pub deleted: bool,
}

/// What `DELETE /api/v1/admin/priority-ranges/{id}` did. Work already
/// carved from the priority range stays accounted for as it is; of the
/// rest, whatever the main sweep will still reach is handed back to it, and
/// whatever the main sweep has already gone past (it never goes back) stays
/// with the priority range so it still gets searched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminDeletePriorityRangeResponse {
    /// True if the priority range is gone entirely: nothing was ever carved
    /// from it and nothing had to be kept.
    pub deleted: bool,
    /// Candidates handed back to the main sweep. 128-bit, since a priority
    /// range at a long candidate length can span more than an i64 holds.
    pub returned_to_main_sweep: i128,
    /// Candidates kept as priority work because the main sweep is already
    /// past them.
    pub kept: i128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphabetInfo {
    pub name: String,
    pub characters: String,
    pub size: i64,
    /// The protocol version this alphabet was introduced in (`"X.Y"`, minor
    /// version only - see `PROTOCOL_VERSION`). A client whose own declared
    /// version is older than this will never be given this alphabet - see
    /// `alphabet::client_alphabet_for`.
    pub since: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphabetsResponse {
    pub alphabets: Vec<AlphabetInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn version_parses_a_well_formed_string() {
        assert_eq!(Version::from_str("1.2.3").unwrap(), Version::new(1, 2, 3));
        assert_eq!(Version::from_str("0.0.0").unwrap(), Version::new(0, 0, 0));
        assert_eq!(Version::from_str(" 10.20.30 ").unwrap(), Version::new(10, 20, 30), "surrounding whitespace is trimmed");
    }

    #[test]
    fn version_rejects_malformed_strings() {
        assert!(Version::from_str("1.2").is_err(), "missing patch component");
        assert!(Version::from_str("1").is_err(), "missing minor and patch components");
        assert!(Version::from_str("1.2.3.4").is_err(), "too many components");
        assert!(Version::from_str("1.2.x").is_err(), "non-numeric component");
        assert!(Version::from_str("").is_err());
        assert!(Version::from_str("1.-2.3").is_err(), "negative component");
    }

    #[test]
    fn version_display_round_trips_through_from_str() {
        let v = Version::new(3, 14, 15);
        assert_eq!(Version::from_str(&v.to_string()).unwrap(), v);
    }

    #[test]
    fn version_orders_major_then_minor_then_patch() {
        assert!(Version::new(1, 0, 0) < Version::new(1, 0, 1));
        assert!(Version::new(1, 0, 1) < Version::new(1, 1, 0));
        assert!(Version::new(1, 9, 9) < Version::new(2, 0, 0));
        assert_eq!(Version::new(1, 2, 3), Version::new(1, 2, 3));
    }
}
