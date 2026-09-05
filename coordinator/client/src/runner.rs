use namebreak_protocol::ClaimResponse;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub struct RunOutcome {
    /// The process exited with the expected code for a completed bounded search
    /// (0 = full match found, 2 = reached the upper bound with no match). Any
    /// other exit (crash, CUDA error, wrong args) is *not* a clean outcome, and
    /// the caller should skip reporting completion so the server reclaims the
    /// range instead of wrongly marking it done.
    pub clean_exit: bool,
    pub found: bool,
    pub filename: Option<String>,
}

const FULL_MATCH_PREFIX: &str = "BOTH HASHES MATCH: ";
const PARTIAL_MATCH_PREFIX: &str = "Hash A matches: ";

/// Shared with the heartbeat loop, which reads it at each tick: the most recent
/// "Hash A matches: <filename>" line namebreak has printed for the range currently
/// running, if any. Used as a checkpoint - see `HeartbeatRequest` for why it's safe
/// to treat everything up to this candidate as searched.
pub type LastHashAMatch = Arc<Mutex<Option<String>>>;

/// Runs `namebreak bounded ...` against exactly the range described by `claim`,
/// streaming its stdout through to this process's own stdout so the operator can
/// still watch progress, while also scanning for the "BOTH HASHES MATCH:" line
/// that namebreak.cu prints on a real find (and updating `last_hash_a_match` as
/// partial matches stream by, for the heartbeat loop running concurrently with this).
pub async fn run_namebreak(
    bin: &Path,
    workdir: &Path,
    claim: &ClaimResponse,
    last_hash_a_match: LastHashAMatch,
) -> anyhow::Result<RunOutcome> {
    let mut cmd = Command::new(bin);
    cmd.current_dir(workdir)
        .arg("bounded")
        .arg(&claim.alphabet)
        .arg(&claim.lower_bound_filename) // startCandidate: begin exactly at this range's start
        .arg(&claim.prefix)
        .arg(&claim.suffix)
        .arg(&claim.lower_bound_filename) // lowerBound
        .arg(&claim.upper_bound_filename) // upperBound
        .arg(&claim.hash_a_hex)
        .arg(&claim.hash_b_hex)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if claim.prune_symbol_runs {
        cmd.arg("--prune-symbol-runs");
    }

    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let mut lines = BufReader::new(stdout).lines();

    let mut found_filename = None;
    while let Some(line) = lines.next_line().await? {
        println!("{line}");
        if let Some(name) = line.strip_prefix(FULL_MATCH_PREFIX) {
            found_filename = Some(name.trim().to_string());
        } else if let Some(name) = line.strip_prefix(PARTIAL_MATCH_PREFIX) {
            *last_hash_a_match.lock().unwrap() = Some(name.trim().to_string());
        }
    }

    let status = child.wait().await?;
    let code = status.code().unwrap_or(-1);
    let found = code == 0;

    if found && found_filename.is_none() {
        // Fall back to matches.txt's last line, in case the stdout scan above
        // somehow missed the device-printed match line.
        found_filename = last_line_of(&workdir.join("matches.txt")).await;
        if found_filename.is_none() {
            tracing::warn!("namebreak exited 0 (found) but no match filename could be determined");
        }
    }

    Ok(RunOutcome { clean_exit: code == 0 || code == 2, found, filename: found_filename })
}

async fn last_line_of(path: &Path) -> Option<String> {
    let contents = tokio::fs::read_to_string(path).await.ok()?;
    contents.lines().last().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}
