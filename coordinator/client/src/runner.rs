use namebreak_protocol::ClaimResponse;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::watch;

pub struct RunOutcome {
    /// The process exited with the expected code for a completed bounded search
    /// (0 = full match found, 2 = reached the upper bound with no match). Any
    /// other exit (crash, CUDA error, wrong args) is *not* a clean outcome, and
    /// the caller should skip reporting completion so the server reclaims the
    /// range instead of wrongly marking it done.
    pub clean_exit: bool,
    pub found: bool,
    pub filename: Option<String>,
    /// True if this range's target was solved elsewhere and the heartbeat loop
    /// told us to stop - the subprocess was killed, not run to completion. The
    /// caller shouldn't report completion (there's nothing to report) or treat
    /// this as a failure (it's expected and intentional).
    pub aborted: bool,
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
    mut abort_rx: watch::Receiver<bool>,
) -> anyhow::Result<RunOutcome> {
    let mut cmd = Command::new(bin);
    cmd.current_dir(workdir)
        // Own process group (pgid = its own pid), so an abort can kill the whole
        // tree in one signal - not just this direct child. Matters if
        // --namebreak-bin ever points at a wrapper script (e.g. one that sources
        // CUDA env vars before exec'ing the real binary): killing only the
        // wrapper would silently orphan the actual GPU-searching process.
        .process_group(0)
        .arg("bounded")
        .arg(&claim.alphabet)
        .arg(claim.max_backslash_count.to_string())
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
    loop {
        tokio::select! {
            line = lines.next_line() => {
                match line? {
                    Some(line) => {
                        println!("{line}");
                        if let Some(name) = line.strip_prefix(FULL_MATCH_PREFIX) {
                            found_filename = Some(name.trim().to_string());
                        } else if let Some(name) = line.strip_prefix(PARTIAL_MATCH_PREFIX) {
                            *last_hash_a_match.lock().unwrap() = Some(name.trim().to_string());
                        }
                    }
                    None => break, // stdout closed - namebreak is finishing up on its own
                }
            }
            _ = abort_rx.changed() => {
                if *abort_rx.borrow() {
                    tracing::warn!("target already solved elsewhere - killing namebreak for this range");
                    kill_process_group(&child);
                    let _ = child.wait().await; // reap, avoid leaving a zombie
                    return Ok(RunOutcome { clean_exit: false, found: false, filename: None, aborted: true });
                }
            }
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

    Ok(RunOutcome { clean_exit: code == 0 || code == 2, found, filename: found_filename, aborted: false })
}

/// SIGKILLs the child's whole process group (see the `.process_group(0)` comment
/// on the Command builder above) rather than `tokio::process::Child::kill()`,
/// which only signals the single direct child.
fn kill_process_group(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        // Safety: just a signal-sending syscall, no memory involved. `-pid`
        // targets the process group namebreak was placed into at spawn time
        // (pgid == pid, from `.process_group(0)`), not an arbitrary group.
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
}

async fn last_line_of(path: &Path) -> Option<String> {
    let contents = tokio::fs::read_to_string(path).await.ok()?;
    contents.lines().last().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    fn dummy_claim() -> ClaimResponse {
        ClaimResponse {
            range_id: 1,
            target_id: 1,
            target_name: "t".into(),
            prefix: "PRE".into(),
            suffix: ".SUF".into(),
            hash_a_hex: "0x0".into(),
            hash_b_hex: "0x0".into(),
            prune_symbol_runs: false,
            max_backslash_count: 0,
            lower_bound_filename: "PREaa.SUF".into(),
            upper_bound_filename: "PREzz.SUF".into(),
            alphabet: "abcdefghijklmnopqrstuvwxyz".into(),
            candidate_count: 676,
            lease_seconds: 60,
        }
    }

    /// Writes a tiny shell script standing in for `namebreak` - ignores whatever
    /// argv it's given (real `namebreak` can't even start in a sandbox with no
    /// working CUDA runtime, so these tests exercise process control, not the
    /// real binary) and runs `body` instead.
    async fn write_stub(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        tokio::fs::write(&path, format!("#!/bin/sh\n{body}\n")).await.unwrap();
        let mut perms = tokio::fs::metadata(&path).await.unwrap().permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&path, perms).await.unwrap();
        path
    }

    #[tokio::test]
    async fn abort_signal_kills_a_long_running_subprocess() {
        let dir = tempdir();
        // Models --namebreak-bin pointing at a wrapper script (e.g. one that
        // sources CUDA env vars before running the real binary): `sleep` here
        // stands in for the actual GPU-searching process, run as a *grandchild*
        // of run_namebreak's direct child (the shell). Recording $! (sleep's own
        // pid, backgrounded) rather than $$ (the shell's pid) is what makes this
        // test actually exercise the process-group kill instead of just the
        // direct child - killing only the shell would leave sleep running.
        let stub = write_stub(&dir, "sleepy", "sleep 30 & echo $! > pid; wait").await;
        let (abort_tx, abort_rx) = watch::channel(false);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            let _ = abort_tx.send(true);
        });

        let last_hash_a_match: LastHashAMatch = Arc::new(Mutex::new(None));
        let outcome = tokio::time::timeout(
            Duration::from_secs(5), // well under the 30s sleep
            run_namebreak(&stub, &dir, &dummy_claim(), last_hash_a_match, abort_rx),
        )
        .await
        .expect("run_namebreak should return promptly once aborted, not wait out the full sleep")
        .unwrap();

        assert!(outcome.aborted);
        assert!(!outcome.found);
        assert!(!outcome.clean_exit);

        let pid: i32 = tokio::fs::read_to_string(dir.join("pid")).await.unwrap().trim().parse().unwrap();
        // SIGKILL delivery is asynchronous, and nothing here explicitly waits for
        // this specific grandchild (only the direct child gets `child.wait()`'d) -
        // so give the kernel a brief window to finish reaping it under load,
        // rather than requiring it to already be gone in the same instant.
        let mut still_alive = process_alive(pid);
        for _ in 0..20 {
            if !still_alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            still_alive = process_alive(pid);
        }
        assert!(!still_alive, "the sleeping subprocess (pid {pid}) should have been killed, not left running");
    }

    fn process_alive(pid: i32) -> bool {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }

    #[tokio::test]
    async fn normal_not_found_completion_still_works_through_the_select_loop() {
        let dir = tempdir();
        let stub = write_stub(&dir, "not_found", "echo 'just some output'\nexit 2").await;
        let (_abort_tx, abort_rx) = watch::channel(false);

        let last_hash_a_match: LastHashAMatch = Arc::new(Mutex::new(None));
        let outcome = run_namebreak(&stub, &dir, &dummy_claim(), last_hash_a_match, abort_rx).await.unwrap();

        assert!(outcome.clean_exit);
        assert!(!outcome.found);
        assert!(!outcome.aborted);
    }

    #[tokio::test]
    async fn full_match_completion_is_detected_through_the_select_loop() {
        let dir = tempdir();
        let stub = write_stub(&dir, "found", "echo 'BOTH HASHES MATCH: PREAB.SUF'\nexit 0").await;
        let (_abort_tx, abort_rx) = watch::channel(false);

        let last_hash_a_match: LastHashAMatch = Arc::new(Mutex::new(None));
        let outcome = run_namebreak(&stub, &dir, &dummy_claim(), last_hash_a_match, abort_rx).await.unwrap();

        assert!(outcome.clean_exit);
        assert!(outcome.found);
        assert!(!outcome.aborted);
        assert_eq!(outcome.filename.as_deref(), Some("PREAB.SUF"));
    }

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("namebreak-client-test-{}-{}", std::process::id(), rand_suffix()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn rand_suffix() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        nanos.wrapping_add(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}
