mod api;
mod cli;
mod runner;

use api::ApiClient;
use clap::Parser;
use namebreak_protocol::{ClaimResponse, CompleteRequest};
use runner::LastHashAMatch;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Fixed rather than derived from the range's lease, so progress checkpoints (and
/// the liveness signal the server's reclaim sweep relies on) land at a steady,
/// predictable cadence regardless of how big a range is or how fast a client is.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = cli::Args::parse();
    let hostname = cli::resolve_hostname(args.hostname.clone());
    tokio::fs::create_dir_all(&args.workdir).await.ok();

    // Resolve to an absolute path up front, before `run_namebreak` ever calls
    // `.current_dir(workdir)` on the spawned Command: a relative program path
    // given to `Command` is resolved against the *new* (post-chdir) working
    // directory on Unix, not this process's cwd - so a relative --namebreak-bin
    // combined with a relative --workdir can silently point at the wrong file
    // (ENOENT) depending on how the two paths relate. Canonicalizing here also
    // fails fast with a clear error if the binary doesn't exist at all, instead
    // of that surfacing as a mysterious ENOENT from inside the spawn loop.
    let namebreak_bin = args.namebreak_bin.canonicalize().map_err(|err| {
        anyhow::anyhow!("--namebreak-bin {:?} not found: {err}", args.namebreak_bin)
    })?;

    let api = ApiClient::register(&args.server_url, &args.username, &hostname).await?;
    let poll_interval = Duration::from_secs(args.poll_interval_secs);

    loop {
        let claim = match api.claim().await {
            Ok(Some(c)) => c,
            Ok(None) => {
                tracing::info!("no work available, sleeping");
                tokio::time::sleep(poll_interval).await;
                continue;
            }
            Err(err) => {
                tracing::warn!(%err, "claim failed, retrying after backoff");
                tokio::time::sleep(poll_interval).await;
                continue;
            }
        };

        if let Err(err) = run_one(&api, &namebreak_bin, &args.workdir, &claim).await {
            tracing::error!(%err, range_id = claim.range_id, "range run failed - letting the lease expire so it gets reassigned");
            // Without this, a persistent local failure (bad binary, missing CUDA
            // driver, etc.) turns into a tight claim/fail loop that hammers the
            // server - unlike the claim-error branch above, which already backs off.
            tokio::time::sleep(poll_interval).await;
        }
    }
}

async fn run_one(
    api: &ApiClient,
    namebreak_bin: &std::path::Path,
    workdir: &std::path::Path,
    claim: &ClaimResponse,
) -> anyhow::Result<()> {
    tracing::info!(
        range_id = claim.range_id,
        target = %claim.target_name,
        lower = %claim.lower_bound_filename,
        upper = %claim.upper_bound_filename,
        "starting range"
    );

    let last_hash_a_match: LastHashAMatch = Arc::new(Mutex::new(None));
    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    let (abort_tx, abort_rx) = tokio::sync::watch::channel(false);
    let heartbeat_handle = {
        let api = api.clone();
        let range_id = claim.range_id;
        let last_hash_a_match = last_hash_a_match.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
            interval.tick().await; // first tick fires immediately; skip it
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let latest = last_hash_a_match.lock().unwrap().clone();
                        match api.heartbeat(range_id, latest).await {
                            Ok(resp) if resp.target_solved => {
                                tracing::info!(range_id, "target already solved elsewhere - signaling abort");
                                let _ = abort_tx.send(true);
                                break;
                            }
                            Ok(_) => {}
                            Err(err) => tracing::warn!(%err, range_id, "heartbeat failed"),
                        }
                    }
                    _ = stop_rx.changed() => break,
                }
            }
        })
    };

    let started = Instant::now();
    let outcome = runner::run_namebreak(namebreak_bin, workdir, claim, last_hash_a_match, abort_rx).await;
    let _ = stop_tx.send(true);
    let _ = heartbeat_handle.await;
    let outcome = outcome?;
    let elapsed_seconds = started.elapsed().as_secs_f64();

    if outcome.aborted {
        tracing::info!(range_id = claim.range_id, "range aborted - target was already solved by someone else");
        return Ok(());
    }

    if !outcome.clean_exit {
        anyhow::bail!("namebreak did not exit cleanly (expected code 0 or 2)");
    }

    if outcome.found {
        tracing::info!(range_id = claim.range_id, filename = ?outcome.filename, "MATCH FOUND");
    } else {
        tracing::info!(range_id = claim.range_id, "range exhausted, no match");
    }

    let filename = outcome.filename;
    match api
        .complete(
            claim.range_id,
            &CompleteRequest {
                found: outcome.found,
                filename: filename.clone(),
                elapsed_seconds,
                candidates_processed: claim.candidate_count,
            },
        )
        .await
    {
        Ok(()) => {}
        Err(err) if err.status() == Some(reqwest::StatusCode::CONFLICT) => {
            // This range's ownership moved on before we could report in - almost
            // certainly a network outage during heartbeating that outlasted the
            // lease, so the server already reassigned it to someone else.
            if outcome.found {
                tracing::error!(
                    range_id = claim.range_id, filename = ?filename,
                    "found a match but lost ownership of this range before reporting it - \
                     the match is still recorded locally in matches.txt, but the server was \
                     never told about it; check matches.txt manually"
                );
            } else {
                tracing::warn!(
                    range_id = claim.range_id,
                    "lost ownership of this range before reporting completion - the server had \
                     already reassigned it, so this GPU time was redundant but nothing is lost"
                );
            }
        }
        Err(err) => return Err(err.into()),
    }

    Ok(())
}
