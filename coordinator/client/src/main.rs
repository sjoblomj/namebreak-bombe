mod api;
mod cli;
mod runner;

use api::ApiClient;
use clap::Parser;
use namebreak_protocol::{ClaimResponse, CompleteRequest};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = cli::Args::parse();
    let hostname = cli::resolve_hostname(args.hostname.clone());
    tokio::fs::create_dir_all(&args.workdir).await.ok();

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

        if let Err(err) = run_one(&api, &args.namebreak_bin, &args.workdir, &claim).await {
            tracing::error!(%err, range_id = claim.range_id, "range run failed - letting the lease expire so it gets reassigned");
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

    let heartbeat_every = Duration::from_secs((claim.lease_seconds / 3).max(10) as u64);
    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    let heartbeat_handle = {
        let api = api.clone();
        let range_id = claim.range_id;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(heartbeat_every);
            interval.tick().await; // first tick fires immediately; skip it
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(err) = api.heartbeat(range_id).await {
                            tracing::warn!(%err, range_id, "heartbeat failed");
                        }
                    }
                    _ = stop_rx.changed() => break,
                }
            }
        })
    };

    let started = Instant::now();
    let outcome = runner::run_namebreak(namebreak_bin, workdir, claim).await;
    let _ = stop_tx.send(true);
    let _ = heartbeat_handle.await;
    let outcome = outcome?;
    let elapsed_seconds = started.elapsed().as_secs_f64();

    if !outcome.clean_exit {
        anyhow::bail!("namebreak did not exit cleanly (expected code 0 or 2)");
    }

    if outcome.found {
        tracing::info!(range_id = claim.range_id, filename = ?outcome.filename, "MATCH FOUND");
    } else {
        tracing::info!(range_id = claim.range_id, "range exhausted, no match");
    }

    api.complete(
        claim.range_id,
        &CompleteRequest {
            found: outcome.found,
            filename: outcome.filename,
            elapsed_seconds,
            candidates_processed: claim.candidate_count,
        },
    )
    .await?;

    Ok(())
}
