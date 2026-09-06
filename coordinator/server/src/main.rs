mod alphabet;
mod auth;
mod dashboard;
mod db;
mod error;
mod handlers;
mod models;
mod ranges;
mod state;

use axum::routing::{get, patch, post};
use axum::Router;
use state::{AppState, Inner, RangeConfig};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite://namebreak.db".to_string());
    let admin_token = std::env::var("ADMIN_TOKEN")
        .expect("ADMIN_TOKEN env var must be set (protects the /admin/* target-management endpoints)");
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    let pool = db::connect(&database_url).await?;
    let config = RangeConfig::from_env();
    let state = AppState(Arc::new(Inner { pool, admin_token, config }));

    spawn_reclaim_task(state.clone());

    let app = Router::new()
        .route("/", get(dashboard::dashboard_page))
        .route("/api/v1/dashboard", get(dashboard::dashboard_data))
        .route("/api/v1/register", post(handlers::register))
        .route("/api/v1/claim", post(handlers::claim))
        .route("/api/v1/ranges/{id}/heartbeat", post(handlers::heartbeat))
        .route("/api/v1/ranges/{id}/complete", post(handlers::complete))
        .route("/api/v1/status", get(handlers::status))
        .route("/api/v1/alphabets", get(handlers::alphabets))
        .route("/api/v1/admin/targets", post(handlers::admin_create_target))
        .route("/api/v1/admin/targets/{id}", patch(handlers::admin_patch_target).delete(handlers::admin_delete_target))
        .with_state(state);

    tracing::info!(%bind_addr, "starting namebreak coordinator server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn spawn_reclaim_task(state: AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(state.config.reclaim_interval_secs));
        loop {
            interval.tick().await;
            match ranges::reclaim_expired(&state.pool).await {
                Ok(0) => {}
                Ok(n) => tracing::info!(count = n, "reclaimed timed-out ranges"),
                Err(err) => tracing::error!(%err, "failed to reclaim timed-out ranges"),
            }
        }
    });
}
