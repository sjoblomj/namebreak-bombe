mod alphabet;
mod auth;
mod basenames;
mod canary;
mod client_release;
mod dashboard;
mod db;
mod dictionary;
#[cfg(test)]
mod dictionary_tests;
mod error;
mod handlers;
mod likely_prefixes;
mod models;
mod ranges;
mod state;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, patch, post, put};
use axum::Router;
use tower_http::compression::CompressionLayer;
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
    config.likely_prefixes.refresh_matches(&pool).await?;
    let state = AppState(Arc::new(Inner { pool, admin_token, config }));

    spawn_reclaim_task(state.clone());
    let app = router(state);

    tracing::info!(%bind_addr, "starting namebreak coordinator server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Every route. Responses are gzipped for a client that says it takes gzip
/// (Accept-Encoding) - a browser, curl --compressed, and the client's word
/// list downloads: the dashboard's JSON, a target's basenames (a million
/// are 31 MB, gzipped about 11) and a word list are worth it.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(dashboard::dashboard_page))
        .route("/api/v1/dashboard", get(dashboard::dashboard_data))
        .route("/api/v1/register", post(handlers::register))
        .route("/api/v1/claim", post(handlers::claim))
        .route("/api/v1/ranges/{id}/heartbeat", post(handlers::heartbeat))
        .route("/api/v1/ranges/{id}/complete", post(handlers::complete))
        .route("/api/v1/ranges/{id}/quit", post(handlers::quit))
        .route("/api/v1/word-lists/{name}", get(handlers::word_list))
        .route("/api/v1/targets/{id}/basenames", get(handlers::target_basenames))
        .route("/api/v1/targets/{id}/word-lists/{name}", get(handlers::target_word_list))
        .route("/api/v1/status", get(handlers::status))
        .route("/api/v1/alphabets", get(handlers::alphabets))
        .route("/api/v1/admin/targets", post(handlers::admin_create_target))
        .route("/api/v1/admin/targets/{id}", patch(handlers::admin_patch_target).delete(handlers::admin_delete_target))
        .route("/api/v1/admin/targets/{id}/priority-ranges", post(handlers::admin_create_priority_range))
        .route("/api/v1/admin/priority-ranges/{id}", axum::routing::delete(handlers::admin_delete_priority_range))
        .route("/api/v1/admin/targets/{id}/skip-ranges", post(handlers::admin_create_skip_range))
        .route("/api/v1/admin/skip-ranges/{id}", axum::routing::delete(handlers::admin_delete_skip_range))
        .route("/api/v1/admin/client-releases", get(handlers::admin_get_client_releases).put(handlers::admin_set_client_releases))
        .route("/api/v1/admin/word-lists", get(handlers::admin_list_word_lists))
        // A word list can be far bigger than the 2 MB a request body may
        // otherwise have.
        .route(
            "/api/v1/admin/word-lists/{name}",
            put(handlers::admin_put_word_list).delete(handlers::admin_delete_word_list).layer(DefaultBodyLimit::max(256 << 20)),
        )
        .layer(CompressionLayer::new())
        .with_state(state)
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
