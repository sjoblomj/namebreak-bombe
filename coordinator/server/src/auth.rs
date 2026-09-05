use axum::extract::{FromRequestParts, State};
use axum::http::request::Parts;

use crate::error::AppError;
use crate::models::User;
use crate::state::{now_unix, AppState};

/// Extracts and validates the `Authorization: Bearer <token>` header, loading the
/// matching user. This is the server's whole bot/scraper defense: no token, no
/// access to /claim, /heartbeat or /complete - and the token is only ever handed
/// out via /register.
pub struct AuthedUser(pub User);

impl FromRequestParts<AppState> for AuthedUser {
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, Self::Rejection> {
        let token = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(|s| s.to_string())
            .ok_or(AppError::Unauthorized)?;

        let State(state) = State::<AppState>::from_request_parts(parts, state)
            .await
            .map_err(|_| AppError::Internal("failed to extract app state".into()))?;

        let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE token = ?")
            .bind(&token)
            .fetch_optional(&state.pool)
            .await?
            .ok_or(AppError::Unauthorized)?;

        sqlx::query("UPDATE users SET last_seen_at = ? WHERE id = ?")
            .bind(now_unix())
            .bind(user.id)
            .execute(&state.pool)
            .await?;

        Ok(AuthedUser(user))
    }
}

/// Extracts and validates the `X-Admin-Token` header for the target-management
/// endpoints, which only the operator (not regular registered users) should reach.
pub struct AdminAuth;

impl FromRequestParts<AppState> for AdminAuth {
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, Self::Rejection> {
        let header = parts
            .headers
            .get("X-Admin-Token")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .ok_or(AppError::Forbidden)?;

        let State(state) = State::<AppState>::from_request_parts(parts, state)
            .await
            .map_err(|_| AppError::Internal("failed to extract app state".into()))?;

        if header == state.admin_token {
            Ok(AdminAuth)
        } else {
            Err(AppError::Forbidden)
        }
    }
}
