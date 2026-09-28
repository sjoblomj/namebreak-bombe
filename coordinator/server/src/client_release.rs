//! Which client releases this server accepts - see
//! `namebreak_protocol::ClientReleases`. Releases are the GitHub release tags
//! the client is built with (`NAMEBREAK_VERSION` in client/CMakeLists.txt):
//! `vYYYY-MM-DD`, optionally with a `.N` for a later release the same day.

use namebreak_protocol::ClientReleases;
use sqlx::SqlitePool;

use crate::error::AppError;

/// Where to point a user who needs a newer client.
pub const RELEASES_URL: &str = "https://github.com/sjoblomj/namebreak-bombe/releases";

/// What a client built from source, rather than released, calls itself. It's
/// never refused: whoever built it is expected to keep it current themselves.
pub const DEV_RELEASE: &str = "dev";

const MINIMUM_KEY: &str = "minimum_client_release";

/// A release tag, ordered by date and then same-day number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Release {
    year: u32,
    month: u32,
    day: u32,
    number: u32,
}

impl std::str::FromStr for Release {
    type Err = String;

    /// `vYYYY-MM-DD`, or `vYYYY-MM-DD.N`.
    fn from_str(s: &str) -> Result<Self, String> {
        let invalid = || format!("'{s}' isn't a release - expected vYYYY-MM-DD, optionally followed by .N");
        let rest = s.trim().strip_prefix('v').ok_or_else(invalid)?;
        let (date, number) = match rest.split_once('.') {
            Some((date, number)) => (date, number.parse::<u32>().map_err(|_| invalid())?),
            None => (rest, 0),
        };
        let parts: Vec<&str> = date.split('-').collect();
        let [year, month, day] = parts.as_slice() else {
            return Err(invalid());
        };
        if year.len() != 4 || month.len() != 2 || day.len() != 2 {
            return Err(invalid());
        }
        let field = |f: &str| f.parse::<u32>().map_err(|_| invalid());
        let release = Release { year: field(year)?, month: field(month)?, day: field(day)?, number };
        if !(1..=12).contains(&release.month) || !(1..=31).contains(&release.day) {
            return Err(invalid());
        }
        Ok(release)
    }
}

/// Whether `client` (as the client reported it) is older than `wanted`. A
/// dev build never is; a missing or unrecognized release always is, since
/// it's from before clients reported one.
fn is_older(client: Option<&str>, wanted: &str) -> bool {
    let Ok(wanted) = wanted.parse::<Release>() else {
        return false; // only valid ones are ever stored - see set_client_releases
    };
    match client {
        Some(DEV_RELEASE) => false,
        Some(client) => client.parse::<Release>().map_or(true, |client| client < wanted),
        None => true,
    }
}

fn describe(client: Option<&str>) -> String {
    match client {
        Some(client) => format!("namebreak {client}"),
        None => "This namebreak".to_string(),
    }
}

pub async fn load_client_releases(pool: &SqlitePool) -> Result<ClientReleases, AppError> {
    let minimum: Option<String> =
        sqlx::query_scalar("SELECT value FROM server_settings WHERE key = ?").bind(MINIMUM_KEY).fetch_optional(pool).await?;
    Ok(ClientReleases { minimum })
}

/// Replaces the settings; `None` clears the minimum. Rejects anything that
/// isn't a release tag.
pub async fn set_client_releases(pool: &SqlitePool, releases: &ClientReleases) -> Result<(), AppError> {
    match &releases.minimum {
        Some(minimum) => {
            minimum.parse::<Release>().map_err(AppError::BadRequest)?;
            sqlx::query("INSERT INTO server_settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value")
                .bind(MINIMUM_KEY)
                .bind(minimum.trim())
                .execute(pool)
                .await?
        }
        None => sqlx::query("DELETE FROM server_settings WHERE key = ?").bind(MINIMUM_KEY).execute(pool).await?,
    };
    Ok(())
}

/// Refuses a client older than the minimum release, with a message for it
/// to show its user (HTTP 426 - see `AppError::UpgradeRequired`).
pub fn check_minimum(releases: &ClientReleases, client: Option<&str>) -> Result<(), AppError> {
    match &releases.minimum {
        Some(minimum) if is_older(client, minimum) => Err(AppError::UpgradeRequired(format!(
            "{} is too old for this server - {minimum} or newer is required. Download it from {RELEASES_URL}",
            describe(client)
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release(s: &str) -> Release {
        s.parse().unwrap()
    }

    #[test]
    fn releases_order_by_date_then_same_day_number() {
        assert!(release("v2026-09-26") < release("v2026-09-27"));
        assert!(release("v2026-09-26") < release("v2026-09-26.1"));
        assert!(release("v2026-09-26.2") < release("v2026-10-01"));
        assert!(release("v2025-12-31.9") < release("v2026-01-01"));
    }

    #[test]
    fn non_releases_are_rejected() {
        for s in ["2026-09-26", "v2026-9-26", "v2026-13-01", "v2026-09-26.x", "dev", "v1.2.3", ""] {
            assert!(s.parse::<Release>().is_err(), "{s}");
        }
    }

    #[test]
    fn check_minimum_refuses_older_and_unreported_releases_but_not_dev_builds() {
        let releases = ClientReleases { minimum: Some("v2026-09-26".into()) };
        assert!(check_minimum(&releases, Some("v2026-09-26")).is_ok());
        assert!(check_minimum(&releases, Some("v2026-10-01")).is_ok());
        assert!(check_minimum(&releases, Some(DEV_RELEASE)).is_ok());
        assert!(matches!(check_minimum(&releases, Some("v2026-09-25")), Err(AppError::UpgradeRequired(msg)) if msg.contains("v2026-09-26")));
        assert!(matches!(check_minimum(&releases, None), Err(AppError::UpgradeRequired(_))));
        assert!(check_minimum(&ClientReleases::default(), None).is_ok(), "no minimum set");
    }
}
