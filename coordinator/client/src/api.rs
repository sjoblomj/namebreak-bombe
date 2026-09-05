use namebreak_protocol::{ClaimResponse, CompleteRequest, HeartbeatRequest, RegisterRequest, RegisterResponse};
use std::time::Duration;

/// Well under HEARTBEAT_INTERVAL, so a request that hangs (rather than failing
/// fast) can't stall the heartbeat loop past its next scheduled tick - without
/// this, reqwest has no default request timeout and a wedged connection could
/// block indefinitely.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Clone)]
pub struct ApiClient {
    http: reqwest::Client,
    base_url: String,
    token: String,
}

impl ApiClient {
    pub async fn register(base_url: &str, username: &str, hostname: &str) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build()?;
        let resp = http
            .post(format!("{base_url}/api/v1/register"))
            .json(&RegisterRequest { username: username.to_string(), hostname: hostname.to_string() })
            .send()
            .await?
            .error_for_status()?
            .json::<RegisterResponse>()
            .await?;
        tracing::info!(user_id = resp.user_id, "registered with coordinator");
        Ok(Self { http, base_url: base_url.trim_end_matches('/').to_string(), token: resp.token })
    }

    pub async fn claim(&self) -> anyhow::Result<Option<ClaimResponse>> {
        let resp = self
            .http
            .post(format!("{}/api/v1/claim", self.base_url))
            .bearer_auth(&self.token)
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NO_CONTENT {
            return Ok(None);
        }
        let resp = resp.error_for_status()?;
        Ok(Some(resp.json::<ClaimResponse>().await?))
    }

    pub async fn heartbeat(&self, range_id: i64, last_hash_a_match_filename: Option<String>) -> anyhow::Result<()> {
        self.http
            .post(format!("{}/api/v1/ranges/{range_id}/heartbeat", self.base_url))
            .bearer_auth(&self.token)
            .json(&HeartbeatRequest { last_hash_a_match_filename })
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Returns the raw `reqwest::Error` (rather than `anyhow::Error`) so the
    /// caller can distinguish a 409 (this range's ownership moved on - e.g. a
    /// network outage during heartbeating outlasted the lease and the server
    /// already reassigned it) from other failures worth surfacing differently.
    pub async fn complete(&self, range_id: i64, req: &CompleteRequest) -> Result<(), reqwest::Error> {
        self.http
            .post(format!("{}/api/v1/ranges/{range_id}/complete", self.base_url))
            .bearer_auth(&self.token)
            .json(req)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}
