use namebreak_protocol::{ClaimResponse, CompleteRequest, RegisterRequest, RegisterResponse};

#[derive(Clone)]
pub struct ApiClient {
    http: reqwest::Client,
    base_url: String,
    token: String,
}

impl ApiClient {
    pub async fn register(base_url: &str, username: &str, hostname: &str) -> anyhow::Result<Self> {
        let http = reqwest::Client::new();
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

    pub async fn heartbeat(&self, range_id: i64) -> anyhow::Result<()> {
        self.http
            .post(format!("{}/api/v1/ranges/{range_id}/heartbeat", self.base_url))
            .bearer_auth(&self.token)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn complete(&self, range_id: i64, req: &CompleteRequest) -> anyhow::Result<()> {
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
