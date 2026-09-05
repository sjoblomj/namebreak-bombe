use clap::Parser;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(about = "Volunteer client for the namebreak coordinator - claims ranges and runs the local CUDA namebreak binary against them")]
pub struct Args {
    /// Base URL of the coordinator server, e.g. https://namebreak.fly.dev
    #[arg(long, env = "NAMEBREAK_SERVER_URL")]
    pub server_url: String,

    /// Rudimentary identity - no password, just used for attribution/leases.
    #[arg(long, env = "NAMEBREAK_USERNAME")]
    pub username: String,

    /// Defaults to this machine's hostname.
    #[arg(long, env = "NAMEBREAK_HOSTNAME")]
    pub hostname: Option<String>,

    /// Path to the locally-built `namebreak` CUDA binary.
    #[arg(long, env = "NAMEBREAK_BIN", default_value = "./namebreak")]
    pub namebreak_bin: PathBuf,

    /// Working directory the binary is run in (it writes matches.txt there).
    #[arg(long, env = "NAMEBREAK_WORKDIR", default_value = ".")]
    pub workdir: PathBuf,

    /// How long to wait between /claim attempts when no work is available.
    #[arg(long, default_value_t = 30)]
    pub poll_interval_secs: u64,
}

pub fn resolve_hostname(given: Option<String>) -> String {
    if let Some(h) = given {
        return h;
    }
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}
