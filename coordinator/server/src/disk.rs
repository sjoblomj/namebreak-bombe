//! How much room the database's disk has left. A dictionary target's
//! basenames are what fill it (see `basenames.rs`), so when it runs low
//! `ranges::record_basenames` discards them rather than let the database -
//! and with it every range and find - run out of room. The dashboard warns
//! well before that, so there's time to make room.

use std::path::{Path, PathBuf};

/// The directory the database is in, how many bytes must stay free on its
/// disk for basenames to be kept, and below how many the dashboard warns
/// that they soon won't be.
#[derive(Default)]
pub struct DiskSpace {
    /// None for an in-memory database, which is never low.
    pub dir: Option<PathBuf>,
    pub min_free_bytes: u64,
    pub warn_free_bytes: u64,
}

/// What the dashboard says about the disk, while it's under
/// `DiskSpace::warn_free_bytes`.
#[derive(Debug, PartialEq, serde::Serialize)]
pub struct LowDisk {
    pub free_bytes: u64,
    /// Below this many free, basenames are discarded.
    pub min_free_bytes: u64,
    /// Whether they are being, now.
    pub discarding: bool,
}

impl DiskSpace {
    /// For the database at `database_url`, keeping `MIN_FREE_DISK_BYTES`
    /// (256 MiB unless set) free and warning below `WARN_FREE_DISK_BYTES`
    /// (512 MiB unless set).
    pub fn from_env(database_url: &str) -> anyhow::Result<Self> {
        use std::str::FromStr;
        let options = sqlx::sqlite::SqliteConnectOptions::from_str(database_url)?;
        let file = options.get_filename();
        // sqlx names an in-memory database "file:sqlx-in-memory-<n>".
        let dir = if file.as_os_str().is_empty() || file.to_string_lossy().starts_with("file:") {
            None
        } else {
            let parent = file.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
            Some(parent.to_path_buf())
        };
        let env_u64 = |key: &str, default: u64| std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        let min_free_bytes = env_u64("MIN_FREE_DISK_BYTES", 256 << 20);
        let warn_free_bytes = env_u64("WARN_FREE_DISK_BYTES", 512 << 20);
        Ok(DiskSpace { dir, min_free_bytes, warn_free_bytes })
    }

    /// The bytes free on the database's disk, if it's too few to keep
    /// basenames - under `min_free_bytes`. None if there's room, or if it
    /// can't be told.
    pub fn low(&self) -> Option<u64> {
        let free = free_bytes(self.dir.as_deref()?)?;
        (free < self.min_free_bytes).then_some(free)
    }

    /// What to warn of, if the disk is getting low - under
    /// `warn_free_bytes`, or `min_free_bytes` should that be more.
    pub fn warning(&self) -> Option<LowDisk> {
        let free = free_bytes(self.dir.as_deref()?)?;
        (free < self.warn_free_bytes.max(self.min_free_bytes))
            .then_some(LowDisk { free_bytes: free, min_free_bytes: self.min_free_bytes, discarding: free < self.min_free_bytes })
    }
}

/// The bytes an unprivileged process may still write on `dir`'s file
/// system. None if it can't be told.
#[cfg(unix)]
fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is NUL-terminated and `stat` is a statvfs to fill in.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        tracing::warn!(dir = %dir.display(), err = %std::io::Error::last_os_error(), "can't tell the free disk space");
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn free_bytes(_dir: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_database_directory() {
        assert_eq!(DiskSpace::from_env("sqlite::memory:").unwrap().dir, None);
        assert_eq!(DiskSpace::from_env("sqlite://namebreak.db").unwrap().dir, Some(PathBuf::from(".")));
        assert_eq!(DiskSpace::from_env("sqlite:///data/namebreak.db").unwrap().dir, Some(PathBuf::from("/data")));
    }

    #[test]
    fn low_when_less_is_free_than_must_be() {
        let here = |min_free_bytes, warn_free_bytes| DiskSpace { dir: Some(PathBuf::from(".")), min_free_bytes, warn_free_bytes };
        let room = here(0, 0);
        assert_eq!((room.low(), room.warning()), (None, None));

        let getting_low = here(0, u64::MAX);
        assert_eq!(getting_low.low(), None, "kept still");
        let warning = getting_low.warning().unwrap();
        assert!(!warning.discarding, "warned before they're discarded");

        let low = here(u64::MAX, 0);
        assert!(low.low().is_some());
        assert!(low.warning().unwrap().discarding, "warned while they are, whatever warn_free_bytes says");

        let in_memory = DiskSpace { dir: None, min_free_bytes: u64::MAX, warn_free_bytes: u64::MAX };
        assert_eq!((in_memory.low(), in_memory.warning()), (None, None));
    }
}
