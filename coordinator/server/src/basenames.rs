//! The basenames clients send for a dictionary target (see
//! `AdminCreateTargetRequest::send_basenames`), kept compressed: three words
//! of english-1 make about a million, 31 MB as text. The new ones of each
//! report, those the target hadn't had, are one gzip member, one per line in
//! the order they were sent (`basename_reports`), and each basename's 8-byte
//! hash (`basename_hashes`) is what keeps it once per target. A million,
//! sent 240 a heartbeat: 33 MB in the database, 17 of reports and 15 of
//! hashes, where a row each took 100 MB. See
//! migrations/0032_compressed_basenames.sql.

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::error::AppError;

/// A basename's key in `basename_hashes`: FNV-1a 64 of its bytes. Two
/// basenames of a million share one with a chance of about 3 in 10^8 - and
/// then the second is taken for the first and not kept. Stored, so it can
/// never change.
pub fn hash(basename: &str) -> i64 {
    basename.bytes().fold(0xCBF2_9CE4_8422_2325u64, |hash, b| (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3)) as i64
}

/// `names` one per line, gzipped: one gzip member, so the reports' blobs
/// concatenated are a gzip file of all of them.
pub fn compress(names: &[String]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    for name in names {
        encoder.write_all(name.as_bytes()).and_then(|_| encoder.write_all(b"\n")).expect("writing to a Vec can't fail");
    }
    encoder.finish().expect("writing to a Vec can't fail")
}

/// The basenames `compress` made `blob` of, in order.
pub fn decompress(blob: &[u8]) -> Result<Vec<String>, AppError> {
    let mut text = String::new();
    GzDecoder::new(blob).read_to_string(&mut text).map_err(|err| AppError::Internal(format!("stored basenames don't decompress: {err}")))?;
    Ok(text.lines().map(str::to_string).collect())
}

/// Keeps those of `basenames` (already checked to be basenames) that target
/// `target_id` hasn't had, as one report of `user_id`'s with `range_id` -
/// none if there are none. How many were new.
pub async fn store(
    tx: &mut sqlx::SqliteConnection,
    target_id: i64,
    user_id: i64,
    range_id: i64,
    now: i64,
    basenames: &[&str],
) -> Result<usize, AppError> {
    let mut new = Vec::new();
    for &basename in basenames {
        let inserted = sqlx::query("INSERT OR IGNORE INTO basename_hashes (target_id, hash) VALUES (?, ?)")
            .bind(target_id)
            .bind(hash(basename))
            .execute(&mut *tx)
            .await?
            .rows_affected();
        if inserted > 0 {
            new.push(basename.to_string());
        }
    }
    if new.is_empty() {
        return Ok(0);
    }
    sqlx::query("INSERT INTO basename_reports (target_id, user_id, range_id, reported_at, count, names) VALUES (?, ?, ?, ?, ?, ?)")
        .bind(target_id)
        .bind(user_id)
        .bind(range_id)
        .bind(now)
        .bind(new.len() as i64)
        .bind(compress(&new))
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE targets SET basename_count = basename_count + ? WHERE id = ?").bind(new.len() as i64).bind(target_id).execute(&mut *tx).await?;
    Ok(new.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_round_trips_and_compresses() {
        let names: Vec<String> = (0..5000).map(|i| format!("WORD{}_OTHER{}-THIRD{}.WAV", i % 97, i % 89, i)).collect();
        let blob = compress(&names);
        assert_eq!(decompress(&blob).unwrap(), names);
        let text: usize = names.iter().map(|n| n.len() + 1).sum();
        assert!(blob.len() * 3 < text, "{} bytes for {} of text", blob.len(), text);
        assert_eq!(&blob[..2], &[0x1F, 0x8B], "a gzip member");
        assert_eq!(decompress(&compress(&[])).unwrap(), Vec::<String>::new());
        assert!(matches!(decompress(b"not gzip"), Err(AppError::Internal(_))));
    }

    #[test]
    fn blobs_together_are_one_gzip_file() {
        let (a, b) = (vec!["A.WAV".to_string()], vec!["B.WAV".to_string(), "C.WAV".to_string()]);
        let joined = [compress(&a), compress(&b)].concat();
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(&joined[..]).read_to_string(&mut text).unwrap();
        assert_eq!(text, "A.WAV\nB.WAV\nC.WAV\n");
    }

    #[test]
    fn hash_is_fnv_1a_and_never_changes() {
        // FNV-1a 64 of "" and "a", the published values (as dictionary.rs's).
        assert_eq!(hash("") as u64, 0xCBF2_9CE4_8422_2325);
        assert_eq!(hash("a") as u64, 0xAF63_DC4C_8601_EC8C);
        assert_ne!(hash("A.WAV"), hash("B.WAV"));
    }
}
