//! The MPQ hash, as the client computes it (`client/src/engine/mpq_hash.h`).
//! The server needs it to plant a canary's filename (`canary.rs`) and to
//! check a junction target's state against the file's hashes
//! (`junction.rs`): in both, a name the server makes up has to hash to
//! exactly what a client will compute for it.

use std::sync::OnceLock;

/// The crypt-table offsets of the hashes this server works with: hashA
/// (Storm's hash type 1), hashB (2), and the one an encryption key is made
/// with (3) - the client's `kHashAOffset` and friends.
pub const HASH_A_OFFSET: usize = 0x100;
pub const HASH_B_OFFSET: usize = 0x200;
pub const FILE_KEY_OFFSET: usize = 0x300;

/// The MPQ hash's state: (seed1, seed2), and what every hash starts from -
/// the client's `HashState` and `kInitialHashState`.
pub type HashState = (u32, u32);
pub const INITIAL_STATE: HashState = (0x7FED_7FED, 0xEEEE_EEEE);

/// MPQ's crypt table, as the client's `prepareCryptTable` builds it.
fn crypt_table() -> &'static [u32; 0x500] {
    static TABLE: OnceLock<[u32; 0x500]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 0x500];
        let mut seed: u32 = 0x0010_0001;
        for index1 in 0..0x100 {
            let mut index2 = index1;
            for _ in 0..5 {
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                let high = (seed & 0xFFFF) << 16;
                seed = (seed * 125 + 3) % 0x2A_AAAB;
                table[index2] = high | (seed & 0xFFFF);
                index2 += 0x100;
            }
        }
        table
    })
}

/// `state` continued over `text`'s bytes as given, with the crypt table at
/// `offset` - the client's `continueHash`.
pub fn continue_hash(state: HashState, text: &str, offset: usize) -> HashState {
    let table = crypt_table();
    let (mut seed1, mut seed2) = state;
    for byte in text.bytes() {
        seed1 = table[offset + byte as usize] ^ seed1.wrapping_add(seed2);
        seed2 = (byte as u32).wrapping_add(seed1).wrapping_add(seed2).wrapping_add(seed2 << 5).wrapping_add(3);
    }
    (seed1, seed2)
}

/// A name's MPQ hash with the crypt table at `offset` (`HASH_A_OFFSET` for
/// hash A, `HASH_B_OFFSET` for hash B), over its bytes as given - as the
/// client hashes it.
pub fn mpq_hash(name: &str, offset: usize) -> u32 {
    continue_hash(INITIAL_STATE, name, offset).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpq_hash_matches_the_clients() {
        // The client's own (client/src/engine/mpq_hash.cpp) - and (LISTFILE)'s
        // are the well-known ones every MPQ tool has.
        assert_eq!(mpq_hash("(LISTFILE)", HASH_A_OFFSET), 0xFD65_7910);
        assert_eq!(mpq_hash("(LISTFILE)", HASH_B_OFFSET), 0x4E9B_98A7);
        assert_eq!(mpq_hash("REZ\\ABC123.WAV", HASH_A_OFFSET), 0xFCAA_88A2);
        assert_eq!(mpq_hash("REZ\\ABC123.WAV", HASH_B_OFFSET), 0xF559_2C3D);
    }

    #[test]
    fn continue_hash_picks_up_where_it_left_off() {
        // The state after a directory, continued over the rest of the name,
        // is the whole name's hash - what a junction target rests on.
        for offset in [HASH_A_OFFSET, HASH_B_OFFSET, FILE_KEY_OFFSET] {
            let whole = continue_hash(INITIAL_STATE, "SETUPDAT\\BWUNIN.PIF", offset);
            let after_dir = continue_hash(INITIAL_STATE, "SETUPDAT", offset);
            assert_eq!(continue_hash(after_dir, "\\BWUNIN.PIF", offset), whole);
        }
        // And the hashes of that name, worked out by another implementation
        // of the MPQ hash altogether (client/tests/junction_test.cpp has the
        // same two).
        assert_eq!(mpq_hash("SETUPDAT\\BWUNIN.PIF", HASH_A_OFFSET), 0x5BAB_63FE);
        assert_eq!(mpq_hash("SETUPDAT\\BWUNIN.PIF", HASH_B_OFFSET), 0x8DF2_CB08);
    }
}
