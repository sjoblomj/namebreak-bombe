//! Meet-in-the-middle targets: a file whose name is a known prefix, an unknown
//! *directory* and a known suffix, where the MPQ hash state right after the
//! directory - the *meeting point* - is already known, for hashA and for
//! hashB.
//! It's the client's meet-in-the-middle mode (see the client's
//! docs/meet-in-the-middle.md), shared out.
//!
//! The client searches such a directory from both ends at once: forward from
//! the prefix into a table of every directory of the first `table_length`
//! characters, by the state it reaches, and backward from the meeting point,
//! every *tail* - the directory's last characters - rewound character by
//! character and looked up in that table. So a directory of `length`
//! characters is one of the table's prefixes followed by a tail of `length -
//! table_length` characters, and the work is the tails, not the directories:
//! there are `alphabet^(length - table_length)` of them.
//!
//! That's what a meet-in-the-middle target's ranges are. A range is a stretch
//! of one length's tails: `candidate_len` is the directory length, and
//! `start_index`/`end_index` are tail indices (always block 0, as a
//! dictionary target's candidate numbers are - see
//! `models::MEET_IN_THE_MIDDLE_ALPHABET_NAME`). The tails of a length are
//! numbered as the client numbers them: by the tail *reversed*, so the
//! directory's last character turns slowest, because it's the first the
//! client rewinds and the one the most tails share.
//!
//! The table is what the search costs in memory, and the server fixes its
//! length per target: the ranges mean nothing without it, since a tail is
//! whatever the table doesn't cover. A client says how much memory it will
//! give a table (`users.meet_in_the_middle_table_bytes`), and gets no work
//! from a target whose table wouldn't fit in it - the same way a client that
//! can't search a target's alphabet gets none (see `ranges::claim_range`).
//! For volunteers with less memory, make a second target on the same hashes
//! with a shorter table.

use crate::error::AppError;
use crate::models::{i64_to_u32, Target};
use crate::mpq::{continue_hash, HashState, HASH_A_OFFSET, HASH_B_OFFSET};

/// The longest directory a meet-in-the-middle target can search: the client's
/// `MAX_CANDIDATE_LEN`.
pub const MAX_DIRECTORY_LENGTH: i64 = 16;
/// The longest filename a client searches: its `MAX_FILENAME_LEN`.
pub const MAX_FILENAME_LENGTH: usize = 128;
/// The most characters an alphabet may have: the client's
/// `MAX_ALPHABET_SIZE`.
pub const MAX_ALPHABET_SIZE: usize = 63;

/// How many bits of filter the client's table gives each key - its
/// `MeetInTheMiddleTable::kFilterBitsPerKey`.
const FILTER_BITS_PER_KEY: u64 = 8;
/// The most bits of filter it ever makes, whatever the table's size.
const MAX_FILTER_BITS: u64 = 1 << 32;

/// What a table of `table_length` characters of an `alphabet_size`-character
/// alphabet takes in memory, in bytes: eight bytes a directory for the keys,
/// plus the filter in front of them. `u64::MAX` for a length whose table
/// can't be counted at all. Must agree with the client's
/// `MeetInTheMiddleTable::bytesFor`, since this is what a client's declared
/// budget is held against.
pub fn table_bytes(alphabet_size: usize, table_length: i64) -> u64 {
    let Some(keys) = (alphabet_size as u64).checked_pow(table_length.max(0) as u32) else {
        return u64::MAX;
    };
    let Some(key_bytes) = keys.checked_mul(8) else {
        return u64::MAX;
    };
    key_bytes.saturating_add(filter_bits(keys) / 8)
}

/// The filter's size in bits for a table of `keys` keys - the client's
/// `filterBitsFor`: about `FILTER_BITS_PER_KEY` a key, rounded up to a power
/// of two (the filter is indexed by the top bits of a mixed key), at least
/// one word and at most `MAX_FILTER_BITS`.
fn filter_bits(keys: u64) -> u64 {
    let wanted = keys.saturating_mul(FILTER_BITS_PER_KEY);
    let mut bits = 64;
    while bits < wanted && bits < MAX_FILTER_BITS {
        bits *= 2;
    }
    bits
}

/// The biggest table a target may ask for. Not a technical limit - the
/// client would hold a bigger one given the memory - but a target whose
/// table no volunteer will give is a target nobody can search, and the
/// mistake (a table one character longer than meant, which is dozens of
/// times bigger) is an easy one to make.
pub const MAX_TABLE_BYTES: u64 = 64 << 30;

/// The longest table that fits in `bytes` for an `alphabet_size`-character
/// alphabet, up to `max_length` - 0 if not even one character's does. For
/// telling an operator what a budget allows.
pub fn table_length_for_bytes(alphabet_size: usize, max_length: i64, bytes: u64) -> i64 {
    (1..=max_length).rev().find(|&length| table_bytes(alphabet_size, length) <= bytes).unwrap_or(0)
}

/// A stretch of one directory length's tails, `[start, end)` - everything a
/// search of that length is handed. The server carves ranges from these, as
/// it does from a dictionary target's windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub length: i64,
    pub start: i64,
    pub end: i64,
}

/// Which directories a meet-in-the-middle target covers, and how its tails are
/// numbered - the server's half of the client's `MeetInTheMiddleSpace`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeetInTheMiddleSpace {
    alphabet: String,
    min_length: i64,
    max_length: i64,
    table_length: i64,
}

impl MeetInTheMiddleSpace {
    /// An `Err` with a human-readable reason for settings no search can be
    /// made from - which only a target being created can have.
    pub fn new(alphabet: &str, min_length: i64, max_length: i64, table_length: i64) -> Result<Self, String> {
        let characters: Vec<char> = alphabet.chars().collect();
        if characters.is_empty() {
            return Err("the alphabet is empty".into());
        }
        if characters.len() > MAX_ALPHABET_SIZE {
            return Err(format!("the alphabet has {} characters - at most {MAX_ALPHABET_SIZE}", characters.len()));
        }
        if let Some(twice) = characters.iter().enumerate().find_map(|(i, c)| characters[i + 1..].contains(c).then_some(c)) {
            return Err(format!("the alphabet has '{twice}' more than once"));
        }
        if min_length < 1 {
            return Err("the shortest directory must be at least one character".into());
        }
        if max_length < min_length {
            return Err(format!("the longest directory ({max_length} characters) is shorter than the shortest ({min_length})"));
        }
        if max_length > MAX_DIRECTORY_LENGTH {
            return Err(format!("the longest directory is {max_length} characters - at most {MAX_DIRECTORY_LENGTH}"));
        }
        if table_length < 0 {
            return Err("the table can't be a negative number of characters long".into());
        }
        // Every directory is at least as long as the table, so that the tails
        // of each length are the whole of its work. A shorter directory
        // couldn't be handed out as a range at all: it has no tail, and the
        // client only ever finds one while building its table.
        if table_length > min_length {
            return Err(format!(
                "the table's {table_length} characters are more than the shortest directory's {min_length} - a directory shorter than the \
                 table has no tail to hand out, so search those with an ordinary alphabet target (there are few of them)"
            ));
        }
        let space = MeetInTheMiddleSpace { alphabet: alphabet.to_string(), min_length, max_length, table_length };
        let bytes = space.table_bytes();
        if bytes > MAX_TABLE_BYTES {
            return Err(format!(
                "a table of {table_length} characters of a {}-character alphabet is {} - more than the {} GiB a volunteer could be asked \
                 for. The longest that fits in 2 GiB is {} characters; every character divides the tails to search by {} and multiplies \
                 the table by it",
                characters.len(),
                if bytes == u64::MAX { "past counting".to_string() } else { format!("{} GiB", bytes / (1 << 30)) },
                MAX_TABLE_BYTES / (1 << 30),
                table_length_for_bytes(characters.len(), max_length, 2 << 30),
                characters.len()
            ));
        }
        // The client numbers a length's tails in 64 bits, and the database
        // keeps an index in one - see
        // `models::MEET_IN_THE_MIDDLE_ALPHABET_NAME`.
        if (characters.len() as u64).checked_pow((max_length - table_length) as u32).is_none_or(|tails| tails > i64::MAX as u64) {
            return Err(format!(
                "too many tails to number: {} characters to the power of {} (the table's {table_length} leave that many of a directory's \
                 {max_length} to search) doesn't fit in 64 bits - search shorter directories, or give the table more characters",
                characters.len(),
                max_length - table_length
            ));
        }
        Ok(space)
    }

    pub fn alphabet(&self) -> &str {
        &self.alphabet
    }

    pub fn alphabet_size(&self) -> usize {
        self.alphabet.chars().count()
    }

    pub fn min_length(&self) -> i64 {
        self.min_length
    }

    pub fn max_length(&self) -> i64 {
        self.max_length
    }

    pub fn table_length(&self) -> i64 {
        self.table_length
    }

    /// What this target's table takes in memory - what a client must be
    /// willing to give one to get any of this work.
    pub fn table_bytes(&self) -> u64 {
        table_bytes(self.alphabet_size(), self.table_length)
    }

    /// How many characters of a directory of `length` the client rewinds:
    /// the ones the table doesn't hold.
    pub fn tail_length(&self, length: i64) -> i64 {
        length - self.table_length
    }

    /// How many tails a directory length has - `MeetInTheMiddleSpace::new` has
    /// made sure this fits.
    pub fn tails_at(&self, length: i64) -> i64 {
        (self.alphabet_size() as i64).pow(self.tail_length(length) as u32)
    }

    /// Every length's tails, shortest directory first - the windows a claim
    /// carves from.
    pub fn windows(&self) -> Vec<Window> {
        (self.min_length..=self.max_length).map(|length| Window { length, start: 0, end: self.tails_at(length) }).collect()
    }

    /// How many tails the whole target is, all lengths together.
    pub fn total_tails(&self) -> i128 {
        self.windows().iter().map(|w| (w.end - w.start) as i128).sum()
    }

    /// The tail of `index` at directory `length`: the directory's last
    /// characters, in the directory's own order. Numbered by the tail
    /// reversed, as the client numbers them (its `tailOf`).
    pub fn tail(&self, length: i64, index: i64) -> String {
        let characters: Vec<char> = self.alphabet.chars().collect();
        let size = characters.len() as i64;
        let mut rest = index;
        // The least significant digit is the tail's first character.
        let mut tail: Vec<char> = Vec::new();
        for _ in 0..self.tail_length(length) {
            tail.push(characters[(rest % size) as usize]);
            rest /= size;
        }
        tail.into_iter().collect()
    }

    /// Where `tail` is among the tails of its length - the inverse of
    /// `tail`. `None` if it has a character the alphabet doesn't.
    pub fn tail_index(&self, tail: &str) -> Option<i64> {
        let characters: Vec<char> = self.alphabet.chars().collect();
        let size = characters.len() as i64;
        // The tail's first character is the least significant digit, so the
        // directory's last character turns slowest.
        let mut index = 0;
        for c in tail.chars().rev() {
            index = index * size + characters.iter().position(|&a| a == c)? as i64;
        }
        Some(index)
    }

    /// A tail with the table's characters as '?' - what a whole directory
    /// looks like at that point of the search, for a human.
    pub fn directory(&self, length: i64, index: i64) -> String {
        "?".repeat(self.table_length as usize) + &self.tail(length, index)
    }
}

/// A meet-in-the-middle target as the server works with it: the space its
/// ranges are carved from, and the meeting point itself. All of it is on the
/// `targets` row, so - unlike a dictionary target's word lists - there's
/// nothing to read or cache.
#[derive(Debug, Clone)]
pub struct MeetInTheMiddleTarget {
    pub space: MeetInTheMiddleSpace,
    pub meeting_point_a: HashState,
    pub meeting_point_b: HashState,
}

/// Reads a meet-in-the-middle target's settings off its row. An `Internal`
/// error for a row that doesn't make a space, which creating the target has
/// already ruled out.
pub fn of_target(target: &Target) -> Result<MeetInTheMiddleTarget, AppError> {
    let columns = (target.meet_in_the_middle_max_length, target.meet_in_the_middle_table_length, target.meeting_point_a_seed1, target.meeting_point_a_seed2, target.meeting_point_b_seed1, target.meeting_point_b_seed2);
    let (Some(max_length), Some(table_length), Some(a1), Some(a2), Some(b1), Some(b2)) = columns else {
        return Err(AppError::Internal(format!("target {} is a meet-in-the-middle target without its meet-in-the-middle settings", target.id)));
    };
    let space = MeetInTheMiddleSpace::new(&target.alphabet, target.start_len, max_length, table_length)
        .map_err(|err| AppError::Internal(format!("target {}'s meet-in-the-middle settings don't make a search: {err}", target.id)))?;
    Ok(MeetInTheMiddleTarget { space, meeting_point_a: (i64_to_u32(a1), i64_to_u32(a2)), meeting_point_b: (i64_to_u32(b1), i64_to_u32(b2)) })
}

/// Whether a meeting point and the file's hashes agree: hashing the suffix
/// onto the meeting point must give the hash, for hashA and for hashB. They
/// come from different places - the meeting point from the other files of
/// its directory, the hashes from the archive - and a meeting point that
/// doesn't fit the hashes would have every client search the whole space and
/// find nothing, so a target is refused unless they do. The client checks
/// the same thing before it starts (`meetingPointFitsHashes`).
pub fn fits_hashes(meeting_point_a: HashState, meeting_point_b: HashState, suffix: &str, hash_a: u32, hash_b: u32) -> Result<(), String> {
    for (which, meeting_point, offset, hash) in
        [("hash_a", meeting_point_a, HASH_A_OFFSET, hash_a), ("hash_b", meeting_point_b, HASH_B_OFFSET, hash_b)]
    {
        let reached = continue_hash(meeting_point, suffix, offset).0;
        if reached != hash {
            return Err(format!(
                "the meeting point for {which} ({:08X} {:08X}) doesn't fit the file: hashing the suffix '{suffix}' onto it gives \
                 {reached:08X}, not the {which} of {hash:08X} - the meeting point must be the state after the directory, right where the \
                 suffix begins",
                meeting_point.0, meeting_point.1
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpq::INITIAL_STATE;

    const REAL: &str = " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_";

    #[test]
    fn table_bytes_match_the_clients() {
        // The same figures as the client's docs/meet-in-the-middle.md
        // table, and its meet_in_the_middle_test's - keys plus the filter in
        // front of them.
        assert_eq!(REAL.chars().count(), 43);
        assert_eq!(table_bytes(43, 0), 8 + 8);
        assert_eq!(table_bytes(43, 3), 79_507 * 8 + (1 << 20) / 8);
        assert_eq!(table_bytes(43, 4), 3_418_801 * 8 + (1 << 25) / 8);
        assert_eq!(table_bytes(43, 5), 147_008_443 * 8 + (1 << 31) / 8);
        // Between 1 and 2 GiB at five characters, 48 GiB at six.
        assert!((1 << 30) < table_bytes(43, 5) && table_bytes(43, 5) < (2 << 30));
        assert_eq!(table_bytes(43, 6) / (1 << 30), 47);
        // Past counting.
        assert_eq!(table_bytes(43, 16), u64::MAX);
    }

    #[test]
    fn table_length_for_bytes_is_the_longest_that_fits() {
        for bytes in [0, 1024, 1 << 20, 1 << 30, 2u64 << 30, 8u64 << 30] {
            let length = table_length_for_bytes(43, 12, bytes);
            assert!(length == 0 || table_bytes(43, length) <= bytes, "{length} in {bytes}");
            assert!(length == 12 || table_bytes(43, length + 1) > bytes, "{length} in {bytes}");
        }
        assert_eq!(table_length_for_bytes(43, 12, 1 << 30), 4, "a gibibyte holds four characters of 43");
        assert_eq!(table_length_for_bytes(43, 12, 2 << 30), 5, "two hold five");
        assert_eq!(table_length_for_bytes(43, 3, 8 << 30), 3, "never longer than asked for");
    }

    #[test]
    fn a_space_refuses_what_cant_be_searched() {
        assert!(MeetInTheMiddleSpace::new("", 1, 4, 1).is_err(), "no alphabet");
        assert!(MeetInTheMiddleSpace::new("AAB", 1, 4, 1).unwrap_err().contains("more than once"));
        assert!(MeetInTheMiddleSpace::new(&"x".repeat(64), 1, 4, 1).is_err(), "too big an alphabet");
        assert!(MeetInTheMiddleSpace::new("AB", 0, 4, 0).is_err(), "no shortest directory");
        assert!(MeetInTheMiddleSpace::new("AB", 5, 4, 1).is_err(), "shortest past the longest");
        assert!(MeetInTheMiddleSpace::new("AB", 1, MAX_DIRECTORY_LENGTH + 1, 1).is_err(), "too long a directory");
        assert!(MeetInTheMiddleSpace::new("AB", 2, 4, 3).unwrap_err().contains("shortest directory"), "a table longer than the directories");
        // A table nobody would hold, with what does fit in the message.
        let too_big = MeetInTheMiddleSpace::new(REAL, 8, 8, 7).unwrap_err();
        assert!(too_big.contains("more than the 64 GiB") && too_big.contains("is 5 characters"), "{too_big}");
        assert!(MeetInTheMiddleSpace::new(REAL, 6, 12, 6).is_ok(), "47 GiB is under the cap, where seven characters' 2 TiB isn't");
        // 43 characters to the power of 12 is past 63 bits; to the power of
        // 11 isn't.
        assert!(MeetInTheMiddleSpace::new(REAL, 12, 12, 0).unwrap_err().contains("64 bits"));
        assert!(MeetInTheMiddleSpace::new(REAL, 11, 11, 0).is_ok());
        assert!(MeetInTheMiddleSpace::new(REAL, 13, 13, 2).is_ok(), "a table makes a long directory searchable");
    }

    #[test]
    fn windows_are_every_lengths_tails() {
        let space = MeetInTheMiddleSpace::new("AB-", 3, 5, 2).unwrap();
        assert_eq!(space.tail_length(3), 1);
        assert_eq!(
            space.windows(),
            vec![Window { length: 3, start: 0, end: 3 }, Window { length: 4, start: 0, end: 9 }, Window { length: 5, start: 0, end: 27 }]
        );
        assert_eq!(space.total_tails(), 3 + 9 + 27);
        assert_eq!(space.table_bytes(), table_bytes(3, 2));
    }

    #[test]
    fn tails_are_numbered_by_the_reversed_tail() {
        // The directory's last character turns slowest: the tails of two
        // characters of "AB-" are AA, BA, -A, AB, BB, -B, A-, B-, --.
        let space = MeetInTheMiddleSpace::new("AB-", 4, 4, 2).unwrap();
        let tails: Vec<String> = (0..space.tails_at(4)).map(|i| space.tail(4, i)).collect();
        assert_eq!(tails, ["AA", "BA", "-A", "AB", "BB", "-B", "A-", "B-", "--"]);
        assert_eq!(space.directory(4, 4), "??BB");
        // Every tail once, and the number back again.
        let mut sorted = tails.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), tails.len());
    }

    #[test]
    fn tail_index_is_the_inverse_of_tail() {
        let space = MeetInTheMiddleSpace::new("AB-\\", 5, 6, 3).unwrap();
        for length in [5, 6] {
            for index in 0..space.tails_at(length) {
                assert_eq!(space.tail_index(&space.tail(length, index)), Some(index));
            }
        }
        assert_eq!(space.tail_index("AZ"), None, "a character the alphabet hasn't");
    }

    #[test]
    fn fits_hashes_holds_the_meet_in_the_middle_against_the_file() {
        let dir = "SETUPDAT";
        let suffix = "\\BWUNIN.PIF";
        let meeting_point_a = continue_hash(INITIAL_STATE, dir, HASH_A_OFFSET);
        let meeting_point_b = continue_hash(INITIAL_STATE, dir, HASH_B_OFFSET);
        // The hashes of SETUPDAT\BWUNIN.PIF, from another implementation of
        // the MPQ hash altogether (see mpq.rs).
        assert!(fits_hashes(meeting_point_a, meeting_point_b, suffix, 0x5BAB_63FE, 0x8DF2_CB08).is_ok());
        assert!(fits_hashes(meeting_point_a, meeting_point_b, suffix, 0x5BAB_63FF, 0x8DF2_CB08).unwrap_err().contains("hash_a"));
        assert!(fits_hashes(meeting_point_a, meeting_point_b, suffix, 0x5BAB_63FE, 0x8DF2_CB09).unwrap_err().contains("hash_b"));
        // A meeting point for the wrong place in the name.
        assert!(fits_hashes(meeting_point_a, meeting_point_b, "\\BWUNIN.EXE", 0x5BAB_63FE, 0x8DF2_CB08).is_err());
    }
}
