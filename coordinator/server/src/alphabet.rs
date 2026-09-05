//! Candidate <-> index math, ported from `namebreaker-cuda/cpu-utils.cpp`
//! (`indexToCandidate` / `getLowerBound` / `getUpperBound`), plus the bound-string
//! generation a `namebreak bounded` invocation needs.
//!
//! Must stay byte-for-byte in sync with the alphabet in `namebreak.cu`'s
//! `d_alphabet` / `alphabet`.

/// Same 49-character alphabet as `namebreak.cu`.
pub const ALPHABET: &str = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";

pub fn alphabet_size() -> i64 {
    ALPHABET.chars().count() as i64
}

fn alphabet_chars() -> Vec<char> {
    ALPHABET.chars().collect()
}

/// The largest candidate length whose full space (`alphabet_size()^len`) still fits
/// in an `i64`, so a range can be represented as a flat `[start_index, end_index)`
/// pair without overflow. `namebreak.cu` sidesteps this same limit (there, for u64)
/// by splitting long candidates into a CPU-enumerated leading part and a GPU-indexed
/// trailing part; the server doesn't replicate that split, so target `max_len` is
/// capped at this value. In practice this is a non-issue: exhaustively searching
/// anywhere near this length is already computationally infeasible.
pub fn max_supported_len() -> i64 {
    let size = alphabet_size() as i128;
    let mut len = 0i64;
    let mut space: i128 = 1;
    while space.saturating_mul(size) <= i64::MAX as i128 {
        space *= size;
        len += 1;
    }
    len
}

/// Total number of distinct candidates of the given length. Caller must ensure
/// `len <= max_supported_len()`.
pub fn space_size(len: i64) -> i64 {
    let size = alphabet_size() as i128;
    let mut space: i128 = 1;
    for _ in 0..len {
        space *= size;
    }
    space as i64
}

/// Same enumeration order as `namebreak.cu`'s `indexToCandidate`: most-significant
/// character first, base-`alphabet_size()` positional encoding.
pub fn index_to_candidate(mut index: i64, len: i64) -> String {
    let chars = alphabet_chars();
    let size = alphabet_size();
    let mut buf = vec![' '; len as usize];
    for i in (0..len as usize).rev() {
        let digit = (index % size) as usize;
        buf[i] = chars[digit];
        index /= size;
    }
    buf.into_iter().collect()
}

/// Builds the `(lowerBoundFilename, upperBoundFilename)` pair to pass to
/// `namebreak bounded` so it covers exactly the half-open range
/// `[start_index, end_index)` at the given candidate length.
///
/// Both returned filenames are *inclusive* bounds for `namebreak bounded`: its
/// start argument is used as the literal first candidate tested, and its upper
/// bound argument is run through `getUpperBound` (successor-of-max-padded), which
/// for a same-length literal candidate resolves to exactly "index of that literal
/// plus one" - i.e. the literal upper-bound candidate itself is the last one
/// tested. So passing `index_to_candidate(end_index - 1, len)` here reproduces a
/// half-open `[start_index, end_index)` range with no gap or overlap at the
/// boundary between adjacent ranges.
pub fn range_bound_filenames(
    prefix: &str,
    suffix: &str,
    len: i64,
    start_index: i64,
    end_index: i64,
) -> (String, String) {
    let lower = index_to_candidate(start_index, len);
    let upper = index_to_candidate(end_index - 1, len);
    (format!("{prefix}{lower}{suffix}"), format!("{prefix}{upper}{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphabet_size_matches_cuda_constant() {
        assert_eq!(alphabet_size(), 49);
    }

    #[test]
    fn max_supported_len_space_fits_i64_but_next_length_does_not() {
        let len = max_supported_len();
        assert!(space_size(len) > 0);
        // One length further should overflow i64 - confirmed via i128 math directly,
        // since space_size() itself assumes no overflow.
        let size = alphabet_size() as i128;
        let mut space: i128 = 1;
        for _ in 0..(len + 1) {
            space *= size;
        }
        assert!(space > i64::MAX as i128);
    }

    #[test]
    fn index_to_candidate_round_trips_boundaries() {
        assert_eq!(index_to_candidate(0, 3), "   ");
        let max_index = space_size(3) - 1;
        assert_eq!(index_to_candidate(max_index, 3), "___");
    }

    #[test]
    fn range_bound_filenames_are_inclusive_and_adjacent_ranges_dont_overlap() {
        let (lower1, upper1) = range_bound_filenames("PRE", ".SUF", 2, 0, 10);
        let (lower2, upper2) = range_bound_filenames("PRE", ".SUF", 2, 10, 20);
        assert_eq!(lower1, format!("PRE{}.SUF", index_to_candidate(0, 2)));
        assert_eq!(upper1, format!("PRE{}.SUF", index_to_candidate(9, 2)));
        assert_eq!(lower2, format!("PRE{}.SUF", index_to_candidate(10, 2)));
        assert_eq!(upper2, format!("PRE{}.SUF", index_to_candidate(19, 2)));
    }
}
