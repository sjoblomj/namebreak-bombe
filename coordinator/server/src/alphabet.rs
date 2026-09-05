//! Candidate <-> index math, ported from `namebreaker-cuda/cpu-utils.cpp`
//! (`indexToCandidate` / `getLowerBound` / `getUpperBound`), plus the bound-string
//! generation a `namebreak bounded` invocation needs.
//!
//! Every function here takes the alphabet as a parameter rather than reading a
//! single global constant: each target picks one of `PREDEFINED_ALPHABETS`, whose
//! *characters* namebreak.cu accepts directly as a new CLI argument (no lookup
//! needed on that side) - but whose *size* is restricted to a small fixed set
//! namebreak.cu has compiled-in template instantiations for (see the comment on
//! `indexToCandidate` in namebreak.cu and the dispatch in `runCudaBatch`), to keep
//! that per-thread decode step a compile-time-constant division rather than a
//! (much slower) runtime one. The set of *distinct sizes* below must stay in sync
//! with that dispatch; adding a same-size profile needs no C++ change at all.

/// name -> characters. Sizes present here (42, 43, 47, 48, 49, 50) must match the
/// sizes `namebreak.cu`'s `runCudaBatch` has compiled-in kernel instantiations
/// for. `size49` is relied on elsewhere (`handlers::admin_create_target`'s
/// fallback when `alphabet_name` is omitted) - keep that name stable even if its
/// characters or position here ever change.
pub const PREDEFINED_ALPHABETS: &[(&str, &str)] = &[
    ("size50", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_"),
    ("size49", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"),
    ("size48", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_"),
    ("size47", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"),
    ("size43", " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_"),
    ("size42", " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"),
];

pub fn lookup_predefined_alphabet(name: &str) -> Option<&'static str> {
    PREDEFINED_ALPHABETS.iter().find(|(n, _)| *n == name).map(|(_, chars)| *chars)
}

pub fn alphabet_size(alphabet: &str) -> i64 {
    alphabet.chars().count() as i64
}

fn alphabet_chars(alphabet: &str) -> Vec<char> {
    alphabet.chars().collect()
}

/// The largest candidate length whose full space (`alphabet_size(alphabet)^len`)
/// still fits in an `i64`, so a range can be represented as a flat
/// `[start_index, end_index)` pair without overflow. `namebreak.cu` sidesteps this
/// same limit (there, for u64) by splitting long candidates into a CPU-enumerated
/// leading part and a GPU-indexed trailing part; the server doesn't replicate that
/// split, so target `max_len` is capped at this value. In practice this is a
/// non-issue: exhaustively searching anywhere near this length is already
/// computationally infeasible. A smaller alphabet permits a larger max length.
pub fn max_supported_len(alphabet: &str) -> i64 {
    let size = alphabet_size(alphabet) as i128;
    let mut len = 0i64;
    let mut space: i128 = 1;
    while space.saturating_mul(size) <= i64::MAX as i128 {
        space *= size;
        len += 1;
    }
    len
}

/// Total number of distinct candidates of the given length. Caller must ensure
/// `len <= max_supported_len(alphabet)`.
pub fn space_size(alphabet: &str, len: i64) -> i64 {
    let size = alphabet_size(alphabet) as i128;
    let mut space: i128 = 1;
    for _ in 0..len {
        space *= size;
    }
    space as i64
}

/// Same enumeration order as `namebreak.cu`'s `indexToCandidate`: most-significant
/// character first, base-`alphabet_size(alphabet)` positional encoding.
pub fn index_to_candidate(alphabet: &str, mut index: i64, len: i64) -> String {
    let chars = alphabet_chars(alphabet);
    let size = alphabet_size(alphabet);
    let mut buf = vec![' '; len as usize];
    for i in (0..len as usize).rev() {
        let digit = (index % size) as usize;
        buf[i] = chars[digit];
        index /= size;
    }
    buf.into_iter().collect()
}

/// Inverse of `index_to_candidate`. Returns `None` if `candidate` contains a
/// character outside the alphabet (or is implausibly long enough to overflow).
pub fn candidate_to_index(alphabet: &str, candidate: &str) -> Option<i64> {
    let chars = alphabet_chars(alphabet);
    let size = alphabet_size(alphabet);
    let mut index: i64 = 0;
    for ch in candidate.chars() {
        let digit = chars.iter().position(|&c| c == ch)? as i64;
        index = index.checked_mul(size)?.checked_add(digit)?;
    }
    Some(index)
}

/// Strips a target's prefix/suffix off a full filename to recover the candidate
/// portion, e.g. for turning a `namebreak`-reported match back into an index via
/// `candidate_to_index`. Alphabet-independent - pure string surgery.
pub fn strip_prefix_suffix<'a>(filename: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    filename.strip_prefix(prefix)?.strip_suffix(suffix)
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
/// tested. So passing `index_to_candidate(alphabet, end_index - 1, len)` here
/// reproduces a half-open `[start_index, end_index)` range with no gap or overlap
/// at the boundary between adjacent ranges.
pub fn range_bound_filenames(
    alphabet: &str,
    prefix: &str,
    suffix: &str,
    len: i64,
    start_index: i64,
    end_index: i64,
) -> (String, String) {
    let lower = index_to_candidate(alphabet, start_index, len);
    let upper = index_to_candidate(alphabet, end_index - 1, len);
    (format!("{prefix}{lower}{suffix}"), format!("{prefix}{upper}{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT: &str = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
    const SIZE42: &str = " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    // Not one of PREDEFINED_ALPHABETS - just small enough to give
    // max_supported_len_grows_as_alphabet_shrinks below a real gap to detect,
    // independent of exactly which sizes the predefined table happens to have.
    const TINY: &str = "0123456789ABCDEF";

    #[test]
    fn predefined_alphabets_are_looked_up_by_name() {
        assert_eq!(lookup_predefined_alphabet("size49"), Some(DEFAULT));
        assert_eq!(lookup_predefined_alphabet("size42"), Some(SIZE42));
        assert_eq!(lookup_predefined_alphabet("no-such-alphabet"), None);
    }

    #[test]
    fn predefined_alphabet_sizes_match_their_declared_names() {
        // Sanity check on the table itself - not exhaustive, but catches an obvious typo.
        for &(name, chars) in PREDEFINED_ALPHABETS {
            let size = alphabet_size(chars);
            assert!(size > 0, "{name} has an empty alphabet");
            let unique: std::collections::HashSet<char> = chars.chars().collect();
            assert_eq!(unique.len() as i64, size, "{name} has duplicate characters, which would break the index<->candidate mapping");
        }
    }

    #[test]
    fn alphabet_size_matches_cuda_constant() {
        assert_eq!(alphabet_size(DEFAULT), 49);
    }

    #[test]
    fn max_supported_len_space_fits_i64_but_next_length_does_not() {
        let len = max_supported_len(DEFAULT);
        assert!(space_size(DEFAULT, len) > 0);
        // One length further should overflow i64 - confirmed via i128 math directly,
        // since space_size() itself assumes no overflow.
        let size = alphabet_size(DEFAULT) as i128;
        let mut space: i128 = 1;
        for _ in 0..(len + 1) {
            space *= size;
        }
        assert!(space > i64::MAX as i128);
    }

    #[test]
    fn max_supported_len_grows_as_alphabet_shrinks() {
        assert!(max_supported_len(TINY) > max_supported_len(DEFAULT));
    }

    #[test]
    fn index_to_candidate_round_trips_boundaries() {
        assert_eq!(index_to_candidate(DEFAULT, 0, 3), "   ");
        let max_index = space_size(DEFAULT, 3) - 1;
        assert_eq!(index_to_candidate(DEFAULT, max_index, 3), "___");
    }

    #[test]
    fn candidate_to_index_round_trips_with_index_to_candidate() {
        for &index in &[0, 1, 48, 49, 2400, space_size(DEFAULT, 4) - 1] {
            let candidate = index_to_candidate(DEFAULT, index, 4);
            assert_eq!(candidate_to_index(DEFAULT, &candidate), Some(index));
        }
    }

    #[test]
    fn candidate_to_index_works_for_a_non_default_alphabet() {
        for &index in &[0, 1, 41, 42, 1000, space_size(SIZE42, 3) - 1] {
            let candidate = index_to_candidate(SIZE42, index, 3);
            assert_eq!(candidate_to_index(SIZE42, &candidate), Some(index));
        }
    }

    #[test]
    fn candidate_to_index_rejects_out_of_alphabet_characters() {
        assert_eq!(candidate_to_index(DEFAULT, "abc"), None); // lowercase isn't in the alphabet
        assert_eq!(candidate_to_index(SIZE42, "A!"), None); // '!' isn't in SIZE42's reduced punctuation
    }

    #[test]
    fn strip_prefix_suffix_recovers_the_candidate() {
        assert_eq!(strip_prefix_suffix("REZ\\AB.WAV", "REZ\\", ".WAV"), Some("AB"));
        assert_eq!(strip_prefix_suffix("WRONG\\AB.WAV", "REZ\\", ".WAV"), None);
    }

    #[test]
    fn range_bound_filenames_are_inclusive_and_adjacent_ranges_dont_overlap() {
        let (lower1, upper1) = range_bound_filenames(DEFAULT, "PRE", ".SUF", 2, 0, 10);
        let (lower2, upper2) = range_bound_filenames(DEFAULT, "PRE", ".SUF", 2, 10, 20);
        assert_eq!(lower1, format!("PRE{}.SUF", index_to_candidate(DEFAULT, 0, 2)));
        assert_eq!(upper1, format!("PRE{}.SUF", index_to_candidate(DEFAULT, 9, 2)));
        assert_eq!(lower2, format!("PRE{}.SUF", index_to_candidate(DEFAULT, 10, 2)));
        assert_eq!(upper2, format!("PRE{}.SUF", index_to_candidate(DEFAULT, 19, 2)));
    }
}
