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

/// name, characters, and the protocol MINOR version this alphabet was
/// introduced in (see `namebreak_protocol::PROTOCOL_VERSION`'s doc comment).
/// Sizes present here (42, 43, 47, 48, 49, 50) must match the sizes
/// `namebreak.cu`'s `runCudaBatch` has compiled-in kernel instantiations
/// for. `size49` is relied on elsewhere (`handlers::admin_create_target`'s
/// fallback when `alphabet_name` is omitted) - keep that name stable even if its
/// characters or position here ever change.
///
/// Every alphabet below predates protocol versioning itself, so they're all
/// tagged `(1, 0)` - the version this feature shipped in. A newly added
/// alphabet should be tagged with whatever the *next* MINOR version will be,
/// so `alphabet_available_to` keeps it hidden from clients that declared an
/// older one (see `ranges::claim_range`) - the entire reason this table
/// carries a version per row instead of just name+characters.
pub const PREDEFINED_ALPHABETS: &[(&str, &str, (u64, u64))] = &[
    ("size50", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_", (1, 0)),
    ("size49", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_", (1, 0)),
    ("size48", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_", (1, 0)),
    ("size47", " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_", (1, 0)),
    ("size43", " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_", (1, 0)),
    ("size42", " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_", (1, 0)),
];

pub fn lookup_predefined_alphabet(name: &str) -> Option<&'static str> {
    PREDEFINED_ALPHABETS.iter().find(|(n, _, _)| *n == name).map(|(_, chars, _)| *chars)
}

/// The `(major, minor)` protocol version a predefined alphabet was
/// introduced in, or `None` if `name` isn't a known alphabet at all.
pub fn alphabet_introduced_in(name: &str) -> Option<(u64, u64)> {
    PREDEFINED_ALPHABETS.iter().find(|(n, _, _)| *n == name).map(|&(_, _, since)| since)
}

/// Whether a client that declared `client_version` can be trusted to make
/// sense of a target using alphabet `name` - see
/// `namebreak_protocol::PROTOCOL_VERSION`'s doc comment on how MINOR
/// versions gate this. An unknown alphabet name is always available (there's
/// nothing to gate it *by* - no version has ever introduced it) - this
/// shouldn't come up in practice, since a real target's `alphabet_name` is
/// always one of `PREDEFINED_ALPHABETS` by construction (see
/// `handlers::admin_create_target`); it's mainly what lets this codebase's
/// own tests freely use synthetic alphabets outside that table without
/// tripping this gate.
pub fn alphabet_available_to(name: &str, client_version: namebreak_protocol::Version) -> bool {
    match alphabet_introduced_in(name) {
        Some((since_major, since_minor)) => (since_major, since_minor) <= (client_version.major, client_version.minor),
        None => true,
    }
}

pub fn alphabet_size(alphabet: &str) -> i64 {
    alphabet.chars().count() as i64
}

fn alphabet_chars(alphabet: &str) -> Vec<char> {
    alphabet.chars().collect()
}

/// A candidate's position within its own length's space, in
/// `index_to_candidate`'s order. An `i128` rather than an `i64`, since
/// `alphabet_size^len` passes `i64::MAX` beyond length 11 for every predefined
/// alphabet. The database has no 128-bit integers, so every stored position
/// is split into a `(block, index)` pair of `i64`s - see `split_pos`.
pub type Pos = i128;

/// The longest candidate the server carves, whatever the alphabet: the
/// client's own `MAX_CANDIDATE_LEN` (client/src/engine/limits.h), which
/// refuses to search anything longer. Keep the two in sync.
pub const MAX_CANDIDATE_LEN: i64 = 16;

/// The longest candidate length the server carves for `alphabet`:
/// `MAX_CANDIDATE_LEN`, unless the alphabet is so large that the whole space
/// at that length wouldn't fit a `Pos` (not the case for any predefined one).
pub fn max_supported_len(alphabet: &str) -> i64 {
    let size = alphabet_size(alphabet) as i128;
    let mut len = 0i64;
    let mut space: i128 = 1;
    while len < MAX_CANDIDATE_LEN {
        let Some(next) = space.checked_mul(size) else { break };
        space = next;
        len += 1;
    }
    len
}

/// How many trailing characters a stored `i64` index covers: the longest
/// length whose whole space still fits an `i64` (11 for every predefined
/// alphabet). A longer candidate's characters before those - its leading
/// prefix - are numbered separately, as a block - see `split_pos`. The same
/// split the client makes between its CPU-enumerated leading part and its
/// natively indexed trailing part (client/src/engine/search.cpp).
pub fn index_width(alphabet: &str) -> i64 {
    let size = alphabet_size(alphabet) as i128;
    let mut len = 0i64;
    let mut space: i128 = 1;
    while space.saturating_mul(size) <= i64::MAX as i128 {
        space *= size;
        len += 1;
    }
    len
}

/// How many candidates share one leading prefix (one block) at `len`. The
/// whole length's space for a length no longer than `index_width` - there's
/// only ever block 0 there.
pub fn block_size(alphabet: &str, len: i64) -> Pos {
    space_size(alphabet, len.min(index_width(alphabet)))
}

/// Splits `pos` into the `(block, index)` pair the database stores it as:
/// which leading prefix it falls under (numbered in `index_to_candidate`'s
/// order), and its index among that prefix's candidates. For a length no
/// longer than `index_width` the block is always 0 and the index is `pos`
/// itself, so rows from before 128-bit positions read back unchanged.
pub fn split_pos(alphabet: &str, len: i64, pos: Pos) -> (i64, i64) {
    let size = block_size(alphabet, len);
    ((pos / size) as i64, (pos % size) as i64)
}

/// Like `split_pos`, but for an exclusive end or a cursor: a `pos` right at
/// the start of a block is stored against the block before it, as an index
/// equal to `block_size`. Such a position is where something *stops* rather
/// than a candidate in its own right, so it belongs with the candidate
/// before it - and the end of a length no longer than `index_width` then
/// reads back as the plain index (the length's size) it always was.
pub fn split_end(alphabet: &str, len: i64, pos: Pos) -> (i64, i64) {
    if pos == 0 {
        return (0, 0);
    }
    let (block, index) = split_pos(alphabet, len, pos - 1);
    (block, index + 1)
}

/// Inverse of `split_pos` and `split_end`.
pub fn join_pos(alphabet: &str, len: i64, block: i64, index: i64) -> Pos {
    block as Pos * block_size(alphabet, len) + index as Pos
}

/// Total number of distinct candidates of the given length. Caller must
/// ensure `len <= max_supported_len(alphabet)`. Used by
/// `transition_alphabet_cursor` as a sentinel "past this length's entire
/// space, in any alphabet" value when nothing at all is left to translate a
/// cursor to; otherwise a natural, directly-tested primitive also heavily
/// used by the test suite to compute expected values independently.
pub fn space_size(alphabet: &str, len: i64) -> Pos {
    let size = alphabet_size(alphabet) as Pos;
    let mut space: Pos = 1;
    for _ in 0..len {
        space *= size;
    }
    space
}

/// Same enumeration order as the client's `indexToCandidate`: most-significant
/// character first, base-`alphabet_size(alphabet)` positional encoding.
pub fn index_to_candidate(alphabet: &str, mut index: Pos, len: i64) -> String {
    let chars = alphabet_chars(alphabet);
    let size = alphabet_size(alphabet) as Pos;
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
pub fn candidate_to_index(alphabet: &str, candidate: &str) -> Option<Pos> {
    let chars = alphabet_chars(alphabet);
    let size = alphabet_size(alphabet) as Pos;
    let mut index: Pos = 0;
    for ch in candidate.chars() {
        let digit = chars.iter().position(|&c| c == ch)? as Pos;
        index = index.checked_mul(size)?.checked_add(digit)?;
    }
    Some(index)
}

/// The alphabet's first and last characters - used to right-pad a bound
/// candidate that's shorter than the length currently being searched. The
/// lower bound pads with the minimum character (permissive on the low side:
/// "this, or anything continuing from it"); the upper bound pads with the
/// maximum character (permissive on the high side). Mirrors the client's
/// `getLowerBound`/`getUpperBound` convention.
fn min_max_chars(alphabet: &str) -> (char, char) {
    let mut chars = alphabet.chars();
    let min = chars.next().expect("alphabet must be non-empty");
    let max = chars.last().unwrap_or(min);
    (min, max)
}

/// The inclusive index range `[lower_at(len), upper_at(len)]` that a target's
/// `[lower_bound, upper_bound]` candidate strings imply at a specific candidate
/// length: whichever bound is shorter than `len` gets right-padded (lower with
/// the alphabet's minimum character, upper with its maximum); whichever is
/// longer gets truncated to `len` characters. This is the only primitive the
/// carving logic needs - it never has to represent "the bound at this length"
/// as a string, only as these two indices, so it's always safe from overflow
/// as long as `len <= max_supported_len(alphabet)` (the only case it's ever
/// called with).
pub fn bound_indices_at_len(alphabet: &str, lower_bound: &str, upper_bound: &str, len: i64) -> (Pos, Pos) {
    let (min_char, max_char) = min_max_chars(alphabet);
    let len = len as usize;

    let lower_str: String = pad_or_truncate(lower_bound, len, min_char);
    let upper_str: String = pad_or_truncate(upper_bound, len, max_char);

    // Both strings are built only from `alphabet`'s own characters (plus
    // whichever characters `lower_bound`/`upper_bound` already contain, which
    // callers must have validated against the alphabet at target-creation
    // time), so these can't fail in practice.
    let lower_idx = candidate_to_index(alphabet, &lower_str).expect("bound candidate contains a character outside the alphabet");
    let upper_idx = candidate_to_index(alphabet, &upper_str).expect("bound candidate contains a character outside the alphabet");
    (lower_idx, upper_idx)
}

fn pad_or_truncate(s: &str, len: usize, pad_with: char) -> String {
    let count = s.chars().count();
    if len <= count {
        s.chars().take(len).collect()
    } else {
        s.chars().chain(std::iter::repeat(pad_with).take(len - count)).collect()
    }
}

/// Whether a target's bounds actually describe a non-empty range: whether
/// `lower_bound` is not alphabetically after `upper_bound`, checked at
/// `lower_bound`'s own length. That's sufficient, not just convenient: in
/// plain lexicographic comparison, once two strings diverge at some position
/// they stay divergent (in the same direction) no matter what's appended
/// after, and when one is a prefix of the other - the only way this check's
/// length can fail to reach an actual divergence point - `bound_indices_at_len`
/// pads the shorter side with the alphabet's own min/max character, which by
/// definition can't flip the comparison at any length beyond that prefix
/// either. Capped at `max_supported_len(alphabet)` purely so a `lower_bound`
/// longer than that (legal - see `admin_create_target`) can't overflow
/// `bound_indices_at_len`'s index math. Rejects bounds given in the wrong
/// order (or that otherwise don't overlap) at target creation, rather than
/// silently carving nothing forever.
pub fn bounds_are_valid(alphabet: &str, lower_bound: &str, upper_bound: &str) -> bool {
    let len = (lower_bound.chars().count() as i64).min(max_supported_len(alphabet));
    let (lower_idx, upper_idx) = bound_indices_at_len(alphabet, lower_bound, upper_bound, len);
    lower_idx <= upper_idx
}

/// Whether `lower_bound` and `upper_bound` diverge at their very first
/// character. Required at target creation (see `admin_create_target`)
/// because carving (`ranges::claim_range`) always starts at
/// `candidate_len = 1`: if the two bounds shared even one leading character,
/// every carve at that length - and, transitively, at every length up to
/// however long the shared part is - would truncate both bounds down to the
/// same string, handing out a range with no searchable candidates in it.
/// Checking only length 1 is enough, not a heuristic: by the same
/// once-diverged-stays-diverged reasoning `bounds_are_valid` above already
/// relies on, a character that differs at any position is that number's
/// most significant digit from then on, so bounds that diverge at length 1
/// stay diverged at every longer length too. A bound pair that fails this
/// should have its shared leading text moved into the target's own
/// prefix/suffix instead - that's what prefix/suffix are for.
pub fn bounds_diverge_immediately(alphabet: &str, lower_bound: &str, upper_bound: &str) -> bool {
    let (lower_idx, upper_idx) = bound_indices_at_len(alphabet, lower_bound, upper_bound, 1);
    lower_idx != upper_idx
}

/// Strips a target's prefix/suffix off a full filename to recover the candidate
/// portion, e.g. for turning a `namebreak`-reported match back into an index via
/// `candidate_to_index`. Alphabet-independent - pure string surgery.
pub fn strip_prefix_suffix<'a>(filename: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    filename.strip_prefix(prefix)?.strip_suffix(suffix)
}

/// Compiles an operator-supplied "skip regex" into the form actually
/// evaluated: forced to anchor at the very start of the candidate regardless
/// of whether the pattern itself starts with `^`, since a skip decision is
/// only ever made from a candidate's *leading character* - see
/// `skip_char_mask`. Returns `None` if the pattern doesn't compile as a regex
/// at all (callers reject target creation/patches in that case).
pub fn compile_skip_regex(pattern: &str) -> Option<regex::Regex> {
    regex::Regex::new(&format!("^(?:{pattern})")).ok()
}

/// For each character of `alphabet` (in order), whether a lone candidate
/// consisting of just that character matches `skip_regex` - i.e. whether any
/// candidate starting with that character should be skipped. This is the only
/// shape of skip regex the carving logic supports: a pattern like `^AB`
/// matches the string "AB", but the mask is computed by testing "A" alone, so
/// it's evaluated as "skip everything starting with A", not "starting with
/// AB" - deliberately weaker than what the regex syntax itself can express,
/// in exchange for a skip decision being a single per-character lookup rather
/// than something that has to inspect individual candidates one at a time
/// (`find_skip_run`'s block search relies on this).
pub fn skip_char_mask(alphabet: &str, skip_regex: &regex::Regex) -> Vec<bool> {
    alphabet_chars(alphabet).iter().map(|c| skip_regex.is_match(&c.to_string())).collect()
}

/// Finds the first run of consecutive skip-marked alphabet characters at or
/// after `lo_index`, restricted to `[lo_index, hi_index_exclusive)`. Returns
/// that run's half-open index range `[skip_start, skip_end)`, or `None` if
/// there's no skip run in that span.
///
/// Every alphabet character skips (or doesn't) as a whole contiguous block of
/// `alphabet_size^(candidate_len - 1)` indices, since `index_to_candidate`
/// orders candidates by leading character first - so this only ever has to
/// walk the (at most `alphabet_size`, typically under 50) characters, never
/// individual candidates within a block.
pub fn find_skip_run(alphabet_size: i64, skip_chars: &[bool], candidate_len: i64, lo_index: Pos, hi_index_exclusive: Pos) -> Option<(Pos, Pos)> {
    if lo_index >= hi_index_exclusive {
        return None;
    }
    let block_size = pow_pos(alphabet_size, candidate_len - 1);
    let mut char_idx = (lo_index / block_size) as usize;
    while char_idx < skip_chars.len() {
        let block_start = char_idx as Pos * block_size;
        if block_start >= hi_index_exclusive {
            break;
        }
        if skip_chars[char_idx] {
            let skip_start = block_start.max(lo_index);
            let mut end_char_idx = char_idx;
            while end_char_idx + 1 < skip_chars.len() && skip_chars[end_char_idx + 1] {
                end_char_idx += 1;
            }
            let skip_end = ((end_char_idx as Pos + 1) * block_size).min(hi_index_exclusive);
            return Some((skip_start, skip_end));
        }
        char_idx += 1;
    }
    None
}

/// `base^exp` - callers only ever pass an `exp` (`candidate_len - 1`) small
/// enough that the result fits a `Pos` (see `max_supported_len`).
fn pow_pos(base: i64, exp: i64) -> Pos {
    let mut result: Pos = 1;
    for _ in 0..exp {
        result *= base as Pos;
    }
    result
}

/// Cap on how many concrete prefixes a single priority-range pattern (see
/// `expand_priority_pattern`) may expand into, so a careless wide character
/// class at several positions can't silently explode into an unmanageable
/// number of `priority_ranges` rows - each one is its own extra cursor
/// `ranges::claim_range` has to check on every claim.
pub const MAX_PRIORITY_PATTERN_EXPANSIONS: usize = 200;

/// Splits a priority-range pattern into its per-position atoms, each of
/// which matches exactly one character. Unlike `compile_skip_regex` (a
/// single regex tested only against a candidate's leading character), a
/// priority pattern is meant to pin down *several* leading positions at
/// once - e.g. `"[ _-]S"` means "space, underscore or hyphen, followed by
/// S", not one combined regex tested some other way. Splitting it into
/// atoms up front lets each position's matching characters be computed
/// independently (via the same single-character-testing technique
/// `skip_char_mask` already uses - see `expand_priority_pattern`) and then
/// combined into the cross-product of concrete literal prefixes.
///
/// Supported per position: a literal character, `.` (any character), a
/// backslash escape (`\d`, `\.`, ...), or a full `[...]` bracket expression
/// (ranges and negation both work, since the bracket's contents are handed
/// to `regex` unchanged). Quantifiers (`+ * ? {m,n}`) and alternation (`|`)
/// are deliberately rejected: they'd make a pattern's matched length
/// ambiguous, and the whole point here is a fixed, known prefix depth -
/// wanting several different lengths or alternative prefixes just means
/// creating several priority ranges (which can freely share a priority
/// value to be treated as one tier).
fn tokenize_pattern_atoms(pattern: &str) -> Result<Vec<String>, String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut atoms = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '[' => {
                let start = i;
                i += 1;
                if i < chars.len() && chars[i] == '^' {
                    i += 1;
                }
                if i < chars.len() && chars[i] == ']' {
                    i += 1; // a ']' right after '[' (or '[^') is a literal member, not the closing bracket
                }
                while i < chars.len() && chars[i] != ']' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                    }
                    i += 1;
                }
                if i >= chars.len() {
                    return Err("unterminated '[' in priority pattern".into());
                }
                i += 1; // consume the closing ']'
                atoms.push(chars[start..i].iter().collect());
            }
            '\\' => {
                if i + 1 >= chars.len() {
                    return Err("priority pattern ends with a trailing '\\'".into());
                }
                atoms.push(chars[i..i + 2].iter().collect());
                i += 2;
            }
            '.' => {
                atoms.push(".".to_string());
                i += 1;
            }
            c @ ('+' | '*' | '?' | '|' | '(' | ')' | '{' | '}' | '^' | '$') => {
                return Err(format!(
                    "unsupported character '{c}' in priority pattern - only literal characters, '.', backslash escapes, \
                     and '[...]' character classes are allowed (no quantifiers, alternation, groups or anchors)"
                ));
            }
            c => {
                atoms.push(c.to_string());
                i += 1;
            }
        }
    }
    if atoms.is_empty() {
        return Err("priority pattern must not be empty".into());
    }
    Ok(atoms)
}

/// Expands a priority-range pattern into the concrete literal prefixes it
/// matches against `alphabet` - see `tokenize_pattern_atoms` for the
/// supported syntax. Each atom is compiled as its own single-character
/// regex (anchored at *both* ends, unlike `compile_skip_regex` - a priority
/// atom must match exactly one character, not "starts with") and tested
/// against every character of `alphabet`; the per-position matching-character
/// lists are then combined into their cross-product. Every returned prefix
/// has the same length (the atom count) - this is what makes a fixed,
/// unambiguous prefix depth possible.
///
/// Returns an error if any atom fails to compile, if any position matches no
/// character in `alphabet` at all (a dead pattern), or if the cross-product
/// would exceed `MAX_PRIORITY_PATTERN_EXPANSIONS`.
pub fn expand_priority_pattern(alphabet: &str, pattern: &str) -> Result<Vec<String>, String> {
    let atoms = tokenize_pattern_atoms(pattern)?;
    let chars = alphabet_chars(alphabet);

    let mut per_position: Vec<Vec<char>> = Vec::with_capacity(atoms.len());
    for atom in &atoms {
        let re = regex::Regex::new(&format!("^(?:{atom})$")).map_err(|_| format!("'{atom}' is not a valid pattern"))?;
        let matching: Vec<char> = chars.iter().copied().filter(|c| re.is_match(&c.to_string())).collect();
        if matching.is_empty() {
            return Err(format!("'{atom}' doesn't match any character in the target's alphabet"));
        }
        per_position.push(matching);
    }

    let total: usize = per_position.iter().map(|m| m.len()).product();
    if total > MAX_PRIORITY_PATTERN_EXPANSIONS {
        return Err(format!(
            "priority pattern expands to {total} concrete prefixes, which is more than the limit of {MAX_PRIORITY_PATTERN_EXPANSIONS} - use a narrower pattern"
        ));
    }

    let mut prefixes = vec![String::new()];
    for matching in &per_position {
        let mut next = Vec::with_capacity(prefixes.len() * matching.len());
        for prefix in &prefixes {
            for &c in matching {
                let mut p = prefix.clone();
                p.push(c);
                next.push(p);
            }
        }
        prefixes = next;
    }
    Ok(prefixes)
}

/// The outcome of moving a target's carving cursor from `old_alphabet` to
/// `new_alphabet`, needed once `handlers::admin_patch_target` changes a
/// target's alphabet: any range already carved keeps its own stored alphabet
/// (see `models::Range::alphabet`), but the cursor (`target_progress`) has to
/// be translated the first time carving reaches it after the patch - lazily,
/// the same way a skip-regex run is only ever materialized once carving
/// actually reaches it (see `find_skip_run`), rather than retroactively.
///
/// Finds the smallest candidate, expressible entirely in `new_alphabet`,
/// whose position in `old_alphabet`'s own ordering is still `>= old_next_index`,
/// i.e. the earliest candidate the old alphabet's sweep hasn't already
/// completed that the new alphabet can actually produce. This is standard
/// "round up to the nearest value expressible with a restricted character
/// set" arithmetic: scanning positions from the *last* toward the *first*,
/// it looks for the rightmost position that can be bumped up to some larger
/// `new_alphabet` character while every position before it stays exactly as
/// the old candidate already has it (padding everything after with
/// `new_alphabet`'s own smallest character then gives the smallest valid
/// completion), falling back to an earlier position only when a later one
/// has nothing bigger available in `new_alphabet` to bump up to.
///
/// Picking the *rightmost* such position, rather than simply the first
/// character that doesn't exist in `new_alphabet` and resetting everything
/// after it to `new_alphabet`'s own minimum character, matters: consider
/// dropping a character that sorts *early* in the alphabet - e.g. going from
/// `size49` to `size42` drops `!`, which sorts right after the space
/// character. A cursor candidate starting with `!` would, if resumed with
/// `new_alphabet`'s own minimum character (space) at every position, land
/// *before* the cursor - re-offering content the old alphabet's sweep had
/// already completed (everything starting with space sorts before anything
/// starting with `!`). Finding the smallest character in `new_alphabet`
/// that's still *greater* than the offending one - `(` in this example, not
/// space - keeps the result safely ahead of the cursor instead.
pub struct AlphabetTransition {
    /// `[start_index, end_index)` under `old_alphabet`, at the same
    /// `candidate_len` as the old cursor, to persist as a `skipped` range -
    /// exactly the candidates that are provably neither already completed
    /// (they're `>= old_next_index`) nor ever expressible in `new_alphabet`
    /// (nothing smaller than `new_next_index` qualifies, by construction).
    /// `None` when the old cursor's candidate is already fully expressible
    /// in the new alphabet, so nothing needs skipping.
    pub skip: Option<(Pos, Pos)>,
    /// Where carving should resume, under `new_alphabet`, at the same
    /// `candidate_len` as the old cursor - or an index already known to be
    /// past this length's own space entirely (see `alphabet::space_size`),
    /// if nothing `new_alphabet` can express is left at this length at all.
    /// The caller is still responsible for bumping past this length in that
    /// case - exactly as it already does for ordinary carving, via
    /// `bump_length_if_exhausted`.
    pub new_next_index: Pos,
}

/// Computes an `AlphabetTransition` for a cursor currently at
/// `(candidate_len, old_next_index)` under `old_alphabet`. Caller must ensure
/// `old_next_index` is still within this length's remaining space under
/// `old_alphabet` (i.e. hasn't already been fully carved) - `claim_range`
/// already checks this before doing anything else on a fresh loop iteration.
pub fn transition_alphabet_cursor(
    old_alphabet: &str,
    new_alphabet: &str,
    lower_bound: &str,
    upper_bound: &str,
    candidate_len: i64,
    old_next_index: Pos,
) -> AlphabetTransition {
    let old_candidate: Vec<char> = index_to_candidate(old_alphabet, old_next_index, candidate_len).chars().collect();
    let len = old_candidate.len();
    let new_chars = alphabet_chars(new_alphabet);
    let (new_min_char, _) = min_max_chars(new_alphabet);
    // Every character involved here - old_candidate's own, and every
    // character of new_alphabet - is itself a member of old_alphabet (the
    // latter because a target's alphabet is only ever patched to another
    // *predefined* alphabet, and every predefined alphabet's characters
    // appear in the same relative order within any other that's a superset
    // of it - see PREDEFINED_ALPHABETS). That's what makes comparing them
    // all via old_alphabet's own ordinal rank meaningful.
    let old_rank = |c: char| candidate_to_index(old_alphabet, &c.to_string()).expect("character must be a member of old_alphabet");
    let is_new = |c: char| new_chars.contains(&c);

    // Fully expressible as-is - just reinterpret the same string under the new alphabet, nothing to skip.
    if old_candidate.iter().copied().all(is_new) {
        let s: String = old_candidate.iter().collect();
        let new_next_index = candidate_to_index(new_alphabet, &s).expect("just confirmed every character is in new_alphabet");
        return AlphabetTransition { skip: None, new_next_index };
    }

    // prefix_representable[k] = whether old_candidate[0..k] are all
    // expressible in new_alphabet - i.e. whether pivoting at position k
    // (see below) can validly preserve that much of the original candidate.
    let mut prefix_representable = vec![true; len + 1];
    for i in 0..len {
        prefix_representable[i + 1] = prefix_representable[i] && is_new(old_candidate[i]);
    }

    let mut found: Option<(usize, char)> = None;
    for k in (0..len).rev() {
        if !prefix_representable[k] {
            continue;
        }
        let threshold = old_rank(old_candidate[k]);
        let bump_to = new_chars.iter().copied().filter(|&c| old_rank(c) > threshold).min_by_key(|&c| old_rank(c));
        if let Some(c) = bump_to {
            found = Some((k, c));
            break;
        }
    }

    let (_, upper_at_len) = bound_indices_at_len(old_alphabet, lower_bound, upper_bound, candidate_len);

    let Some((k, bumped_char)) = found else {
        // Nothing new_alphabet can express is left anywhere in this length,
        // from old_next_index onward - skip the whole remainder, and report
        // an index already past this length's space in any alphabet, so the
        // caller's bump_length_if_exhausted moves on exactly as it would for
        // ordinary exhaustion.
        return AlphabetTransition { skip: Some((old_next_index, upper_at_len + 1)), new_next_index: space_size(new_alphabet, candidate_len) };
    };

    let mut new_candidate: Vec<char> = old_candidate[..k].to_vec();
    new_candidate.push(bumped_char);
    new_candidate.extend(std::iter::repeat(new_min_char).take(len - k - 1));
    let new_candidate: String = new_candidate.into_iter().collect();

    let new_next_index = candidate_to_index(new_alphabet, &new_candidate).expect("built only from new_alphabet's own characters");
    // Exactly the candidates that are provably neither already completed
    // (old_next_index is where the old sweep left off) nor ever expressible
    // in new_alphabet (new_candidate is the smallest one that is, by
    // construction of the rightmost-pivot search above) - so this is the
    // tightest correct skip, not merely "the rest of the shared prefix's
    // whole block" (which would also be correct, just far wider than
    // necessary, since most of that block's own candidates end up covered
    // separately by new_alphabet's own future walk anyway).
    let skip_end = candidate_to_index(old_alphabet, &new_candidate)
        .expect("new_candidate's characters are all in new_alphabet, itself a subset of old_alphabet")
        .min(upper_at_len + 1);

    AlphabetTransition { skip: Some((old_next_index, skip_end)), new_next_index }
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
    start_index: Pos,
    end_index: Pos,
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
        for &(name, chars, _since) in PREDEFINED_ALPHABETS {
            let size = alphabet_size(chars);
            assert!(size > 0, "{name} has an empty alphabet");
            let unique: std::collections::HashSet<char> = chars.chars().collect();
            assert_eq!(unique.len() as i64, size, "{name} has duplicate characters, which would break the index<->candidate mapping");
        }
    }

    #[test]
    fn alphabet_introduced_in_looks_up_a_known_alphabets_version_but_not_an_unknown_ones() {
        assert_eq!(alphabet_introduced_in("size49"), Some((1, 0)));
        assert_eq!(alphabet_introduced_in("no-such-alphabet"), None);
    }

    #[test]
    fn alphabet_available_to_gates_on_minor_version_but_not_patch() {
        use namebreak_protocol::Version;
        // Every real predefined alphabet is tagged (1, 0) - available to
        // anything from 1.0.0 onward, patch version doesn't matter.
        assert!(alphabet_available_to("size49", Version::new(1, 0, 0)));
        assert!(alphabet_available_to("size49", Version::new(1, 0, 99)));
        assert!(alphabet_available_to("size49", Version::new(1, 5, 0)));
        assert!(alphabet_available_to("size49", Version::new(2, 0, 0)), "a newer major version can still use an old alphabet");
        assert!(!alphabet_available_to("size49", Version::new(0, 9, 0)), "a client older than this alphabet's own introduced-in version is refused");
    }

    #[test]
    fn alphabet_available_to_fails_open_for_an_unrecognized_name() {
        use namebreak_protocol::Version;
        // Not a real code path (a target's alphabet_name is always validated
        // against PREDEFINED_ALPHABETS elsewhere) - but this codebase's own
        // tests rely on it to freely use synthetic alphabets without tripping
        // this gate, so it's worth pinning down explicitly.
        assert!(alphabet_available_to("totally-made-up", Version::new(0, 0, 1)));
    }

    #[test]
    fn alphabet_size_matches_cuda_constant() {
        assert_eq!(alphabet_size(DEFAULT), 49);
    }

    #[test]
    fn index_width_space_fits_i64_but_next_length_does_not() {
        let len = index_width(DEFAULT);
        assert_eq!(len, 11);
        assert!(space_size(DEFAULT, len) <= i64::MAX as Pos);
        assert!(space_size(DEFAULT, len + 1) > i64::MAX as Pos);
    }

    #[test]
    fn index_width_grows_as_alphabet_shrinks() {
        assert!(index_width(TINY) > index_width(DEFAULT));
    }

    #[test]
    fn max_supported_len_is_the_clients_limit_for_every_predefined_alphabet() {
        for &(name, chars, _since) in PREDEFINED_ALPHABETS {
            assert_eq!(max_supported_len(chars), MAX_CANDIDATE_LEN, "{name}");
        }
    }

    #[test]
    fn split_pos_leaves_short_lengths_in_block_0_and_round_trips_long_ones() {
        assert_eq!(split_pos(DEFAULT, 11, space_size(DEFAULT, 11) - 1), (0, (space_size(DEFAULT, 11) - 1) as i64));
        let block = space_size(DEFAULT, 11);
        // "!" followed by fourteen spaces: block 1 of length 15 (leading "!   "), index 0.
        let pos = candidate_to_index(DEFAULT, &format!("!{}", " ".repeat(14))).unwrap();
        assert_eq!(split_pos(DEFAULT, 15, pos), (49 * 49 * 49, 0));
        for &p in &[0, 1, block - 1, block, block + 7, space_size(DEFAULT, 16) - 1] {
            let (b, i) = split_pos(DEFAULT, 16, p);
            assert_eq!(join_pos(DEFAULT, 16, b, i), p);
        }
        // An exclusive end at the very end of a block may be stored against that block.
        assert_eq!(join_pos(DEFAULT, 16, 0, block as i64), block);
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
    fn bound_indices_at_len_pads_the_shorter_bound_and_truncates_the_longer_one() {
        // The motivating example: "BLACKSMITH" (10 chars) to "CATAPULT" (8
        // chars) - upper bound shorter than lower bound.
        let lower = "BLACKSMITH";
        let upper = "CATAPULT";

        // At length 10 (lower's own length): lower is used as-is; upper (2
        // chars short) is padded with 2 max characters.
        let (lo, hi) = bound_indices_at_len(DEFAULT, lower, upper, 10);
        assert_eq!(lo, candidate_to_index(DEFAULT, lower).unwrap());
        assert_eq!(hi, candidate_to_index(DEFAULT, &format!("{upper}__")).unwrap());
        assert!(lo <= hi, "BLACKSMITH must sort before CATAPULT + padding");

        // At length 11: lower gets 1 min-char, upper gets 3 max-chars.
        let (lo11, hi11) = bound_indices_at_len(DEFAULT, lower, upper, 11);
        assert_eq!(lo11, candidate_to_index(DEFAULT, &format!("{lower} ")).unwrap());
        assert_eq!(hi11, candidate_to_index(DEFAULT, &format!("{upper}___")).unwrap());

        // Growing into an already-padded length is exactly a base shift.
        assert_eq!(lo11, lo * alphabet_size(DEFAULT) as Pos);
    }

    #[test]
    fn bound_indices_at_len_handles_a_longer_upper_bound_too() {
        // Symmetric case: upper bound longer than lower bound.
        let lower = "CAT";
        let upper = "CATAPULTMK2";
        let (lo, hi) = bound_indices_at_len(DEFAULT, lower, upper, 3);
        assert_eq!(lo, candidate_to_index(DEFAULT, lower).unwrap());
        assert_eq!(hi, candidate_to_index(DEFAULT, "CAT").unwrap()); // upper truncated to 3 chars
        assert_eq!(lo, hi, "only 'CAT' itself is in range at this length");
    }

    #[test]
    fn bounds_are_valid_rejects_reversed_bounds() {
        assert!(bounds_are_valid(DEFAULT, "BLACKSMITH", "CATAPULT"));
        assert!(!bounds_are_valid(DEFAULT, "CATAPULT", "BLACKSMITH"));
    }

    #[test]
    fn bounds_are_valid_handles_a_lower_bound_longer_than_max_supported_len_without_panicking() {
        // A bound can now be an arbitrary known filename used purely for its
        // alphabetical position (see admin_create_target), so it's no longer
        // guaranteed to fit within max_supported_len(alphabet) - bounds_are_valid
        // must clamp its own comparison length rather than handing
        // bound_indices_at_len a length it can't safely index.
        let long_lower = "A".repeat((max_supported_len(DEFAULT) + 5) as usize);
        assert!(bounds_are_valid(DEFAULT, &long_lower, "ZZZZZ"));
        assert!(!bounds_are_valid(DEFAULT, &long_lower, " "));
    }

    #[test]
    fn bounds_diverge_immediately_accepts_bounds_that_differ_at_the_first_character() {
        // The real-world shape (see admin_create_target's own example):
        // bounds that already differ at their first character, regardless
        // of whatever they share (or don't) after that.
        assert!(bounds_diverge_immediately(DEFAULT, "FINZ09BX", "GAMEMENU"));
        // Also holds for two entirely unrelated filenames used purely for
        // their alphabetical position (see admin_create_target's doc
        // comment on that use case) - nothing about this check requires the
        // bounds to relate to each other beyond differing at position 0.
        assert!(bounds_diverge_immediately(DEFAULT, "GLUE PALCS DLG.GRP", "MUSIC MENGSKVICTORY.WAV"));
    }

    #[test]
    fn bounds_diverge_immediately_rejects_a_shared_leading_character() {
        // This is exactly the shape that used to make claim_range carve an
        // unsearchable single-candidate range at short lengths (both bounds
        // truncate to the same "T", "TE", "TES", ... until length finally
        // exceeds the shared "TEST" run) - see ranges::claim_range's carving
        // and namebreak.cu's former strict lowerBound < upperBound check.
        assert!(!bounds_diverge_immediately(DEFAULT, "TESTAAA", "TESTZZZ"));
        // Even sharing just the first character is enough to reject -
        // nothing about *how much* they share matters, only whether they
        // share anything at position 0.
        assert!(!bounds_diverge_immediately(DEFAULT, "AAA", "AZZ"));
    }

    #[test]
    fn bounds_diverge_immediately_rejects_fully_equal_bounds() {
        // The degenerate case of sharing a leading character: sharing all of
        // it. Still correctly rejected by the same length-1 check.
        assert!(!bounds_diverge_immediately(DEFAULT, "SAME", "SAME"));
    }

    #[test]
    fn bounds_diverge_immediately_pads_a_bound_shorter_than_one_character() {
        // An empty bound pads to the alphabet's min (lower_bound) or max
        // (upper_bound) character at length 1 - same padding
        // bound_indices_at_len already applies elsewhere, not a special case
        // here. Both empty is the widest possible bound pair ("no
        // constraint on either side"), min character vs max character -
        // about as diverged as two bounds can get, not equal.
        assert!(bounds_diverge_immediately(DEFAULT, "", "A"));
        assert!(bounds_diverge_immediately(DEFAULT, "", ""));
        // Only degenerate if the min and max character happen to coincide -
        // i.e. an alphabet with just one character, which nothing in this
        // codebase actually allows (MAX_ALPHABET_SIZE's whole *point* is a
        // useful brute-force space), but bounds_diverge_immediately itself
        // has no lower limit on alphabet size to enforce that - worth
        // documenting the edge rather than leaving it implicit.
        assert!(!bounds_diverge_immediately("A", "", ""));
    }

    #[test]
    fn strip_prefix_suffix_recovers_the_candidate() {
        assert_eq!(strip_prefix_suffix("REZ\\AB.WAV", "REZ\\", ".WAV"), Some("AB"));
        assert_eq!(strip_prefix_suffix("WRONG\\AB.WAV", "REZ\\", ".WAV"), None);
    }

    #[test]
    fn compile_skip_regex_rejects_invalid_patterns_but_accepts_valid_ones() {
        assert!(compile_skip_regex("[M-Q]").is_some());
        assert!(compile_skip_regex("[").is_none(), "unclosed character class must not compile");
    }

    #[test]
    fn skip_char_mask_marks_exactly_the_characters_the_regex_matches() {
        let regex = compile_skip_regex("[M-Q]").unwrap();
        let mask = skip_char_mask(DEFAULT, &regex);
        let chars = alphabet_chars(DEFAULT);
        for (c, &skip) in chars.iter().zip(mask.iter()) {
            assert_eq!(skip, ('M'..='Q').contains(c), "mismatch for {c:?}");
        }
    }

    #[test]
    fn find_skip_run_locates_the_first_run_at_or_after_lo_index() {
        // 26-letter alphabet, length-1 candidates so block_size == 1 and
        // "leading character" is just the candidate itself.
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let regex = compile_skip_regex("[M-Q]").unwrap();
        let mask = skip_char_mask(letters, &regex);

        // Searching the whole space finds M(12)..Q(16) inclusive, i.e. [12, 17).
        assert_eq!(find_skip_run(26, &mask, 1, 0, 26), Some((12, 17)));
        // Starting already inside the run clips skip_start to lo_index.
        assert_eq!(find_skip_run(26, &mask, 1, 14, 26), Some((14, 17)));
        // Starting after the run finds nothing.
        assert_eq!(find_skip_run(26, &mask, 1, 17, 26), None);
        // A hi bound that cuts the run off is respected.
        assert_eq!(find_skip_run(26, &mask, 1, 0, 15), Some((12, 15)));
    }

    #[test]
    fn find_skip_run_operates_on_whole_leading_character_blocks_at_longer_lengths() {
        // At length 2 over a 26-letter alphabet, each leading character owns a
        // block of 26 indices - skipping "M" must skip all 26 of "MA".."MZ",
        // not just the single index that would apply at length 1.
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let regex = compile_skip_regex("^M").unwrap();
        let mask = skip_char_mask(letters, &regex);
        let hi = 26 * 26;
        assert_eq!(find_skip_run(26, &mask, 2, 0, hi), Some((12 * 26, 13 * 26)));
    }

    #[test]
    fn transition_alphabet_cursor_skips_the_shared_prefix_block_when_the_cursor_diverges() {
        // The feature spec's motivating example: digits+letters -> letters-only.
        // The skip runs exactly up to "ABCAAA" (the smallest letters-only
        // candidate sharing the "ABC" prefix), not all the way to "ABCZZZ"
        // (the end of the whole "ABC"-prefixed block under the old
        // alphabet) - everything in between is either also unrepresentable
        // (still correctly excluded by the new alphabet's own walk never
        // producing it) or would itself have been an even smaller valid
        // resume point, contradicting "ABCAAA" being the smallest one.
        let old = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let old_index = candidate_to_index(old, "ABC001").unwrap();
        let full_bounds_lower = "AAA000";
        let full_bounds_upper = "ZZZZZZ";

        let transition = transition_alphabet_cursor(old, new, full_bounds_lower, full_bounds_upper, 6, old_index);

        let (skip_start, skip_end) = transition.skip.expect("digits aren't in the new alphabet - a skip is required");
        assert_eq!(skip_start, old_index);
        assert_eq!(skip_end, candidate_to_index(old, "ABCAAA").unwrap());
        assert_eq!(transition.new_next_index, candidate_to_index(new, "ABCAAA").unwrap());
    }

    #[test]
    fn transition_alphabet_cursor_clips_the_skip_to_the_targets_own_upper_bound() {
        let old = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let old_index = candidate_to_index(old, "ABC001").unwrap();
        // Upper bound cuts off partway through the "ABC" block instead of at its natural end.
        let transition = transition_alphabet_cursor(old, new, "AAA000", "ABC500", 6, old_index);

        let (_, skip_end) = transition.skip.unwrap();
        assert_eq!(skip_end, candidate_to_index(old, "ABC500").unwrap() + 1, "must not skip past the target's own upper bound");
    }

    #[test]
    fn transition_alphabet_cursor_needs_no_skip_when_the_cursor_is_already_expressible() {
        // Shrinking size49 -> size42 (dropping punctuation the cursor doesn't use).
        let old = DEFAULT;
        let new = SIZE42;
        let old_index = candidate_to_index(old, "ABC").unwrap();

        let transition = transition_alphabet_cursor(old, new, "AAA", "ZZZ", 3, old_index);

        assert!(transition.skip.is_none(), "\"ABC\" is expressible in size42 too - nothing to skip");
        assert_eq!(transition.new_next_index, candidate_to_index(new, "ABC").unwrap());
    }

    #[test]
    fn transition_alphabet_cursor_handles_divergence_at_the_very_first_character() {
        let old = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let old_index = candidate_to_index(old, "5XY").unwrap();

        let transition = transition_alphabet_cursor(old, new, "000", "ZZZ", 3, old_index);

        let (skip_start, skip_end) = transition.skip.unwrap();
        assert_eq!(skip_start, old_index);
        assert_eq!(skip_end, candidate_to_index(old, "AAA").unwrap(), "skip runs exactly up to the smallest representable candidate, \"AAA\"");
        assert_eq!(transition.new_next_index, candidate_to_index(new, "AAA").unwrap());
    }

    /// Regression test for the real bug this rewrite fixes: size49 -> size42
    /// drops '!', which sorts right after the space character - resuming at
    /// new_alphabet's own minimum character (space) would land *before* the
    /// old cursor, re-offering content the old alphabet's sweep had already
    /// completed (everything starting with space sorts before anything
    /// starting with '!'). The fix must instead resume at the smallest
    /// new_alphabet character that's still bigger than '!' itself.
    #[test]
    fn transition_alphabet_cursor_does_not_jump_backward_when_a_dropped_character_sorts_early() {
        let old = " !ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // '!' dropped, everything else unchanged
        let old_index = candidate_to_index(old, "!A").unwrap();

        let transition = transition_alphabet_cursor(old, new, "  ", "ZZ", 2, old_index);

        // The decisive check: comparing both under old_alphabet's own
        // ordering (valid since new_alphabet's characters are a subset of
        // old_alphabet's) the resume point must land strictly *after* the
        // old cursor - never at or before it, which is exactly the bug this
        // rewrite fixes.
        let resumed_candidate = index_to_candidate(new, transition.new_next_index, 2);
        let resumed_old_index = candidate_to_index(old, &resumed_candidate).unwrap();
        assert!(resumed_old_index > old_index, "must not resume at a position the old alphabet's sweep already passed");

        assert_eq!(resumed_candidate, "A ", "'A' is the smallest new_alphabet character bigger than the dropped '!'");
        let (skip_start, skip_end) = transition.skip.expect("'!' isn't in the new alphabet - a skip is required");
        assert_eq!(skip_start, old_index);
        assert_eq!(skip_end, resumed_old_index, "the skip must run exactly up to (not past) the resume point");
    }

    /// When nothing in `new_alphabet` is even bigger than the old cursor's
    /// very first character, there's no candidate at this length, in any
    /// position, that's both `>= old_next_index` and expressible in the new
    /// alphabet - the whole remainder of the length must be skipped, and the
    /// caller told (via the same sentinel ordinary exhaustion produces) to
    /// bump to the next length instead of resuming here at all.
    #[test]
    fn transition_alphabet_cursor_reports_nothing_left_when_no_new_alphabet_character_is_big_enough() {
        let old = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let new = "0123456789"; // digits only - smaller than any letter
        let old_index = candidate_to_index(old, "Z").unwrap(); // the very last candidate at length 1

        let transition = transition_alphabet_cursor(old, new, "0", "Z", 1, old_index);

        let (skip_start, skip_end) = transition.skip.expect("nothing after 'Z' can be expressed in a digits-only alphabet");
        assert_eq!(skip_start, old_index);
        assert_eq!(skip_end, candidate_to_index(old, "Z").unwrap() + 1, "skips to this length's own upper bound");
        assert_eq!(transition.new_next_index, space_size(new, 1), "signals bump_length_if_exhausted the same way ordinary exhaustion does");
    }

    #[test]
    fn expand_priority_pattern_cross_products_a_bracket_class_with_a_literal() {
        // The feature's motivating example: "[ _-]S" over the default alphabet.
        let mut prefixes = expand_priority_pattern(DEFAULT, "[ _-]S").unwrap();
        prefixes.sort();
        let mut expected = vec![" S".to_string(), "_S".to_string(), "-S".to_string()];
        expected.sort();
        assert_eq!(prefixes, expected);
    }

    #[test]
    fn expand_priority_pattern_handles_a_single_literal_atom() {
        assert_eq!(expand_priority_pattern(DEFAULT, "S").unwrap(), vec!["S".to_string()]);
    }

    #[test]
    fn expand_priority_pattern_supports_dot_and_backslash_escapes() {
        // "." matches every character in the alphabet; "\d" only the digits.
        let dot = expand_priority_pattern(SIZE42, ".").unwrap();
        assert_eq!(dot.len(), alphabet_size(SIZE42) as usize);

        let digits = expand_priority_pattern(SIZE42, "\\d").unwrap();
        let mut digits_sorted = digits.clone();
        digits_sorted.sort();
        assert_eq!(digits_sorted, vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
    }

    #[test]
    fn expand_priority_pattern_rejects_a_dead_position() {
        // SIZE42 has no lowercase letters at all.
        let err = expand_priority_pattern(SIZE42, "z").unwrap_err();
        assert!(err.contains('z'), "error should name the offending atom: {err}");
    }

    #[test]
    fn expand_priority_pattern_rejects_quantifiers_and_alternation() {
        assert!(expand_priority_pattern(DEFAULT, "A+").is_err());
        assert!(expand_priority_pattern(DEFAULT, "A*").is_err());
        assert!(expand_priority_pattern(DEFAULT, "A|B").is_err());
        assert!(expand_priority_pattern(DEFAULT, "(AB)").is_err());
        assert!(expand_priority_pattern(DEFAULT, "A{2,3}").is_err());
    }

    #[test]
    fn expand_priority_pattern_rejects_an_unterminated_bracket() {
        assert!(expand_priority_pattern(DEFAULT, "[AB").is_err());
    }

    #[test]
    fn expand_priority_pattern_rejects_an_empty_pattern() {
        assert!(expand_priority_pattern(DEFAULT, "").is_err());
    }

    #[test]
    fn expand_priority_pattern_rejects_an_oversized_expansion() {
        // Every position matches broadly, so the cross-product blows well past the cap.
        let err = expand_priority_pattern(DEFAULT, "...").unwrap_err();
        assert!(err.contains("200"), "error should mention the limit: {err}");
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
