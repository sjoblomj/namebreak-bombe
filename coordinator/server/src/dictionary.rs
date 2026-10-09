//! Dictionary targets: candidates made of words from word lists, with
//! separators between them, rather than every string of an alphabet - and
//! their numbers. This is the server's copy of the client's own
//! (client/src/engine/dictionary.h and wordlist.h), and has to agree with it
//! exactly: a dictionary target's ranges are ranges of these numbers, which
//! the client searches the candidates of as it numbers them. A claim carries
//! checksums of the word lists, which the client checks its own against.
//!
//! A candidate is one to `max_words` words, with a separator between each
//! two, and then a tail: word, separator, word, ..., tail - every word from
//! the same list (the target's word lists, merged: sorted, without
//! duplicates), every separator from the same list, every tail from the
//! same list (only "", without tails). Its filename is prefix + candidate +
//! suffix. Every candidate has a number, counted the way an odometer counts
//! with a wheel per word, per separator and for the tail: all one-word
//! candidates first (if min_words is 1), then all two-word ones, and so on.
//! Within a word count the wheels from the left are word 1, separator 1,
//! word 2, ..., word k, and the tail, which turns fastest:
//!
//!   number - start of the k-word block = (((w1 * S + s1) * W + w2) * S + s2 ... * W + wk) * T + t
//!
//! with W words, S separators and T tails (1, without tails).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::AppError;
use crate::models::Target;

/// The most words a candidate can have - the client's `kMaxDictionaryWords`
/// (client/src/engine/dictionary.h). Keep the two in sync.
pub const MAX_WORDS: i64 = 8;

/// The longest a tail can be, and the most tails a target can have - the
/// client's `kMaxDictionaryTailLength` and `kMaxDictionaryTails`. Keep them
/// in sync.
pub const MAX_TAIL_LENGTH: usize = 8;
pub const MAX_TAILS: usize = 1 << 20;

/// The dictionary compiled into the client (client/data/english-1.txt), and
/// the checksum of its words - pinned by the client's wordlist_test too. The
/// server has it built in as well (`server/data/english-1.txt`, a copy of the
/// client's), stored as a word list when it starts (`store_built_in`), so a
/// target can use it without uploading it. A client uses its own copy
/// rather than download it. The other `english-` names are kept for the
/// dictionaries the two may have built in one day.
pub const ENGLISH_1: &str = "english-1";
pub const ENGLISH_1_CHECKSUM: u64 = 0x63B3_5282_3C60_59B0;
const ENGLISH_1_TEXT: &[u8] = include_bytes!("../data/english-1.txt");

/// Whether `name` is one of the names kept for the dictionaries built in -
/// which can't be uploaded or deleted.
pub fn is_built_in_name(name: &str) -> bool {
    name.starts_with("english-")
}

/// Stores the word lists built into the server - `english-1` - unless
/// they're there already. Run whenever the server connects to its database
/// (see `db::connect`).
pub async fn store_built_in(pool: &sqlx::SqlitePool) -> anyhow::Result<()> {
    let words = sorted_unique(parse_word_list(ENGLISH_1_TEXT).0);
    let words_checksum = checksum(&words);
    anyhow::ensure!(
        words_checksum == ENGLISH_1_CHECKSUM,
        "the built-in {ENGLISH_1} has checksum {}, not {} - data/english-1.txt isn't the client's",
        hex64(words_checksum),
        hex64(ENGLISH_1_CHECKSUM)
    );
    sqlx::query("INSERT OR IGNORE INTO word_lists (name, content, word_count, checksum, created_at) VALUES (?, ?, ?, ?, ?)")
        .bind(ENGLISH_1)
        .bind(ENGLISH_1_TEXT)
        .bind(words.len() as i64)
        .bind(hex64(words_checksum))
        .bind(crate::state::now_unix())
        .execute(pool)
        .await?;
    Ok(())
}

/// `text` as Storm hashes it: ASCII lowercase letters made uppercase and
/// '/' made '\' - the client's `normalizeMpqName`. Words, separators, prefix,
/// suffix and bounds all go through this.
pub fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'a'..='z' => c.to_ascii_uppercase(),
            '/' => '\\',
            _ => c,
        })
        .collect()
}

/// Whether `text` is all printable ASCII (' ' to '~'), as every word is.
pub fn is_printable_ascii(text: &str) -> bool {
    text.bytes().all(|b| (0x20..=0x7E).contains(&b))
}

/// A word list read as the client reads one (`parseWordList`): one word per
/// line, LF or CRLF. Spaces and tabs around a word are dropped, and blank
/// lines and lines starting with '#' are skipped, as is - with its line
/// number in the second list - a line with anything but printable ASCII.
/// Each word is normalized, in file order, duplicates and all.
pub fn parse_word_list(content: &[u8]) -> (Vec<String>, Vec<usize>) {
    let mut words = Vec::new();
    let mut skipped = Vec::new();
    for (i, line) in content.split(|&b| b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let is_blank = |b: &u8| *b == b' ' || *b == b'\t';
        let Some(first) = line.iter().position(|b| !is_blank(b)) else { continue };
        let last = line.iter().rposition(|b| !is_blank(b)).expect("a line with a non-blank byte has a last one");
        let word = &line[first..=last];
        if word[0] == b'#' {
            continue;
        }
        if !word.iter().all(|b| (0x20..=0x7E).contains(b)) {
            skipped.push(i + 1);
            continue;
        }
        words.push(normalize(std::str::from_utf8(word).expect("printable ASCII is UTF-8")));
    }
    (words, skipped)
}

/// `words` sorted (byte order, the order a search walks them in) and
/// without duplicates - the client's `sortedUniqueWords`.
pub fn sorted_unique(mut words: Vec<String>) -> Vec<String> {
    words.sort();
    words.dedup();
    words
}

/// Every string of `min` to `max` characters of `alphabet`, shortest first -
/// unless there are more than `limit`.
fn strings_of(alphabet: &[u8], min: usize, max: usize, limit: usize) -> Option<Vec<String>> {
    let count: u64 = (min..=max).map(|length| (alphabet.len() as u64).checked_pow(length as u32).unwrap_or(u64::MAX)).fold(0, u64::saturating_add);
    if count > limit as u64 {
        return None;
    }
    let mut out = Vec::new();
    let mut level = vec![String::new()];
    for length in 0..=max {
        if length >= min {
            out.extend(level.iter().cloned());
        }
        if length == max {
            break;
        }
        level = level.iter().flat_map(|s| alphabet.iter().map(move |&c| format!("{s}{}", c as char))).collect();
    }
    Some(out)
}

/// The strings one tail element stands for (see `expand_tails`).
fn tail_element_options(element: &str) -> Result<Vec<String>, String> {
    for (name, alphabet) in [("digits:", &b"0123456789"[..]), ("letters:", &b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"[..])] {
        let Some(range) = element.strip_prefix(name) else { continue };
        let (low, high) = range.split_once('-').unwrap_or((range, range));
        let number = |text: &str| (!text.is_empty() && text.len() <= 2 && text.bytes().all(|b| b.is_ascii_digit())).then(|| text.parse::<usize>().expect("digits"));
        let (Some(low), Some(high)) = (number(low), number(high)) else {
            return Err(format!("invalid tail element '{element}' (expected {name}N or {name}A-B, with A <= B)"));
        };
        if low > high {
            return Err(format!("invalid tail element '{element}' (expected {name}N or {name}A-B, with A <= B)"));
        }
        if high > MAX_TAIL_LENGTH {
            return Err(format!("tail element '{element}' makes tails longer than {MAX_TAIL_LENGTH} characters"));
        }
        return strings_of(alphabet, low, high, MAX_TAILS).ok_or_else(|| format!("tail element '{element}' makes more than {MAX_TAILS} tails"));
    }
    // One of some literal strings, separated by '|' - but no name has a ':',
    // so one with a ':' is a misspelled digits: or letters:.
    if element.contains(':') {
        return Err(format!("unknown tail element '{element}' (expected digits:A-B, letters:A-B, or literal strings separated by '|')"));
    }
    Ok(element.split('|').map(str::to_string).collect())
}

/// The tails a list of tail elements makes - the client's
/// `expandDictionaryTails` (client/src/engine/dictionary.h), which this has
/// to agree with exactly: every string made of one of each element's, in
/// turn - sorted, without duplicates, normalized as words are. An element is
/// `digits:A-B` (every string of A to B digits, `digits:N` for exactly N),
/// `letters:A-B` (the same of A-Z), or `X|Y|...` (one of these literal
/// strings, "" among them if the element ends with '|' or has "||" - none
/// with a ':', which only a misspelled `digits:`/`letters:` has). No
/// elements make `[""]`. An error for an element that's none of these, A >
/// B, a tail with a '\', longer than `MAX_TAIL_LENGTH` or with characters
/// past ASCII, or more than `MAX_TAILS` tails.
pub fn expand_tails(elements: &[String]) -> Result<Vec<String>, String> {
    let mut made = vec![String::new()];
    for element in elements {
        let options = tail_element_options(element)?;
        if made.len().checked_mul(options.len()).is_none_or(|count| count > MAX_TAILS) {
            return Err(format!("the tails are more than {MAX_TAILS} - use fewer, or shorter"));
        }
        made = made.iter().flat_map(|a| options.iter().map(move |b| format!("{a}{b}"))).collect();
    }
    for tail in &mut made {
        *tail = normalize(tail);
        if tail.len() > MAX_TAIL_LENGTH {
            return Err(format!("the tail '{tail}' is longer than {MAX_TAIL_LENGTH} characters"));
        }
        if tail.contains('\\') || !is_printable_ascii(tail) {
            return Err(format!("the tail '{tail}' has a '\\' or a character that isn't printable ASCII - a tail is part of a basename"));
        }
    }
    Ok(sorted_unique(made))
}

const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;

fn fnv1a64(bytes: &[u8], mut hash: u64) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// A 64-bit FNV-1a checksum of `words`, in order, each followed by '\n' -
/// the client's `wordListChecksum`. To tell two lists apart, not a
/// cryptographic hash.
pub fn checksum(words: &[String]) -> u64 {
    words.iter().fold(FNV_OFFSET_BASIS, |hash, word| fnv1a64(b"\n", fnv1a64(word.as_bytes(), hash)))
}

/// `value` as 16 lowercase hex digits, as the client writes checksums.
pub fn hex64(value: u64) -> String {
    format!("{value:016x}")
}

/// The candidates of a dictionary target, and their numbers - see the
/// module's doc comment. The client's `DictionarySpace`, but numbered in an
/// `i64`, since that's what the database stores: a target can't have more
/// candidates than that (the client's limit is 2^64).
#[derive(Debug)]
pub struct DictionarySpace {
    words: Vec<String>,
    separators: Vec<String>,
    min_words: i64,
    max_words: i64,
    tails: Vec<String>,
    /// Indexed by word count.
    block_start: Vec<i64>,
    block_size: Vec<i64>,
    size: i64,
}

impl DictionarySpace {
    /// `words` sorted and without duplicates (`sorted_unique`), at least
    /// one; `separators` at least one ("" for words written together), each
    /// once; 1 <= `min_words` <= `max_words` <= `MAX_WORDS`; `tails` at least
    /// one (`[""]` for none), as `expand_tails` makes them; and no more
    /// candidates than an `i64` holds.
    pub fn new(words: Vec<String>, separators: Vec<String>, min_words: i64, max_words: i64, tails: Vec<String>) -> Result<Self, String> {
        if words.is_empty() {
            return Err("no words to search".into());
        }
        if words.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("the words must be sorted and without duplicates".into());
        }
        if separators.is_empty() {
            return Err("no separators - give \"\" for words written together".into());
        }
        if separators.iter().enumerate().any(|(i, s)| separators[..i].contains(s)) {
            return Err("a separator is listed more than once".into());
        }
        if min_words < 1 || max_words < min_words || max_words > MAX_WORDS {
            return Err(format!("the word counts must be 1 <= min_words <= max_words <= {MAX_WORDS} (got {min_words} to {max_words})"));
        }
        if tails.is_empty() || tails.len() > MAX_TAILS {
            return Err(format!("the tails must be 1 to {MAX_TAILS} - give \"\" for none"));
        }
        if tails.windows(2).any(|pair| pair[0] >= pair[1]) || tails.iter().any(|t| t.len() > MAX_TAIL_LENGTH || t.contains('\\')) {
            return Err(format!("the tails must be sorted, without duplicates, at most {MAX_TAIL_LENGTH} characters each, and without a '\\'"));
        }
        let (w, s, t) = (words.len() as i64, separators.len() as i64, tails.len() as i64);
        let too_many = || {
            let tails = if t > 1 { format!(" and {t} tails") } else { String::new() };
            format!("{max_words} words of {w}, with {s} separators{tails}, are more candidates than the server can number - use fewer words")
        };
        let mut block_start = vec![0; max_words as usize + 1];
        let mut block_size = vec![0; max_words as usize + 1];
        let mut total: i64 = 0;
        for k in min_words..=max_words {
            // W^k * S^(k-1) * T
            let mut size = w;
            for _ in 1..k {
                size = size.checked_mul(w).and_then(|n| n.checked_mul(s)).ok_or_else(too_many)?;
            }
            size = size.checked_mul(t).ok_or_else(too_many)?;
            block_start[k as usize] = total;
            block_size[k as usize] = size;
            total = total.checked_add(size).ok_or_else(too_many)?;
        }
        Ok(DictionarySpace { words, separators, min_words, max_words, tails, block_start, block_size, size: total })
    }

    pub fn words(&self) -> &[String] {
        &self.words
    }

    pub fn separators(&self) -> &[String] {
        &self.separators
    }

    pub fn min_words(&self) -> i64 {
        self.min_words
    }

    pub fn max_words(&self) -> i64 {
        self.max_words
    }

    pub fn tails(&self) -> &[String] {
        &self.tails
    }

    /// How many candidates there are, of every word count.
    pub fn size(&self) -> i64 {
        self.size
    }

    /// The number of the first k-word candidate, and how many there are
    /// (min_words <= k <= max_words).
    pub fn block_start(&self, k: i64) -> i64 {
        self.block_start[k as usize]
    }

    pub fn block_size(&self, k: i64) -> i64 {
        self.block_size[k as usize]
    }

    /// How many words candidate `number` (< `size()`) has.
    pub fn word_count_of(&self, number: i64) -> i64 {
        (self.min_words..=self.max_words).find(|&k| number < self.block_start(k) + self.block_size(k)).expect("number < size()")
    }

    /// Candidate `number` (< `size()`), as positions in the word, separator
    /// and tail lists: its words, the separators between them, and its tail.
    pub fn decode(&self, number: i64) -> (Vec<usize>, Vec<usize>, usize) {
        let k = self.word_count_of(number);
        let mut local = number - self.block_start(k);
        let (w, s, t) = (self.words.len() as i64, self.separators.len() as i64, self.tails.len() as i64);
        let mut words = vec![0; k as usize];
        let mut separators = vec![0; k as usize - 1];
        // The wheels from the right: the tail, the last word, then separator
        // k-1 and word k-1, and so on.
        let tail = (local % t) as usize;
        local /= t;
        words[k as usize - 1] = (local % w) as usize;
        local /= w;
        for i in (0..k as usize - 1).rev() {
            separators[i] = (local % s) as usize;
            local /= s;
            words[i] = (local % w) as usize;
            local /= w;
        }
        (words, separators, tail)
    }

    /// Candidate `number`'s text: its words and separators, in order, and
    /// its tail.
    pub fn text(&self, number: i64) -> String {
        let (words, separators, tail) = self.decode(number);
        let mut text = self.words[words[0]].clone();
        for (word, separator) in words[1..].iter().zip(&separators) {
            text += &self.separators[*separator];
            text += &self.words[*word];
        }
        text + &self.tails[tail]
    }
}

/// The lower and upper bound of a dictionary target, as whole filenames,
/// both inclusive - either may be left out. Compared byte by byte, so they
/// need to be normalized like the filenames are. The client's
/// `FilenameBounds`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilenameBounds {
    pub lower: Option<String>,
    pub upper: Option<String>,
}

/// Where the filenames that start with some text are - see
/// `FilenameBounds::classify_start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// No filename that starts with it is within the bounds.
    Outside,
    /// Every filename that starts with it is.
    Inside,
    /// Some may be, some not.
    Mixed,
}

/// Where `start` and `bound` first differ: Less if `start` sorts first
/// there, Greater if `bound` does, Equal if they don't differ on their
/// common length.
fn compare_common(start: &[u8], bound: &[u8]) -> Ordering {
    start.iter().zip(bound).map(|(a, b)| a.cmp(b)).find(|o| o.is_ne()).unwrap_or(Ordering::Equal)
}

impl FilenameBounds {
    pub fn contains(&self, filename: &str) -> bool {
        self.lower.as_deref().is_none_or(|lower| filename >= lower) && self.upper.as_deref().is_none_or(|upper| filename <= upper)
    }

    /// Where the filenames that start with `start` (`start` itself included)
    /// are - decided by `start` alone.
    pub fn classify_start(&self, start: &str) -> Verdict {
        let start = start.as_bytes();
        let mut lower_settled = true; // every filename >= lower
        if let Some(lower) = &self.lower {
            let lower = lower.as_bytes();
            match compare_common(start, lower) {
                Ordering::Less => return Verdict::Outside, // every filename < lower
                // Differing upwards, or lower is a prefix of start (and so of every filename).
                c => lower_settled = c.is_gt() || start.len() >= lower.len(),
            }
        }
        let mut upper_settled = true; // every filename <= upper
        if let Some(upper) = &self.upper {
            let upper = upper.as_bytes();
            match compare_common(start, upper) {
                Ordering::Greater => return Verdict::Outside, // every filename > upper
                // upper is a proper prefix of every filename.
                Ordering::Equal if start.len() > upper.len() => return Verdict::Outside,
                c => upper_settled = c.is_lt(),
            }
        }
        if lower_settled && upper_settled {
            Verdict::Inside
        } else {
            Verdict::Mixed
        }
    }
}

/// A stretch of candidate numbers, `[start, end)`, all of `words` words,
/// that holds every candidate of that many words within a target's bounds -
/// see `windows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub words: i64,
    pub start: i64,
    pub end: i64,
}

/// Where a dictionary target's candidates within its bounds are: one window
/// per word count that has any, from the first first word (with the
/// separator after it) the bounds don't rule out to the last - exactly, for
/// one-word candidates (to the first and last tail of those words within),
/// and for more, every candidate those first words start. The numbers turn their first word slowest, so this is all a
/// bound's effect on the numbers, but for the odd first word in between
/// whose candidates the bounds leave out after all (as `AB_` sorts after
/// `ABC`): the client skips those. Without bounds, one window per word count
/// with all of it. The server carves ranges from these.
pub fn windows(space: &DictionarySpace, prefix: &str, suffix: &str, bounds: &FilenameBounds) -> Vec<Window> {
    let root = bounds.classify_start(prefix);
    if root == Verdict::Outside {
        return Vec::new();
    }
    let inside = root == Verdict::Inside;
    let first_last = |within: &dyn Fn(usize) -> bool, count: usize| -> Option<(usize, usize)> {
        if inside {
            return Some((0, count - 1));
        }
        let first = (0..count).find(|&i| within(i))?;
        let last = (0..count).rev().find(|&i| within(i))?;
        Some((first, last))
    };
    let (words, separators) = (space.words(), space.separators());

    let mut windows = Vec::new();
    if space.min_words() == 1 {
        let tails = space.tails();
        let tail_within = |i: usize, j: usize| bounds.contains(&format!("{prefix}{}{}{suffix}", words[i], tails[j]));
        // Whether any of word i's candidates are within.
        let within = |i: usize| match bounds.classify_start(&format!("{prefix}{}", words[i])) {
            Verdict::Outside => false,
            Verdict::Inside => true,
            Verdict::Mixed => (0..tails.len()).any(|j| tail_within(i, j)),
        };
        if let Some((first, last)) = first_last(&within, words.len()) {
            let (first_tail, last_tail) = if inside {
                (0, tails.len() - 1)
            } else {
                let first_tail = (0..tails.len()).find(|&j| tail_within(first, j)).unwrap_or(0);
                let last_tail = (0..tails.len()).rev().find(|&j| tail_within(last, j)).unwrap_or(tails.len() - 1);
                (first_tail, last_tail)
            };
            let (start, t) = (space.block_start(1), tails.len());
            windows.push(Window { words: 1, start: start + (first * t + first_tail) as i64, end: start + (last * t + last_tail) as i64 + 1 });
        }
    }
    if space.max_words() >= 2 {
        // A child of the root is a first word and the separator after it,
        // the separator turning faster - the same for every word count.
        let pairs = words.len() * separators.len();
        let within = |c: usize| {
            let start = format!("{prefix}{}{}", words[c / separators.len()], separators[c % separators.len()]);
            bounds.classify_start(&start) != Verdict::Outside
        };
        if let Some((first, last)) = first_last(&within, pairs) {
            for k in space.min_words().max(2)..=space.max_words() {
                let child_size = space.block_size(k) / pairs as i64;
                let start = space.block_start(k);
                windows.push(Window { words: k, start: start + first as i64 * child_size, end: start + (last as i64 + 1) * child_size });
            }
        }
    }
    windows
}

/// Reads a target's JSON array of strings column (`targets.word_lists`,
/// `targets.separators`).
pub fn parse_string_list(json: Option<&str>) -> Result<Vec<String>, AppError> {
    serde_json::from_str(json.unwrap_or("[]")).map_err(|err| AppError::Internal(format!("a stored list of strings doesn't read back: {err}")))
}

/// Everything about a dictionary target the server works out from its word
/// lists - what a claim and the dashboard need. Never changes: a target's
/// word lists, separators, word counts, prefix, suffix and bounds can't be
/// changed once it's made, and a word list can't be changed once stored.
#[derive(Debug)]
pub struct DictionaryTarget {
    pub space: DictionarySpace,
    pub windows: Vec<Window>,
    /// The word lists' names, in the order given, and each one's checksum
    /// (of its words sorted, as stored) - what a client checks the copy it
    /// has of each against.
    pub word_lists: Vec<String>,
    pub word_list_checksums: Vec<String>,
    /// `checksum` of every word of every list, merged: sorted, without
    /// duplicates - the words the candidates are made of.
    pub words_checksum: String,
    /// The tail elements, as given (empty for none), and `checksum` of the
    /// tails they make - what a client checks its own expansion against.
    pub tail_elements: Vec<String>,
    pub tails_checksum: String,
    pub bounds: FilenameBounds,
}

impl DictionaryTarget {
    /// How many candidates the windows hold - everything a search of the
    /// target is handed.
    pub fn window_candidates(&self) -> i64 {
        self.windows.iter().map(|w| w.end - w.start).sum()
    }
}

/// The bounds a dictionary target was made with: "" is none.
pub fn target_bounds(target: &Target) -> FilenameBounds {
    let bound = |b: &str| (!b.is_empty()).then(|| b.to_string());
    FilenameBounds { lower: bound(&target.lower_bound), upper: bound(&target.upper_bound) }
}

/// Builds a dictionary target's `DictionaryTarget` from its settings and its
/// word lists, read from the database. A `BadRequest` for settings that
/// can't make one (a list that isn't stored, too many candidates, bounds
/// that leave none) - which only a target being made can have.
#[allow(clippy::too_many_arguments)]
pub async fn load(
    conn: &mut sqlx::SqliteConnection,
    word_lists: &[String],
    separators: &[String],
    min_words: i64,
    max_words: i64,
    tail_elements: &[String],
    prefix: &str,
    suffix: &str,
    bounds: FilenameBounds,
) -> Result<DictionaryTarget, AppError> {
    if word_lists.is_empty() {
        return Err(AppError::BadRequest("a dictionary target needs at least one word list".into()));
    }
    let mut words = Vec::new();
    let mut word_list_checksums = Vec::new();
    for (i, name) in word_lists.iter().enumerate() {
        if word_lists[..i].contains(name) {
            return Err(AppError::BadRequest(format!("the word list '{name}' is listed more than once")));
        }
        let row: Option<(Vec<u8>, String)> = sqlx::query_as("SELECT content, checksum FROM word_lists WHERE name = ?").bind(name).fetch_optional(&mut *conn).await?;
        let Some((content, list_checksum)) = row else {
            return Err(AppError::BadRequest(format!("there's no word list named '{name}' - upload it first (PUT /api/v1/admin/word-lists/{name})")));
        };
        words.extend(parse_word_list(&content).0);
        word_list_checksums.push(list_checksum);
    }
    let words = sorted_unique(words);
    let words_checksum = hex64(checksum(&words));
    let tails = expand_tails(tail_elements).map_err(|err| AppError::BadRequest(format!("invalid tails: {err}")))?;
    let tails_checksum = hex64(checksum(&tails));
    let space = DictionarySpace::new(words, separators.to_vec(), min_words, max_words, tails).map_err(AppError::BadRequest)?;
    let windows = windows(&space, prefix, suffix, &bounds);
    if windows.is_empty() {
        return Err(AppError::BadRequest("no candidate is within the bounds".into()));
    }
    Ok(DictionaryTarget {
        space,
        windows,
        word_lists: word_lists.to_vec(),
        word_list_checksums,
        words_checksum,
        tail_elements: tail_elements.to_vec(),
        tails_checksum,
        bounds,
    })
}

/// Every dictionary target's `DictionaryTarget`, by target id, built the
/// first time one is needed: building one reads and sorts all of its words
/// (about 10 ms for english-1), which a claim or a dashboard refresh
/// shouldn't do each time. Nothing in one ever changes (see
/// `DictionaryTarget`), so an entry is never out of date.
#[derive(Default)]
pub struct DictionaryCache(Mutex<HashMap<i64, Arc<DictionaryTarget>>>);

impl DictionaryCache {
    pub async fn get(&self, conn: &mut sqlx::SqliteConnection, target: &Target) -> Result<Arc<DictionaryTarget>, AppError> {
        if let Some(found) = self.0.lock().expect("dictionary cache lock").get(&target.id) {
            return Ok(found.clone());
        }
        let loaded = load(
            conn,
            &parse_string_list(target.word_lists.as_deref())?,
            &parse_string_list(target.separators.as_deref())?,
            target.min_words.unwrap_or(1),
            target.max_words.unwrap_or(1),
            &parse_string_list(target.tails.as_deref())?,
            &target.prefix,
            &target.suffix,
            target_bounds(target),
        )
        .await
        // Its settings made one when it was created, and nothing they read
        // has changed since - so this is the server's fault, not the caller's.
        .map_err(|err| match err {
            AppError::BadRequest(msg) => AppError::Internal(format!("dictionary target {} no longer loads: {msg}", target.id)),
            other => other,
        })?;
        let loaded = Arc::new(loaded);
        self.0.lock().expect("dictionary cache lock").insert(target.id, loaded.clone());
        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn normalize_makes_lowercase_uppercase_and_slashes_backslashes() {
        assert_eq!(normalize("music/bg_a-z.wav"), "MUSIC\\BG_A-Z.WAV");
        assert_eq!(normalize("ÉÀé{}"), "ÉÀé{}", "only ASCII letters change");
    }

    #[test]
    fn parse_word_list_reads_as_the_client_does() {
        let (words, skipped) = parse_word_list(b"alpha\r\n  beta\t \n\n# comment\n \t\ngam/ma\ncaf\xC3\xA9\nlast");
        assert_eq!(words, strings(&["ALPHA", "BETA", "GAM\\MA", "LAST"]));
        assert_eq!(skipped, vec![7], "the line with a byte past ASCII is skipped, by its line number");
        let (words, _) = parse_word_list(b"a\r\r\nb\n");
        assert_eq!(words, strings(&["B"]), "only one \\r is taken off - the other makes the line unprintable");
        let (words, _) = parse_word_list(b"  # not a comment? yes: it starts with '#' once trimmed\nx # y\n");
        assert_eq!(words, strings(&["X # Y"]));
        assert_eq!(parse_word_list(b"").0, Vec::<String>::new());
    }

    #[test]
    fn checksums_match_the_clients() {
        // The values client/tests/wordlist_test.cpp checks.
        assert_eq!(fnv1a64(b"", FNV_OFFSET_BASIS), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64(b"a", FNV_OFFSET_BASIS), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(checksum(&strings(&["A", "B"])), 0x76FC_A88C_B6D9_040E);
        assert_eq!(hex64(0x0123_4567_89AB_CDEF), "0123456789abcdef");
    }

    #[test]
    fn english_1_has_the_words_the_client_compiles_in() {
        let (words, skipped) = parse_word_list(ENGLISH_1_TEXT);
        assert!(skipped.is_empty());
        let words = sorted_unique(words);
        assert_eq!(words.len(), 63_875);
        assert_eq!(checksum(&words), ENGLISH_1_CHECKSUM);
        assert!(ENGLISH_1_TEXT == include_bytes!("../../../client/data/english-1.txt"), "server/data/english-1.txt is a copy of the client's");
    }

    /// Every candidate of `space`, in number order, made the slow way.
    fn every_candidate(space: &DictionarySpace) -> Vec<String> {
        let mut all = Vec::new();
        for k in space.min_words()..=space.max_words() {
            let mut texts = space.words().to_vec();
            for _ in 1..k {
                let mut longer = Vec::new();
                for text in &texts {
                    for separator in space.separators() {
                        for word in space.words() {
                            longer.push(format!("{text}{separator}{word}"));
                        }
                    }
                }
                texts = longer;
            }
            all.extend(texts.iter().flat_map(|text| space.tails().iter().map(move |tail| format!("{text}{tail}"))));
        }
        all
    }

    fn no_tails() -> Vec<String> {
        vec![String::new()]
    }

    #[test]
    fn candidates_are_numbered_like_an_odometer_with_the_last_word_fastest() {
        let space = DictionarySpace::new(strings(&["A", "BB", "C"]), strings(&["", "_"]), 1, 3, no_tails()).unwrap();
        assert_eq!(space.size(), 3 + 3 * 2 * 3 + 9 * 4 * 3);
        assert_eq!((space.block_start(1), space.block_size(1)), (0, 3));
        assert_eq!((space.block_start(2), space.block_size(2)), (3, 18));
        assert_eq!((space.block_start(3), space.block_size(3)), (21, 108));
        let all = every_candidate(&space);
        assert_eq!(all.len() as i64, space.size());
        for (number, text) in all.iter().enumerate() {
            assert_eq!(&space.text(number as i64), text, "number {number}");
        }
        assert_eq!(space.text(3), "AA");
        assert_eq!(space.text(4), "ABB");
        assert_eq!(space.text(6), "A_A");
        assert_eq!(space.word_count_of(20), 2);
        assert_eq!(space.word_count_of(21), 3);

        let two_only = DictionarySpace::new(strings(&["A", "B"]), strings(&["-"]), 2, 2, no_tails()).unwrap();
        assert_eq!(two_only.size(), 4);
        assert_eq!((0..4).map(|n| two_only.text(n)).collect::<Vec<_>>(), strings(&["A-A", "A-B", "B-A", "B-B"]));
    }

    #[test]
    fn tails_turn_fastest_of_all() {
        // The client's worked example (client/tests/dictionary_test.cpp).
        let space = DictionarySpace::new(strings(&["A", "B"]), strings(&["", "_"]), 1, 2, strings(&["", "1", "2B"])).unwrap();
        assert_eq!(space.size(), (2 + 2 * 2 * 2) * 3);
        let all = every_candidate(&space);
        for (number, text) in all.iter().enumerate() {
            assert_eq!(&space.text(number as i64), text, "number {number}");
        }
        for (number, text) in [(0, "A"), (1, "A1"), (2, "A2B"), (3, "B"), (6, "AA"), (7, "AA1"), (9, "AB"), (29, "B_B2B")] {
            assert_eq!(space.text(number), text);
        }
        assert_eq!(space.decode(6 + ((2 + 1) * 2) * 3 + 2), (vec![1, 0], vec![1], 2), "(((w1 * S + s1) * W + w2) * T + t)");
        let new = |tails: &[&str]| DictionarySpace::new(strings(&["A"]), strings(&[""]), 1, 1, strings(tails));
        assert!(new(&[]).is_err(), "no tails at all: \"\" is none");
        assert!(new(&["1", ""]).is_err(), "unsorted");
        assert!(new(&["", "123456789"]).is_err(), "too long");
        assert!(new(&["", "A\\B"]).is_err(), "a '\\'");
        assert_eq!(new(&["", "1", "2"]).unwrap().size(), 3);
    }

    #[test]
    fn tails_expand_as_the_client_expands_them() {
        let expand = |elements: &[&str]| expand_tails(&strings(elements));
        assert_eq!(expand(&[]).unwrap(), strings(&[""]));
        let digits = expand(&["digits:1-2"]).unwrap();
        assert_eq!((digits.len(), &digits[..3], &digits[11]), (110, &strings(&["0", "00", "01"])[..], &"1".to_string()));
        assert_eq!(expand(&["letters:0-1"]).unwrap().len(), 27);
        assert_eq!(expand(&["digits:1", "letters:1"]).unwrap().len(), 260);
        assert_eq!(expand(&["_|", "digits:2"]).unwrap()[100], "_00");
        assert_eq!(expand(&["x|y||", "letters:0-1"]).unwrap().len(), 79, "X and Y made twice, kept once");
        assert_eq!(expand(&["digits:0-1", "digits:0-1"]).unwrap().len(), 111);
        assert_eq!(expand(&["digits:6"]).unwrap().len(), 1_000_000);
        // The checksums the client's tests pin too, worked out a third way
        // (in Python): the two expand alike.
        assert_eq!(hex64(checksum(&expand(&["digits:1-2", "letters:0-1"]).unwrap())), "3c4d231dd57c0078");
        assert_eq!(hex64(checksum(&expand(&["digits:0-2", "_|"]).unwrap())), "a0735b5058f2e9be");
        for (elements, in_error) in [
            (&["digits:2-1"][..], "A <= B"),
            (&["digit:1-2"], "unknown tail element"),
            (&["digits:"], "digits:"),
            (&["letters:a"], "letters:"),
            (&["digits:9"], "longer than"),
            (&["digits:7"], "more than"),
            (&["digits:3", "letters:3"], "more than"),
            (&["ABCDE", "FGHIJ"], "longer than"),
            (&["A/B"], "'\\'"),
            (&["\u{e9}"], "printable ASCII"),
        ] {
            let err = expand(elements).unwrap_err();
            assert!(err.contains(in_error), "{elements:?}: {err}");
        }
    }

    #[test]
    fn a_space_breaking_a_rule_is_refused() {
        let new = |words: &[&str], separators: &[&str], min, max| DictionarySpace::new(strings(words), strings(separators), min, max, no_tails());
        assert!(new(&[], &[""], 1, 1).is_err());
        assert!(new(&["B", "A"], &[""], 1, 1).is_err(), "unsorted");
        assert!(new(&["A", "A"], &[""], 1, 1).is_err(), "a duplicate");
        assert!(new(&["A"], &[], 1, 1).is_err());
        assert!(new(&["A"], &["_", "_"], 1, 1).is_err());
        assert!(new(&["A"], &[""], 0, 1).is_err());
        assert!(new(&["A"], &[""], 2, 1).is_err());
        assert!(new(&["A"], &[""], 1, MAX_WORDS + 1).is_err());
        assert!(new(&["A"], &[""], 1, MAX_WORDS).is_ok());
        // 1,000 words and 8 separators: 2^63 is about 9.2e18, and 4 words
        // are 1e12 * 512 = 5.1e14, 6 words 1e18 * 8^5 = 3.3e22.
        let words: Vec<String> = (0..1000).map(|i| format!("W{i:04}")).collect();
        let seps: Vec<String> = (0..8).map(|i| format!("{i}")).collect();
        assert!(DictionarySpace::new(words.clone(), seps.clone(), 1, 4, no_tails()).is_ok());
        let err = DictionarySpace::new(words, seps, 1, 6, no_tails()).unwrap_err();
        assert!(err.contains("more candidates than the server can number"), "{err}");
    }

    fn bounds(lower: Option<&str>, upper: Option<&str>) -> FilenameBounds {
        FilenameBounds { lower: lower.map(str::to_string), upper: upper.map(str::to_string) }
    }

    #[test]
    fn classify_start_agrees_with_every_continuation() {
        // Every string of up to 4 characters from a small alphabet, as
        // starts and as their continuations, against bounds of every kind.
        let alphabet = ['A', 'B', 'C'];
        let mut strings = vec![String::new()];
        let mut layer = vec![String::new()];
        for _ in 0..4 {
            layer = layer.iter().flat_map(|s| alphabet.iter().map(move |c| format!("{s}{c}"))).collect();
            strings.extend(layer.iter().cloned());
        }
        let bound_choices: Vec<Option<&str>> = vec![None, Some(""), Some("A"), Some("B"), Some("BA"), Some("BAC"), Some("C"), Some("CCC")];
        for &lower in &bound_choices {
            for &upper in &bound_choices {
                let b = bounds(lower, upper);
                for start in strings.iter().filter(|s| s.len() <= 2) {
                    let continuations: Vec<&String> = strings.iter().filter(|s| s.starts_with(start.as_str())).collect();
                    let within = continuations.iter().filter(|s| b.contains(s)).count();
                    match b.classify_start(start) {
                        Verdict::Outside => assert_eq!(within, 0, "{start:?} in {b:?}: Outside, yet some continuation is within"),
                        Verdict::Inside => assert_eq!(within, continuations.len(), "{start:?} in {b:?}: Inside, yet some continuation isn't"),
                        Verdict::Mixed => {}
                    }
                }
            }
        }
        // And not needlessly Mixed - which is never wrong, only less tight.
        assert_eq!(bounds(Some("B"), Some("C")).classify_start("BA"), Verdict::Inside);
        assert_eq!(bounds(Some("B"), Some("C")).classify_start("B"), Verdict::Inside, "the lower bound itself, and all after it");
        assert_eq!(bounds(None, Some("B")).classify_start("BA"), Verdict::Outside, "the upper bound a proper prefix: all after it");
        assert_eq!(bounds(None, Some("B")).classify_start("B"), Verdict::Mixed, "B itself is within, BA isn't");
        assert_eq!(bounds(Some("B"), Some("C")).classify_start("C"), Verdict::Mixed, "C itself is within, CA isn't");
        assert_eq!(bounds(Some("B"), None).classify_start("A"), Verdict::Outside);
        assert_eq!(bounds(None, None).classify_start("ANY"), Verdict::Inside);
    }

    #[test]
    fn windows_hold_every_candidate_within_the_bounds() {
        let words = strings(&["A", "AB", "ABC", "B", "BA", "C", "CAB"]);
        let separators = strings(&["", "_", "B"]);
        let space = DictionarySpace::new(words, separators, 1, 3, no_tails()).unwrap();
        let all = every_candidate(&space);
        let cases = [
            ("X\\", "", bounds(None, None)),
            ("X\\", ".WAV", bounds(Some("X\\AB"), Some("X\\BA_C.WAV"))),
            ("X\\", ".WAV", bounds(Some("X\\ABC_"), None)),
            ("X\\", "", bounds(None, Some("X\\ABCZ"))),
            ("X\\", "", bounds(Some("X\\B"), Some("X\\B"))),
            ("", "", bounds(Some("CA"), Some("CAB_A"))),
            ("Y\\", "", bounds(Some("X\\"), Some("X\\ZZZ"))),
        ];
        for (prefix, suffix, b) in cases {
            let windows = windows(&space, prefix, suffix, &b);
            let mut in_window = vec![false; all.len()];
            for w in &windows {
                assert!(w.start < w.end, "{w:?}");
                assert!(w.start >= space.block_start(w.words) && w.end <= space.block_start(w.words) + space.block_size(w.words), "{w:?}");
                for n in w.start..w.end {
                    in_window[n as usize] = true;
                }
            }
            let within: Vec<usize> = (0..all.len()).filter(|&n| b.contains(&format!("{prefix}{}{suffix}", all[n]))).collect();
            for &n in &within {
                assert!(in_window[n], "{prefix:?} {suffix:?} {b:?}: candidate {n} ({}) is within the bounds but in no window", all[n]);
            }
            for k in space.min_words()..=space.max_words() {
                let ks: Vec<usize> = within.iter().copied().filter(|&n| space.word_count_of(n as i64) == k).collect();
                let window = windows.iter().find(|w| w.words == k);
                match (ks.first(), window) {
                    (None, None) => {}
                    (Some(_), None) => panic!("no window for {k} words"),
                    (None, Some(w)) => assert!(k > 1, "an empty one-word window {w:?}"),
                    (Some(&first), Some(w)) if k == 1 => {
                        assert_eq!((w.start, w.end), (first as i64, *ks.last().unwrap() as i64 + 1), "one-word windows are exact");
                    }
                    (Some(_), Some(_)) => {}
                }
            }
        }
        assert_eq!(
            windows(&space, "X\\", "", &bounds(None, None)),
            vec![
                Window { words: 1, start: 0, end: 7 },
                Window { words: 2, start: 7, end: 7 + 7 * 3 * 7 },
                Window { words: 3, start: 7 + 147, end: 7 + 147 + 7 * 7 * 7 * 9 },
            ]
        );
        assert!(windows(&space, "Y\\", "", &bounds(Some("X\\"), Some("X\\ZZZ"))).is_empty());
    }

    #[test]
    fn one_word_windows_are_exact_with_tails_too() {
        let words = strings(&["A", "AB", "B", "BA", "C"]);
        let space = DictionarySpace::new(words, strings(&["", "_"]), 1, 2, strings(&["", "1", "1B", "Z"])).unwrap();
        let all = every_candidate(&space);
        for b in [bounds(Some("X\\A1"), Some("X\\BA1B")), bounds(Some("X\\AB1B"), None), bounds(None, Some("X\\A1")), bounds(Some("X\\B_A1"), None)] {
            let windows = windows(&space, "X\\", "", &b);
            let within: Vec<i64> = (0..space.block_size(1)).filter(|&n| b.contains(&format!("X\\{}", all[n as usize]))).collect();
            let one = windows.iter().find(|w| w.words == 1);
            match within.first() {
                None => assert!(one.is_none(), "{b:?}"),
                Some(&first) => assert_eq!(one.map(|w| (w.start, w.end)), Some((first, within.last().unwrap() + 1)), "{b:?}"),
            }
            for n in (0..space.size()).filter(|&n| b.contains(&format!("X\\{}", all[n as usize]))) {
                assert!(windows.iter().any(|w| (w.start..w.end).contains(&n)), "{b:?}: candidate {n} ({}) in no window", all[n as usize]);
            }
        }
    }

    #[test]
    fn windows_narrow_by_first_word() {
        // english-1-like: the bounds pick out the first words from BA to BE.
        let words: Vec<String> = ["AA", "AB", "BA", "BB", "BC", "BE", "BF", "CA"].iter().map(|s| s.to_string()).collect();
        let space = DictionarySpace::new(words, strings(&["", "_"]), 1, 2, no_tails()).unwrap();
        let w = windows(&space, "MUSIC\\", ".WAV", &bounds(Some("MUSIC\\BA"), Some("MUSIC\\BE_ZZ")));
        assert_eq!(w[0], Window { words: 1, start: 2, end: 6 }, "BA to BE");
        // Two words: the children (first word, separator) from (BA, "") to
        // (BE, "_"), 8 candidates each.
        assert_eq!(w[1], Window { words: 2, start: 8 + 2 * 2 * 8, end: 8 + 6 * 2 * 8 });
    }
}
