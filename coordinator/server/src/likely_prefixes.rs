//! The prefixes automatic priority ranges are made from (see
//! `AdminCreateTargetRequest::auto_priority` and
//! `ranges::create_next_auto_priority_range`), most likely first: the starts
//! of the words in a list of real filenames, ranked by how many words start
//! that way, and after those the starts of English words, ranked the same
//! way. The names found so far count as real filenames too. Counted at
//! startup and again after each find, so a claim only ever walks a ready
//! list.
//!
//! The counting follows tools/prefix_counter.py (the listfile) and
//! tools/next_letters.py --rank (the dictionary).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

use sqlx::SqlitePool;

/// Real filenames, one per line - a copy of tools/sc-listfile.txt.
const LISTFILE: &str = include_str!("../data/sc-listfile.txt");

/// Where to look for an English word list when `DICTIONARY_PATH` isn't set -
/// the Docker image installs `wamerican` for the first.
const DEFAULT_DICTIONARIES: &[&str] = &["/usr/share/dict/words", "/usr/share/dict/american-english", "/usr/share/dict/british-english"];

/// The prefix lengths `prefix_len_for` ever asks for.
const PREFIX_LENS: [usize; 2] = [2, 3];

/// How many leading characters of a `candidate_len`-character candidate to
/// prioritize by: none below 11, where the whole length is quick to search
/// anyway.
pub fn prefix_len_for(candidate_len: i64) -> Option<usize> {
    match candidate_len {
        11 => Some(2),
        len if len >= 12 => Some(3),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Listfile,
    Dictionary,
}

impl Source {
    /// As stored in `priority_ranges.auto_source`.
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Listfile => "listfile",
            Source::Dictionary => "dictionary",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LikelyPrefix {
    /// Uppercase letters and digits.
    pub prefix: String,
    /// How many words start with it.
    pub count: i64,
    pub source: Source,
}

/// Every prefix of each length in `PREFIX_LENS`, most likely first.
#[derive(Default)]
pub struct Ranking {
    by_len: HashMap<usize, Vec<LikelyPrefix>>,
}

impl Ranking {
    /// Every prefix of `len` characters, most likely first.
    pub fn get(&self, len: usize) -> &[LikelyPrefix] {
        self.by_len.get(&len).map_or(&[], Vec::as_slice)
    }
}

/// The listfile's words and the dictionary's counts, read once, and the
/// ranking made from them and the names found so far - remade by
/// `include_matches` after every find.
#[derive(Default)]
pub struct LikelyPrefixes {
    listfile_words: HashSet<String>,
    dictionary_counts: HashMap<usize, HashMap<String, i64>>,
    ranking: RwLock<Arc<Ranking>>,
}

impl LikelyPrefixes {
    /// The built-in listfile, and the dictionary at `DICTIONARY_PATH` or the
    /// first of `DEFAULT_DICTIONARIES` that exists - without one, only the
    /// listfile's prefixes are prioritized. No matches yet - see
    /// `refresh_matches`.
    pub fn load() -> Self {
        let path = std::env::var("DICTIONARY_PATH").ok().or_else(|| DEFAULT_DICTIONARIES.iter().find(|p| Path::new(p).is_file()).map(|p| p.to_string()));
        let dictionary = match &path {
            Some(path) => match std::fs::read(path) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(err) => {
                    tracing::warn!(%path, %err, "could not read the dictionary - automatic priority ranges will only use the listfile");
                    String::new()
                }
            },
            None => {
                tracing::warn!("no dictionary found (set DICTIONARY_PATH) - automatic priority ranges will only use the listfile");
                String::new()
            }
        };
        Self::from_texts(LISTFILE, &dictionary)
    }

    /// `listfile`: filenames, one per line. `dictionary`: words, one per line.
    pub fn from_texts(listfile: &str, dictionary: &str) -> Self {
        let prefixes = LikelyPrefixes {
            listfile_words: listfile_words(listfile),
            dictionary_counts: PREFIX_LENS.into_iter().map(|len| (len, dictionary_counts(dictionary, len))).collect(),
            ranking: Default::default(),
        };
        prefixes.include_matches(&[]);
        prefixes
    }

    /// The current ranking.
    pub fn ranking(&self) -> Arc<Ranking> {
        self.ranking.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Remakes the ranking with the words of `matches` (found filenames)
    /// counted as the listfile's own.
    pub fn include_matches(&self, matches: &[String]) {
        let mut words = self.listfile_words.clone();
        words.extend(listfile_words(&matches.join("\n")));
        let by_len = PREFIX_LENS
            .into_iter()
            .map(|len| {
                let mut ranked: Vec<LikelyPrefix> =
                    by_count(listfile_counts(&words, len)).map(|(prefix, count)| LikelyPrefix { prefix, count, source: Source::Listfile }).collect();
                let seen: HashSet<String> = ranked.iter().map(|p| p.prefix.clone()).collect();
                ranked.extend(
                    by_count(self.dictionary_counts.get(&len).cloned().unwrap_or_default())
                        .filter(|(prefix, _)| !seen.contains(prefix))
                        .map(|(prefix, count)| LikelyPrefix { prefix, count, source: Source::Dictionary }),
                );
                (len, ranked)
            })
            .collect();
        *self.ranking.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(Ranking { by_len });
    }

    /// `include_matches` with every name found so far - at startup, and
    /// after each find (see `handlers::complete`).
    pub async fn refresh_matches(&self, pool: &SqlitePool) -> Result<(), sqlx::Error> {
        let matches: Vec<String> =
            sqlx::query_scalar("SELECT DISTINCT found_filename FROM targets WHERE found_filename IS NOT NULL AND is_virtual = 0").fetch_all(pool).await?;
        self.include_matches(&matches);
        let ranking = self.ranking();
        for len in PREFIX_LENS {
            let ranked = ranking.get(len);
            let from_listfile = ranked.iter().filter(|p| p.source == Source::Listfile).count();
            tracing::info!(len, matches = matches.len(), from_listfile, from_dictionary = ranked.len() - from_listfile, "counted likely prefixes");
        }
        Ok(())
    }
}

/// Most words first, then alphabetically.
fn by_count(counts: HashMap<String, i64>) -> impl Iterator<Item = (String, i64)> {
    let mut sorted: Vec<(String, i64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sorted.into_iter()
}

/// Every distinct word of every filename in `listfile`: split on anything
/// but ASCII letters and digits, uppercased, after dropping an extension of
/// exactly three letters or digits.
fn listfile_words(listfile: &str) -> HashSet<String> {
    let mut words = HashSet::new();
    for line in listfile.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let bytes = line.as_bytes();
        let has_extension = bytes.len() >= 4 && bytes[bytes.len() - 4] == b'.' && bytes[bytes.len() - 3..].iter().all(u8::is_ascii_alphanumeric);
        let name = if has_extension { &line[..line.len() - 4] } else { line };
        words.extend(name.split(|c: char| !c.is_ascii_alphanumeric()).filter(|word| !word.is_empty()).map(str::to_ascii_uppercase));
    }
    words
}

/// How many of `words` start with each `len`-character prefix. A word too
/// short to have one counts for every prefix of a longer word it starts.
fn listfile_counts(words: &HashSet<String>, len: usize) -> HashMap<String, i64> {
    let mut counts: HashMap<String, i64> = HashMap::new();
    for word in words.iter().filter(|word| word.len() >= len) {
        *counts.entry(word[..len].to_string()).or_default() += 1;
    }
    for word in words.iter().filter(|word| word.len() < len) {
        for (prefix, count) in counts.iter_mut() {
            if prefix.starts_with(word.as_str()) {
                *count += 1;
            }
        }
    }
    counts
}

/// How many of `dictionary`'s distinct words start with each
/// `len`-character prefix, uppercased. Only all-lowercase ASCII words count,
/// which leaves out most names and abbreviations.
fn dictionary_counts(dictionary: &str, len: usize) -> HashMap<String, i64> {
    let words: HashSet<&str> =
        dictionary.lines().map(str::trim).filter(|word| word.len() >= len && word.bytes().all(|b| b.is_ascii_lowercase())).collect();
    let mut counts: HashMap<String, i64> = HashMap::new();
    for word in words {
        *counts.entry(word[..len].to_ascii_uppercase()).or_default() += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(prefixes: &LikelyPrefixes, len: usize) -> Vec<(String, i64, Source)> {
        prefixes.ranking().get(len).iter().map(|p| (p.prefix.clone(), p.count, p.source)).collect()
    }

    fn row(prefix: &str, count: i64, source: Source) -> (String, i64, Source) {
        (prefix.to_string(), count, source)
    }

    #[test]
    fn listfile_words_drop_three_character_extensions_and_split_on_symbols() {
        let words = listfile_words("rez\\GameMenu.bin\r\nsound\\Zerg\\Zov00.wav\nMUSIC\\title.mpeg\n(1)Enslavers01.scm\nmaps\\X.TXT\n\n");
        let mut words: Vec<&str> = words.iter().map(String::as_str).collect();
        words.sort();
        assert_eq!(words, ["1", "ENSLAVERS01", "GAMEMENU", "MAPS", "MPEG", "MUSIC", "REZ", "SOUND", "TITLE", "X", "ZERG", "ZOV00"]);
    }

    #[test]
    fn listfile_counts_each_distinct_word_once_and_short_words_for_every_prefix_they_start() {
        let words = listfile_words("FOO\\FOR.TXT\nFOR\\FOX\nFO\\FAB\nF\\BAR");
        let counts = listfile_counts(&words, 3);
        // FO and F count towards every prefix they start; F has none of its own.
        assert_eq!(counts.get("FOR"), Some(&3));
        assert_eq!(counts.get("FOO"), Some(&3));
        assert_eq!(counts.get("FAB"), Some(&2));
        assert_eq!(counts.get("BAR"), Some(&1));
        assert_eq!(counts.get("F"), None);
    }

    #[test]
    fn dictionary_counts_only_lowercase_ascii_words_long_enough() {
        let counts = dictionary_counts("fox\nfog\nFoxes\nfo\nfox's\nfoxy\nfox\n", 3);
        assert_eq!(counts, HashMap::from([("FOX".to_string(), 2), ("FOG".to_string(), 1)]));
    }

    #[test]
    fn the_listfiles_prefixes_come_first_then_the_dictionarys_it_lacks() {
        let prefixes = LikelyPrefixes::from_texts("FOX\nFOXES\nFOG\nBAT", "fox\nfoxy\nfoxes\ncat\ncats\ncatty\nbat\n");
        assert_eq!(
            ranked(&prefixes, 2),
            [row("FO", 3, Source::Listfile), row("BA", 1, Source::Listfile), row("CA", 3, Source::Dictionary)],
            "FO is the listfile's already, so the dictionary only adds CA"
        );
    }

    #[test]
    fn matches_count_as_listfile_words() {
        let prefixes = LikelyPrefixes::from_texts("FOX\nBAT", "cat\ncats\n");
        prefixes.include_matches(&["REZ\\CATAPULT.TXT".to_string(), "REZ\\CATS.WAV".to_string(), "FOXES.BIN".to_string()]);
        assert_eq!(
            ranked(&prefixes, 2),
            [row("CA", 2, Source::Listfile), row("FO", 2, Source::Listfile), row("BA", 1, Source::Listfile), row("RE", 1, Source::Listfile)],
            "REZ counts once, and CA is a filename's now rather than the dictionary's"
        );
    }

    #[test]
    fn prefix_len_grows_from_two_at_eleven_characters_to_three() {
        assert_eq!([10, 11, 12, 13, 16].map(prefix_len_for), [None, Some(2), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn the_built_in_listfile_ranks_something() {
        let ranking = LikelyPrefixes::from_texts(LISTFILE, "").ranking();
        assert!(ranking.get(3).len() > 100);
        assert!(ranking.get(3).windows(2).all(|w| w[0].count >= w[1].count));
    }
}
