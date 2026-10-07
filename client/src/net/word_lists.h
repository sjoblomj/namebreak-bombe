#ifndef NAMEBREAK_NET_WORD_LISTS_H
#define NAMEBREAK_NET_WORD_LISTS_H

#include <functional>
#include <string>
#include <vector>

#include "net/protocol.h"

// The word lists a coordinator's dictionary target is searched with (see
// ClaimResponse::dictionary): the server stores them, and a client keeps a
// copy of each it has needed, in a directory of its own.

// The directory word lists are kept in, under the matches directory.
inline constexpr const char* kWordListCacheDirName = "word-lists";

// Gets a stored word list's text from the server -
// CoordinatorClient::downloadWordList, or a stand-in in the tests.
using WordListDownloader = std::function<bool(const std::string& name, std::string& text, std::string& error)>;

// Whether `name` is a name a word list can have: 1 to 64 letters, digits,
// '.', '-' and '_', starting with a letter or digit - as the server only
// stores, so it's safe as a file's name in the cache.
bool isValidWordListName(const std::string& name);

// The words a dictionary claim's candidates are made of: every word list it
// names, each of which has to have the checksum the claim gives it, merged
// (sorted, without duplicates) - which then has to have the claim's
// words_checksum. A dictionary compiled in (english-1) is used as it is.
// Any other list is read from `<cacheDir>/<name>.txt`, or - if it isn't
// there, or the copy there has another checksum - downloaded with
// `download` and saved there. False, with `error` set, if a list can't be
// had or doesn't check out; `downloaded` says which lists were downloaded.
bool resolveClaimWords(const ClaimResponse& claim, const std::string& cacheDir, const WordListDownloader& download,
                       std::vector<std::string>& words, std::vector<std::string>& downloaded, std::string& error);

#endif // NAMEBREAK_NET_WORD_LISTS_H
