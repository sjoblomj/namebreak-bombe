#!/usr/bin/env python3
"""Dictionary targets end to end: the real coordinator server
(../coordinator) and this client in coordinator mode, talking over HTTP on
localhost.

Phase 1 - a quit part way: a big dictionary target (english-1, two words),
whose second range the client is still searching when it's told to quit
(SIGTERM). Its quit report has to carry how far it got, as a candidate
number - the server splits the range there - and the basename it found
matching the target's key, which the server keeps.

Phase 2 - searching to the end: a dictionary target with a planted name
(english-1 and a word list of the test's own, which the client downloads
once and then keeps) whose basename other candidates, in other directories,
have before it, another one searched through without a find, one with a
key but no basenames to send, and an alphabet target, all in one run. The finds have to be reported, every
basename matching a key sent - none ever written to a file - and the
dashboard has to show it all.

Phase 3 - the server goes away: with a client that heartbeats every second
(--client-heartbeat-1s), the server is stopped while the client searches a
range, before it finds a basename. The heartbeats fail, and the basename
waits - until the server is back, when the next heartbeat sends it, with
how far the search has got.

Phase 4 - a completion that doesn't get through: the server is stopped
while the client searches a range, which it finishes - with a basename
found - while the server's away. The completion is tried again until the
server is back, and the basename goes with it.

Phase 5 - the range taken away: the server is stopped while the client
searches a range, and, while it's away, the range is given to someone else.
Back, the server answers the client's next heartbeat, which carries the
basename found meanwhile, with a 409 - having kept the basename all the
same.

Phase 6 - more basenames than a report carries, with a client that sends at
most one a report and heartbeats every ten seconds (--client-one-basename),
and pairs of candidates whose basenames share a key: a quit carries the
first of two, and doesn't say the search got past the second - which the
rest of the range, handed out again, finds; a range finished with two waiting
sends the first in a heartbeat before its completion carries the second; and
after the server's been away, the second goes right after the first, rather
than a heartbeat later.

    coordinator_dictionary_test.py --client <namebreak> --client-heartbeat-1s <namebreak built so>
                                   --client-one-basename <namebreak built so> --coordinator <coordinator dir> [--backend cpu]

Builds the server with cargo first. Runs in a directory of its own under
the current one.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

ADMIN_TOKEN = "e2e-admin"


def crypt_table():
    seed = 0x00100001
    table = [0] * 0x500
    for i in range(0x100):
        index = i
        for _ in range(5):
            seed = (seed * 125 + 3) % 0x2AAAAB
            high = (seed & 0xFFFF) << 16
            seed = (seed * 125 + 3) % 0x2AAAAB
            table[index] = high | (seed & 0xFFFF)
            index += 0x100
    return table


TABLE = crypt_table()


def mpq_hash(text, offset):
    """Storm's hash of `text` (already uppercase, with '\\'): 0x100 is hash
    A, 0x200 hash B, 0x300 the one an encryption key is made with."""
    seed1, seed2 = 0x7FED7FED, 0xEEEEEEEE
    for ch in text.encode("ascii"):
        seed1 = (TABLE[offset + ch] ^ (seed1 + seed2)) & 0xFFFFFFFF
        seed2 = (ch + seed1 + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
    return seed1


def hex32(value):
    return "0x%08X" % value


def normalized_words(path):
    """A word list read as the client and server read one: sorted, without
    duplicates."""
    words = set()
    with open(path, "rb") as f:
        for line in f.read().split(b"\n"):
            line = line[:-1] if line.endswith(b"\r") else line
            word = line.strip(b" \t")
            if not word or word.startswith(b"#") or any(b < 0x20 or b > 0x7E for b in word):
                continue
            words.add(word.decode("ascii").upper().replace("/", "\\"))
    return sorted(words)


class Failed(Exception):
    pass


def check(ok, what):
    if not ok:
        raise Failed(what)
    print("  ok: " + what, flush=True)


class Server:
    # Every server started, to stop whatever happens.
    started = []

    def __init__(self, binary, workdir, port=None, log_name="server.log"):
        if port is None:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        self.binary, self.workdir, self.port = binary, workdir, port
        self.url = "http://127.0.0.1:%d" % self.port
        env = dict(os.environ)
        env.update({
            "ADMIN_TOKEN": ADMIN_TOKEN,
            "DATABASE_URL": "sqlite://" + os.path.join(workdir, "coordinator.db"),
            "BIND_ADDR": "127.0.0.1:%d" % self.port,
            "CANARY_PROBABILITY": "0",
            # Every range at least 1e10 candidates: phase 1's two-word range
            # takes the cpu backend about a minute - long enough to quit it
            # part way - and phase 2's targets are a range per window.
            "MIN_CHUNK_CANDIDATES": "10000000000",
            "RUST_LOG": "warn",
        })
        self.log = open(os.path.join(workdir, log_name), "w")
        self.process = subprocess.Popen([binary], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        Server.started.append(self)
        for _ in range(100):
            try:
                self.get("/api/v1/status")
                return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        raise Failed("the server didn't start - see server.log")

    def request(self, method, path, body=None, admin=False, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if admin:
            req.add_header("X-Admin-Token", ADMIN_TOKEN)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as response:
            text = response.read().decode()
            return json.loads(text) if text and response.headers.get("Content-Type", "").startswith("application/json") else text

    def get(self, path):
        return self.request("GET", path)

    def admin(self, method, path, body=None, raw=None):
        return self.request(method, path, body=body, admin=True, raw=raw)

    def target(self, target_id):
        return next(t for t in self.get("/api/v1/dashboard")["targets"] if t["id"] == target_id)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()

    def restarted(self):
        """The same server again - same port, same database."""
        return Server(self.binary, self.workdir, self.port, "server-restarted.log")


class Client:
    # Every client started, to stop whatever happens.
    started = []

    def __init__(self, binary, workdir, server, backend, name):
        self.matches = os.path.join(workdir, "matches")
        config = os.path.join(workdir, "config.conf")
        with open(config, "w") as f:
            f.write("mode = coordinator\n")
            f.write("matches_dir = %s\n" % self.matches)
            f.write("backend = %s\n" % backend)
            f.write("check_for_updates = false\n")
            f.write("[coordinator]\n")
            f.write("server_url = %s\n" % server.url)
            f.write("username = e2e\n")
            f.write("hostname = test\n")
            f.write("poll_interval_secs = 1\n")
        self.output_path = os.path.join(workdir, name + ".log")
        self.output = open(self.output_path, "w")
        self.process = subprocess.Popen([binary, "--config", config], stdin=subprocess.DEVNULL, stdout=self.output, stderr=subprocess.STDOUT,
                                        cwd=workdir)
        Client.started.append(self)

    def quit(self):
        """SIGTERM: the client tells the server how far it got, and exits."""
        self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(timeout=60)
        self.output.close()
        return code

    def text(self):
        if not self.output.closed:
            self.output.flush()
        with open(self.output_path) as f:
            return f.read()

    def alive(self):
        return self.process.poll() is None


def basenames(server, target_id):
    """The basenames clients sent for a target. Besides the one each test
    target's key is made from, any candidate can match a 32-bit key by
    chance: about one in 2^32 does - so the checks look for it among them."""
    return server.get("/api/v1/targets/%d/basenames" % target_id).split()


def found(client, basename):
    """Whether `client` has found `basename` - it says so as it does."""
    return "BASENAME MATCH: " + basename + " " in client.text()


def basename_files(client):
    """Files a client keeps basenames in - which it never should."""
    return [f for f in os.listdir(client.matches) if "basename" in f] if os.path.isdir(client.matches) else []


def wait_for(what, condition, client, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        if not client.alive():
            raise Failed("the client stopped while waiting for " + what + ":\n" + client.text()[-3000:])
        time.sleep(0.2)
    raise Failed("timed out waiting for " + what + ":\n" + client.text()[-3000:])


def phase1(server, client_binary, workdir, backend, english):
    print("--- phase 1: quitting part way through a range ---", flush=True)
    # Basename key: the 4th two-word candidate, early in the second range
    # (the first is the one-word candidates).
    key_basename = english[0] + english[3] + ".WAV"
    big = server.admin("POST", "/api/v1/admin/targets", {
        "name": "big", "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x00000001", "hash_b_hex": "0x00000002",
        "dictionary": {"word_lists": ["english-1"], "separators": ["", "_", "-", " "], "max_words": 2},
        "encryption_key_hex": hex32(mpq_hash(key_basename, 0x300)), "send_basenames": True, "priority": 10,
    })["target_id"]

    client = Client(client_binary, workdir, server, backend, "client1")
    wait_for("the second range's basename", lambda: found(client, key_basename), client)
    check(any(r["status"] == "in_progress" and r["candidate_len"] == 2 for r in server.target(big)["ranges"]),
          "the basename found part way through the two-word range")
    check(client.quit() == 0, "the client quits when told to")
    output = client.text()
    check("told the coordinator it's searched every candidate numbered below" in output, "... telling the server how far it got")
    check(basename_files(client) == [], "... and no basename kept in a file")
    check(key_basename in basenames(server, big), "the server has the basename")

    t = server.target(big)
    statuses = [(r["status"], r["candidate_len"]) for r in t["ranges"]]
    check(statuses == [("completed", 1), ("completed", 2), ("pending", 2)],
          "the one-word range completed, the two-word one split where the client got to: %s" % statuses)
    searched, rest = t["ranges"][1], t["ranges"][2]
    check(int(searched["candidate_count"]) > 3 and int(rest["candidate_count"]) > 0,
          "... %s candidates searched, %s handed back" % (searched["candidate_count"], rest["candidate_count"]))
    shown = [b for b in t["dictionary"]["basenames"] if b["basename"] == key_basename]
    check(t["dictionary"]["basename_count"] >= 1 and len(shown) == 1 and shown[0]["reported_by"] == "e2e@test",
          "the dashboard shows the basename, and who sent it")
    server.admin("PATCH", "/api/v1/admin/targets/%d" % big, {"status": "paused"})
    return big


def phase2(server, client_binary, workdir, backend, english, own_words):
    print("--- phase 2: searching dictionary and alphabet targets to the end ---", flush=True)
    words = sorted(set(english) | set(own_words))
    # A planted name: one of the last few first words as a directory, then a
    # word only the test's own list has. The bounds take in the last 20
    # first words - so the planted name's basename is first found, and sent,
    # in the directories of the ten before it.
    first = words[-10]
    early = "QUARTZBLASTER" + ".WAV"
    planted = "MUSIC\\" + first + "\\" + early
    found = server.admin("POST", "/api/v1/admin/targets", {
        "name": "found", "prefix": "music\\", "suffix": ".wav",
        "hash_a_hex": hex32(mpq_hash(planted, 0x100)), "hash_b_hex": hex32(mpq_hash(planted, 0x200)),
        "dictionary": {"word_lists": ["english-1", "own"], "separators": ["", "_", "\\"], "max_words": 2},
        "lower_bound": "MUSIC\\" + words[-20], "upper_bound": "MUSIC\\" + words[-1] + "~",
        "encryption_key_hex": hex32(mpq_hash(early, 0x300)), "send_basenames": True, "priority": 3,
    })["target_id"]
    # Searched through: the test's words alone, up to three of them.
    three = own_words[1] + "-" + own_words[2] + "-" + own_words[0] + ".OGG"
    through = server.admin("POST", "/api/v1/admin/targets", {
        "name": "through", "prefix": "sound/", "suffix": ".ogg", "hash_a_hex": "0x00000003", "hash_b_hex": "0x00000004",
        "dictionary": {"word_lists": ["own"], "separators": ["", "-"], "max_words": 3},
        "encryption_key_hex": hex32(mpq_hash(three, 0x300)), "send_basenames": True, "priority": 2,
    })["target_id"]
    # A key, but no basenames to send: found all the same, the key used to
    # compare only the candidates whose basename matches it to the hashes.
    quiet_basename = own_words[3] + "_" + own_words[1] + ".WAV"
    quiet_name = "VOICE\\" + quiet_basename
    quiet = server.admin("POST", "/api/v1/admin/targets", {
        "name": "quiet", "prefix": "voice/", "suffix": ".wav",
        "hash_a_hex": hex32(mpq_hash(quiet_name, 0x100)), "hash_b_hex": hex32(mpq_hash(quiet_name, 0x200)),
        "dictionary": {"word_lists": ["own"], "separators": ["", "_"], "max_words": 2},
        "encryption_key_hex": hex32(mpq_hash(quiet_basename, 0x300)), "priority": 2,
    })["target_id"]
    alphabet_name = "TEST\\BCA.TXT"
    alphabet = server.admin("POST", "/api/v1/admin/targets", {
        "name": "alphabet", "prefix": "TEST\\", "suffix": ".TXT", "alphabet": "ABC", "lower_bound": "A", "upper_bound": "CCC",
        "hash_a_hex": hex32(mpq_hash(alphabet_name, 0x100)), "hash_b_hex": hex32(mpq_hash(alphabet_name, 0x200)), "priority": 1,
    })["target_id"]

    client = Client(client_binary, workdir, server, backend, "client2")

    def done():
        targets = {t["id"]: t for t in server.get("/api/v1/dashboard")["targets"]}
        t = targets[through]
        return (targets[found]["status"] == "solved" and targets[alphabet]["status"] == "solved" and targets[quiet]["status"] == "solved"
                and t["cursor_candidate"] == "" and all(r["status"] == "completed" for r in t["ranges"]))

    wait_for("every target done", done, client)
    check(client.quit() == 0, "the client quits when told to")
    output = client.text()

    t = server.target(found)
    check(t["status"] == "solved" and t["found_filename"] == planted, "the planted name found: " + str(t["found_filename"]))
    check(early in basenames(server, found), "... and its basename, found before it, sent")
    t = server.target(through)
    check([r["candidate_len"] for r in t["ranges"]] == [1, 2, 3], "every word count searched, a range each")
    check(three in basenames(server, through), "the three-word candidate's basename sent")
    t = server.target(quiet)
    check(t["status"] == "solved" and t["found_filename"] == quiet_name, "a target with a key, no basenames to send, found: " + str(t["found_filename"]))
    check(basenames(server, quiet) == [] and "matching basenames aren't recorded" in output, "... the key used, none of its basenames sent")
    check(server.target(alphabet)["found_filename"] == alphabet_name, "the alphabet target's name found in the same run")

    check(output.count("downloading word list own") == 1 and "downloading word list english-1" not in output,
          "the test's word list downloaded once, english-1 not at all")
    cache = os.path.join(client.matches, "word-lists")
    check(sorted(os.listdir(cache)) == ["own.txt"], "... and kept: %s" % sorted(os.listdir(cache)))
    check(basename_files(client) == [], "no basename written to a file: %s" % basename_files(client))

    d = server.target(found)["dictionary"]
    check(d["word_lists"] == ["english-1", "own"] and d["separators"] == ["", "_", "\\"] and d["min_words"] == 1 and d["max_words"] == 2,
          "the dashboard shows what the candidates are made of")
    volunteers = server.get("/api/v1/dashboard")["volunteers"]
    check([(v["username"], v["found"]) for v in volunteers] == [("e2e", 3)], "the volunteer credited with all three finds")


def phase3(server, client_binary, workdir, backend, english):
    print("--- phase 3: the server goes away, and comes back ---", flush=True)
    # A basename about 8,000 first words into the two-word candidates: some
    # seconds into the range, on the cpu backend - after the server's gone.
    separators = ["", "_", "-", " "]
    key_basename = english[8000] + english[0] + ".WAV"
    outage = server.admin("POST", "/api/v1/admin/targets", {
        "name": "outage", "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x00000005", "hash_b_hex": "0x00000006",
        "dictionary": {"word_lists": ["english-1"], "separators": separators, "max_words": 2},
        "encryption_key_hex": hex32(mpq_hash(key_basename, 0x300)), "send_basenames": True, "priority": 20,
    })["target_id"]
    key_number = len(english) + 8000 * len(separators) * len(english)

    client = Client(client_binary, workdir, server, backend, "client3")
    wait_for("the two-word range", lambda: any(r["status"] == "in_progress" and r["candidate_len"] == 2 for r in server.target(outage)["ranges"]),
             client)
    range_id = next(r["id"] for r in server.target(outage)["ranges"] if r["candidate_len"] == 2)
    server.stop()

    wait_for("the basename, while the server's away", lambda: found(client, key_basename), client)
    failures_then = client.text().count("heartbeat failed")
    time.sleep(3)
    check(client.text().count("heartbeat failed") >= failures_then + 2, "the heartbeats fail while the server's away")

    server = server.restarted()
    wait_for("the basename to be sent", lambda: key_basename in basenames(server, outage), client, timeout=60)
    check(key_basename in basenames(server, outage), "the server back: the next heartbeat sent it: %s" % basenames(server, outage))
    # The two-word range starts at the first two-word candidate, numbered
    # after the one-word ones.
    two_words = next(r for r in server.target(outage)["ranges"] if r["id"] == range_id)
    past_key = (key_number - len(english) + 1) / int(two_words["candidate_count"]) * 100
    check(two_words["status"] == "in_progress" and two_words["progress_percent"] is not None and two_words["progress_percent"] >= past_key,
          "... with how far the search got: %s, %.2f%% of the range - past the basename's %.2f%%"
          % (two_words["progress_candidate"], two_words["progress_percent"] or 0, past_key))
    check(client.quit() == 0, "the client quits when told to")
    server.admin("PATCH", "/api/v1/admin/targets/%d" % outage, {"status": "paused"})
    return server


def two_word_range(server, target_id):
    return next((r for r in server.target(target_id)["ranges"] if r["candidate_len"] == 2), None)


def phase4(server, client_binary, workdir, backend, english):
    print("--- phase 4: a completion that doesn't get through ---", flush=True)
    # About a billion two-word candidates: the first words from the
    # 10,000th to the 14,000th - some seconds on the cpu backend - with the
    # basename near the end.
    separators = ["", "_", "-", " "]
    key_basename = english[13500] + english[0] + ".WAV"
    late = server.admin("POST", "/api/v1/admin/targets", {
        "name": "late", "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x00000007", "hash_b_hex": "0x00000008",
        "dictionary": {"word_lists": ["english-1"], "separators": separators, "max_words": 2},
        "lower_bound": "MUSIC\\" + english[10000], "upper_bound": "MUSIC\\" + english[14000] + "~",
        "encryption_key_hex": hex32(mpq_hash(key_basename, 0x300)), "send_basenames": True, "priority": 30,
    })["target_id"]
    client = Client(client_binary, workdir, server, backend, "client4")
    wait_for("the two-word range", lambda: (two_word_range(server, late) or {}).get("status") == "in_progress", client)
    range_id = two_word_range(server, late)["id"]
    server.stop()

    wait_for("a completion that doesn't get through", lambda: "failed to report completion" in client.text(), client)
    check(found(client, key_basename), "the range searched while the server's away, its basename found")
    server = server.restarted()
    wait_for("the completion to get through", lambda: key_basename in basenames(server, late), client, timeout=120)
    check(key_basename in basenames(server, late), "the server back: the completion got through, with the basename")
    done = next(r for r in server.target(late)["ranges"] if r["id"] == range_id)
    check(done["status"] == "completed", "... and the range completed")
    check(client.quit() == 0, "the client quits when told to")
    server.admin("PATCH", "/api/v1/admin/targets/%d" % late, {"status": "paused"})
    return server


def phase5(server, client_binary, workdir, backend, english):
    print("--- phase 5: the range taken away ---", flush=True)
    # About three billion two-word candidates, the basename a sixth of the
    # way in - found well before the range is done.
    separators = ["", "_", "-", " "]
    key_basename = english[21000] + english[0] + ".WAV"
    taken = server.admin("POST", "/api/v1/admin/targets", {
        "name": "taken", "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x00000009", "hash_b_hex": "0x0000000A",
        "dictionary": {"word_lists": ["english-1"], "separators": separators, "max_words": 2},
        "lower_bound": "MUSIC\\" + english[20000], "upper_bound": "MUSIC\\" + english[32000] + "~",
        "encryption_key_hex": hex32(mpq_hash(key_basename, 0x300)), "send_basenames": True, "priority": 40,
    })["target_id"]
    other = server.request("POST", "/api/v1/register", {"username": "other", "hostname": "elsewhere", "backend": "cpu", "protocol_version": "1.5.0"})
    client = Client(client_binary, workdir, server, backend, "client5")
    wait_for("the two-word range", lambda: (two_word_range(server, taken) or {}).get("status") == "in_progress", client)
    range_id = two_word_range(server, taken)["id"]
    server.stop()

    wait_for("the basename, while the server's away", lambda: found(client, key_basename), client)
    with sqlite3.connect(os.path.join(workdir, "coordinator.db")) as db:
        db.execute("UPDATE ranges SET assigned_user_id = ?, last_assigned_user_id = ? WHERE id = ?", (other["user_id"], other["user_id"], range_id))
    server = server.restarted()
    wait_for("the client to find the range taken", lambda: "no longer assigned to us" in client.text(), client, timeout=60)
    check(key_basename in basenames(server, taken), "the heartbeat got a 409 - and the server kept its basename")
    reported = [b for b in server.target(taken)["dictionary"]["basenames"] if b["basename"] == key_basename]
    check(reported and reported[0]["reported_by"] == "e2e@test", "... as this client's")
    check(next(r for r in server.target(taken)["ranges"] if r["id"] == range_id)["worker"] == "other@elsewhere", "the range stays the other's")
    check(client.quit() == 0, "the client quits when told to")
    return server


def phase6(server, client_binary, workdir, backend, english):
    print("--- phase 6: more basenames than a report carries ---", flush=True)
    separators = ["", "_", "-", " "]
    # Two pairs of english-1's two-word candidates whose basenames share a key
    # - found by searching for them once, as english-1 never changes - each
    # early in its range.
    extolled, extolled_key = ["EXTOLLED_CORPSE.WAV", "EXTOLLED_HISTAMINE.WAV"], 0x91635E69
    pearliest, pearliest_key = ["PEARLIEST_BALLERINA.WAV", "PEARLIEST_MULLED.WAV"], 0x659422A8
    check(all(mpq_hash(b, 0x300) == extolled_key for b in extolled) and all(mpq_hash(b, 0x300) == pearliest_key for b in pearliest),
          "each pair's basenames share a key")

    def two_word_target(name, first_word, key, priority, first_words=12000):
        return server.admin("POST", "/api/v1/admin/targets", {
            "name": name, "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x0000000B", "hash_b_hex": "0x0000000C",
            "dictionary": {"word_lists": ["english-1"], "separators": separators, "max_words": 2},
            "lower_bound": "MUSIC\\" + first_word, "upper_bound": "MUSIC\\" + english[english.index(first_word) + first_words] + "~",
            "encryption_key_hex": hex32(key), "send_basenames": True, "priority": priority,
        })["target_id"]

    # A quit with both waiting carries the first, and says the search got no
    # further than where the second was found.
    capped = two_word_target("capped", "EXTOLLED", extolled_key, 50)
    client = Client(client_binary, workdir, server, backend, "client6")
    wait_for("both basenames found", lambda: all(found(client, b) for b in extolled), client)
    check(client.quit() == 0, "the client quits with both waiting - before its first heartbeat")
    check(basenames(server, capped) == extolled[:1], "the quit carried the first, as a report has room for one: %s" % basenames(server, capped))

    # The rest of the range, handed out again, has the second.
    client = Client(client_binary, workdir, server, backend, "client7")
    wait_for("the second basename, found again", lambda: extolled[1] in basenames(server, capped), client, timeout=120)
    check(basenames(server, capped)[:2] == extolled, "the rest of the range, handed out again, found the second - the quit hadn't said it was searched")
    check(client.quit() == 0, "the client quits when told to")
    server.admin("PATCH", "/api/v1/admin/targets/%d" % capped, {"status": "paused"})

    # A range finished with both waiting: the first goes in a heartbeat, then
    # the completion carries the second.
    server.admin("PUT", "/api/v1/admin/word-lists/pair", raw=b"pearliest_ballerina\npearliest_mulled\nsome\nother\nwords\n")
    flush = server.admin("POST", "/api/v1/admin/targets", {
        "name": "flush", "prefix": "music\\", "suffix": ".wav", "hash_a_hex": "0x0000000D", "hash_b_hex": "0x0000000E",
        "dictionary": {"word_lists": ["pair"], "separators": [""], "max_words": 1},
        "encryption_key_hex": hex32(pearliest_key), "send_basenames": True, "priority": 60,
    })["target_id"]
    client = Client(client_binary, workdir, server, backend, "client8")
    wait_for("the range completed", lambda: [r["status"] for r in server.target(flush)["ranges"]] == ["completed"], client, timeout=60)
    check(basenames(server, flush) == pearliest, "both delivered: %s" % basenames(server, flush))
    check(client.quit() == 0, "the client quits when told to")
    check("sending the 2 basenames found before completing it" in client.text(), "... the first in a heartbeat before the completion")

    # Both found while the server's away: once it's back, the second goes
    # right after the first, rather than a heartbeat (ten seconds) later -
    # in a range long enough (six billion candidates, half a minute on the
    # cpu backend) that its completion can't carry it first.
    drain = two_word_target("drain", "PEARLIEST", pearliest_key, 70, first_words=23000)
    client = Client(client_binary, workdir, server, backend, "client9")
    wait_for("the two-word range", lambda: (two_word_range(server, drain) or {}).get("status") == "in_progress", client)
    server.stop()
    wait_for("both basenames found, while the server's away", lambda: all(found(client, b) for b in pearliest), client)
    server = server.restarted()
    wait_for("the first basename sent", lambda: basenames(server, drain)[:1] == pearliest[:1], client, timeout=60)
    first_at = time.time()
    wait_for("the second basename sent", lambda: basenames(server, drain)[:2] == pearliest, client, timeout=60)
    check(time.time() - first_at < 5, "the second sent %.1f s after the first - not a heartbeat later" % (time.time() - first_at))
    check(client.quit() == 0, "the client quits when told to")
    server.admin("PATCH", "/api/v1/admin/targets/%d" % drain, {"status": "paused"})
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--client-heartbeat-1s", required=True)
    parser.add_argument("--client-one-basename", required=True)
    parser.add_argument("--coordinator", required=True)
    # The cpu backend: phase 1 needs a range to take long enough to quit it
    # part way, which on a GPU it doesn't.
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    client_binary = os.path.abspath(args.client)
    coordinator = os.path.abspath(args.coordinator)
    english_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "english-1.txt")

    if shutil.which("cargo") is None:
        print("cargo isn't installed - can't build the server")
        return 77
    build = subprocess.run(["cargo", "build", "--quiet", "-p", "namebreak-server"], cwd=coordinator)
    if build.returncode != 0:
        print("building the server failed")
        return 1
    server_binary = os.path.join(coordinator, "target", "debug", "namebreak-server")

    workdir = os.path.abspath("coordinator_dictionary_test")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    server = Server(server_binary, workdir)
    try:
        english = normalized_words(english_path)
        # english-1 is built into the server as into the client: never
        # uploaded.
        try:
            with open(english_path, "rb") as f:
                server.admin("PUT", "/api/v1/admin/word-lists/english-1", raw=f.read())
            raise Failed("the server took an upload of english-1")
        except urllib.error.HTTPError as refused:
            check(refused.code == 400, "english-1 is the server's own already")
        own_words = ["QUARTZBLASTER", "ZEALOT", "HYDRA", "MUTALISK"]
        server.admin("PUT", "/api/v1/admin/word-lists/own", raw=("\n".join(w.lower() for w in own_words) + "\n").encode())
        phase1(server, client_binary, workdir, args.backend, english)
        phase2(server, client_binary, workdir, args.backend, english, own_words)
        server = phase3(server, os.path.abspath(args.client_heartbeat_1s), workdir, args.backend, english)
        server = phase4(server, client_binary, workdir, args.backend, english)
        server = phase5(server, os.path.abspath(args.client_heartbeat_1s), workdir, args.backend, english)
        server = phase6(server, os.path.abspath(args.client_one_basename), workdir, args.backend, english)
    except Failed as failure:
        print("FAILED: %s" % failure, flush=True)
        return 1
    finally:
        for client in Client.started:
            if client.alive():
                client.process.kill()
                client.process.wait()
        for started in Server.started:
            started.stop()
    print("ALL CHECKS PASSED")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
