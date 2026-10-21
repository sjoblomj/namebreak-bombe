#!/usr/bin/env python3
"""False matches end to end: the real coordinator server (../coordinator) and
this client in coordinator mode, talking over HTTP on localhost.

A name that matches both of a target's hashes but not the rest of what's
known about its file - its encryption key, or where its entry is in the
archive's hash table - is a false match, not the file: two 32-bit hashes
match by chance about once in 2^64 names, and REZ\\SFZSSP050WSRL.WAV did.

Phase 1 - an older client's find: a client of protocol 1.6, played here
over plain HTTP, is told nothing about the key, takes the planted name for
the file and completes its range with it. The server checks it itself: the
target stays unsolved, the name is kept as a false match, and the rest of the
range, past the name, is handed out again.

Phase 2 - this client: three targets over a two-letter alphabet, each with
a planted name - the first with a key that isn't its name's, the second with
hash table slots its name's lookup doesn't start at, the third with both
right. The client is told them with its claims, reports the first two
planted names as false matches and searches on past them, every range of
both targets searched to the end - and finds the third, whose name passes.
The dashboard shows the false matches, and the client said what it found.

    coordinator_false_match_test.py --client <namebreak> --coordinator <coordinator dir> [--backend cpu]

Builds the server with cargo first. Runs in a directory of its own under the
current one.
"""

import argparse
import json
import os
import shutil
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
    """Storm's hash of `text` (already uppercase, with '\\'): 0x000 is the one
    the hash table is indexed by, 0x100 hash A, 0x200 hash B, 0x300 the one
    an encryption key is made with."""
    seed1, seed2 = 0x7FED7FED, 0xEEEEEEEE
    for ch in text.encode("ascii"):
        seed1 = (TABLE[offset + ch] ^ (seed1 + seed2)) & 0xFFFFFFFF
        seed2 = (ch + seed1 + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
    return seed1


def hex32(value):
    return "0x%08X" % value


def key_of(filename):
    return mpq_hash(filename.rsplit("\\", 1)[-1], 0x300)


def slot_of(filename, table_size):
    return mpq_hash(filename, 0x000) & (table_size - 1)


class Failed(Exception):
    pass


def check(ok, what):
    if not ok:
        raise Failed(what)
    print("  ok: " + what, flush=True)


class Server:
    def __init__(self, binary, workdir):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.workdir = workdir
        self.url = "http://127.0.0.1:%d" % port
        env = dict(os.environ)
        env.update({
            "ADMIN_TOKEN": ADMIN_TOKEN,
            "DATABASE_URL": "sqlite://" + os.path.join(workdir, "coordinator.db"),
            "BIND_ADDR": "127.0.0.1:%d" % port,
            "CANARY_PROBABILITY": "0",
            # A whole length of a target a range, however fast a client is.
            "MIN_CHUNK_CANDIDATES": "10000000000",
            "RUST_LOG": "warn",
        })
        self.log = open(os.path.join(workdir, "server.log"), "w")
        self.process = subprocess.Popen([binary], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                self.request("GET", "/api/v1/status")
                return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        raise Failed("the server didn't start - see server.log")

    def request(self, method, path, body=None, admin=False, token=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if admin:
            req.add_header("X-Admin-Token", ADMIN_TOKEN)
        if token:
            req.add_header("Authorization", "Bearer " + token)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as response:
            text = response.read().decode()
            return json.loads(text) if text and response.headers.get("Content-Type", "").startswith("application/json") else text

    def create(self, name, planted, **extra):
        """An alphabet target for REZ\\<planted>.WAV: the eight-letter
        candidates of "AB" and every longer one up to sixteen - 130,816 of
        them, a moment's work."""
        filename = "REZ\\" + planted + ".WAV"
        body = {"name": name, "prefix": "REZ\\", "suffix": ".WAV", "alphabet": "AB", "lower_bound": "AAAAAAAA", "upper_bound": "BBBBBBBB",
                "start_len": 8, "hash_a_hex": hex32(mpq_hash(filename, 0x100)), "hash_b_hex": hex32(mpq_hash(filename, 0x200))}
        body.update(extra)
        return self.request("POST", "/api/v1/admin/targets", body, admin=True)["target_id"]

    def target(self, target_id):
        return next(t for t in self.request("GET", "/api/v1/dashboard")["targets"] if t["id"] == target_id)

    def rows(self, query, *params):
        with sqlite3.connect(os.path.join(self.workdir, "coordinator.db")) as db:
            return db.execute(query, params).fetchall()

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()


def phase1(server):
    """An older client's false find."""
    print("--- phase 1: an older client's find ---", flush=True)
    planted = "ABBABAAB"
    filename = "REZ\\" + planted + ".WAV"
    target = server.create("older-client", planted, encryption_key_hex=hex32(key_of(filename) ^ 1))
    old = server.request("POST", "/api/v1/register", {"username": "old", "hostname": "test", "backend": "cpu", "protocol_version": "1.6.0"})
    claim = server.request("POST", "/api/v1/claim", token=old["token"])
    check(claim["target_id"] == target and claim["lower_bound_filename"] == "REZ\\AAAAAAAA.WAV" and claim["upper_bound_filename"] == "REZ\\BBBBBBBB.WAV",
          "an older client claims the whole first length")
    check("encryption_key_hex" not in claim and "hash_table_size" not in claim, "and is told nothing it doesn't know what to do with")
    server.request("POST", "/api/v1/ranges/%d/complete" % claim["range_id"],
                   {"found": True, "filename": filename, "elapsed_seconds": 1.0, "candidates_processed": 256}, token=old["token"])

    shown = server.target(target)
    check(shown["status"] == "active" and shown["found_filename"] is None, "the target stays unsolved")
    check([f["filename"] for f in shown["false_matches"]] == [filename] and shown["false_matches"][0]["reported_by"] == "old@test",
          "the name is kept as a false match, reported by the older client")
    check(shown["false_matches"][0]["reason"].startswith("its basename's key is"), "with why: " + shown["false_matches"][0]["reason"])
    first = sorted((r["first_candidate"], r["last_candidate"], r["status"]) for r in shown["ranges"])
    check(first == [("AAAAAAAA", "ABBABAAB", "completed"), ("ABBABABA", "BBBBBBBB", "pending")],
          "everything up to it is searched, the rest is handed out again: %s" % first)
    # Out of the way of phase 2.
    server.request("PATCH", "/api/v1/admin/targets/%d" % target, {"status": "paused"}, admin=True)


def phase2(server, client_binary, workdir, backend):
    """This client searches on past false matches."""
    print("--- phase 2: this client ---", flush=True)
    wrong_key, wrong_slots, right = "ABBABAAB", "BAABBABA", "BBBAAABA"
    names = {p: "REZ\\" + p + ".WAV" for p in (wrong_key, wrong_slots, right)}
    key_target = server.create("wrong-key", wrong_key, priority=3, encryption_key_hex=hex32(key_of(names[wrong_key]) ^ 1))
    slot = slot_of(names[wrong_slots], 2048)
    slots_target = server.create("wrong-slots", wrong_slots, priority=2,
                                 hash_table_position={"table_size": 2048, "first_start_slot": (slot + 1) % 2048, "last_start_slot": (slot + 3) % 2048})
    slot = slot_of(names[right], 2048)
    right_target = server.create("right", right, priority=1, encryption_key_hex=hex32(key_of(names[right])),
                                 hash_table_position={"table_size": 2048, "first_start_slot": (slot - 1) % 2048, "last_start_slot": slot})

    matches = os.path.join(workdir, "matches")
    config = os.path.join(workdir, "client.conf")
    with open(config, "w") as f:
        f.write("mode = coordinator\nmatches_dir = %s\nbackend = %s\ncheck_for_updates = false\n" % (matches, backend))
        f.write("[coordinator]\nserver_url = %s\nusername = new\nhostname = test\npoll_interval_secs = 1\n" % server.url)
    log_path = os.path.join(workdir, "client.log")
    with open(log_path, "w") as log:
        client = subprocess.Popen([client_binary, "--config", config], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=workdir)
        try:
            # The lowest priority last: by the time it's solved, both others
            # have been searched to the end.
            deadline = time.time() + 300
            while server.target(right_target)["status"] != "solved":
                if client.poll() is not None or time.time() > deadline:
                    raise Failed("the client never found %s - see client.log" % names[right])
                time.sleep(0.2)
        finally:
            client.kill()
            client.wait()
    with open(log_path) as f:
        said = f.read()

    shown = server.target(right_target)
    check(shown["found_filename"] == names[right] and shown["found_by"] == "new@test" and not shown["false_matches"],
          "the name that passes the key and the slots is the find")
    for target, planted, reason in [(key_target, wrong_key, "its basename's key is"), (slots_target, wrong_slots, "its lookup starts at hash table slot")]:
        shown = server.target(target)
        check(shown["status"] == "active", "%s: unsolved" % shown["name"])
        check([f["filename"] for f in shown["false_matches"]] == [names[planted]] and shown["false_matches"][0]["reason"].startswith(reason),
              "%s: its planted name a false match, with why: %s" % (shown["name"], shown["false_matches"][0]["reason"]))
        ranges = shown["ranges"]
        check(len(ranges) == 9 and all(r["status"] == "completed" for r in ranges) and not shown["gaps"],
              "%s: every length searched to the end, past it, a range each, no gaps" % shown["name"])
        check(ranges[0]["first_candidate"] == "AAAAAAAA" and ranges[0]["last_candidate"] == "BBBBBBBB" and ranges[0]["candidate_count"] == "256",
              "%s: the range it's in completed whole, not stopped at it" % shown["name"])
        check("FALSE MATCH (both hashes, but %s" % reason in said and names[planted] in said, "%s: the client said so" % shown["name"])
    check(server.target(right_target)["hash_table_position"]["table_size"] == 2048, "the dashboard shows the hash table position")
    with open(os.path.join(matches, "false_matches.txt")) as f:
        written = sorted(line.split("\t")[0] for line in f.read().splitlines())
    check(written == sorted([names[wrong_key], names[wrong_slots]]), "the client wrote both to false_matches.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    client_binary = os.path.abspath(args.client)
    coordinator = os.path.abspath(args.coordinator)

    if shutil.which("cargo") is None:
        print("cargo isn't installed - can't build the server")
        return 77
    build = subprocess.run(["cargo", "build", "--quiet", "-p", "namebreak-server"], cwd=coordinator)
    if build.returncode != 0:
        print("building the server failed")
        return 1
    server_binary = os.path.join(coordinator, "target", "debug", "namebreak-server")

    workdir = os.path.abspath("coordinator_false_match_test")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    server = Server(server_binary, workdir)
    try:
        phase1(server)
        phase2(server, client_binary, workdir, args.backend)
    except Failed as failure:
        print("FAILED: %s" % failure, flush=True)
        return 1
    finally:
        server.stop()
    print("ALL CHECKS PASSED")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
