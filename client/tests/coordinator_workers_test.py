#!/usr/bin/env python3
"""Several GPUs' workers in one client, end to end: the real coordinator
server (../coordinator) and this client in coordinator mode, talking over
HTTP on localhost. The client is namebreak_two_workers, whose cpu backend
searches as two workers - as a machine with two GPUs does, one worker each.

Phase 1 - sharing a target: a target cut into ranges of 2,048 candidates.
Both workers register, each as a client of its own - the second as
<hostname>/gpu1 - claim its ranges between them and find the planted name. Every message
about a range says which worker's it is, and each writes a matches file of
its own.

Phase 2 - quitting: a target of long ranges. Once both workers have a range
in hand, the client is told to quit (SIGTERM): each worker tells the server
how far its range got, the rest of both is handed out again, and it exits.

    coordinator_workers_test.py --client <namebreak_two_workers> --coordinator <coordinator dir>

Builds the server with cargo first. Runs in a directory of its own under the
current one.
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
    A, 0x200 hash B."""
    seed1, seed2 = 0x7FED7FED, 0xEEEEEEEE
    for ch in text.encode("ascii"):
        seed1 = (TABLE[offset + ch] ^ (seed1 + seed2)) & 0xFFFFFFFF
        seed2 = (ch + seed1 + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
    return seed1


def hex32(value):
    return "0x%08X" % value


class Failed(Exception):
    pass


def check(ok, what):
    if not ok:
        raise Failed(what)
    print("  ok: " + what, flush=True)


class Server:
    def __init__(self, binary, workdir, chunk):
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
            # Ranges of `chunk` candidates, however fast a client is.
            "MIN_CHUNK_CANDIDATES": str(chunk),
            "MAX_CHUNK_CANDIDATES": str(chunk),
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

    def request(self, method, path, body=None, admin=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if admin:
            req.add_header("X-Admin-Token", ADMIN_TOKEN)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as response:
            text = response.read().decode()
            return json.loads(text) if text and response.headers.get("Content-Type", "").startswith("application/json") else text

    def create(self, name, alphabet, length, planted):
        """An alphabet target for REZ\\<candidate>.WAV, candidates of `length`
        characters of `alphabet` and on - its hashes `planted`'s."""
        filename = "REZ\\" + planted + ".WAV"
        body = {"name": name, "prefix": "REZ\\", "suffix": ".WAV", "alphabet": alphabet, "lower_bound": alphabet[0] * length,
                "upper_bound": alphabet[-1] * length, "start_len": length, "hash_a_hex": hex32(mpq_hash(filename, 0x100)),
                "hash_b_hex": hex32(mpq_hash(filename, 0x200))}
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


def start_client(client_binary, workdir, server):
    config = os.path.join(workdir, "client.conf")
    with open(config, "w") as f:
        f.write("mode = coordinator\nmatches_dir = %s\nbackend = cpu\ncheck_for_updates = false\n" % os.path.join(workdir, "matches"))
        f.write("[coordinator]\nserver_url = %s\nusername = w\nhostname = test\npoll_interval_secs = 1\n" % server.url)
        # Two threads a worker, so that two of them don't take the whole
        # machine.
        f.write("[hardware]\ncpu_threads = 2\n")
    log = open(os.path.join(workdir, "client.log"), "w")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen([client_binary, "--config", config], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=workdir,
                               creationflags=flags)
    return process, log


def read_log(workdir):
    with open(os.path.join(workdir, "client.log")) as f:
        return f.read()


def phase1(server_binary, client_binary, workdir):
    """Both workers share one target's ranges."""
    print("--- phase 1: sharing a target ---", flush=True)
    server = Server(server_binary, workdir, 2048)
    try:
        # Lengths 8 to 16 of "AB": 130,816 candidates, 64 ranges and more.
        planted = "BABBAABABBBAABAB"
        target = server.create("shared", "AB", 8, planted)
        client, log = start_client(client_binary, workdir, server)
        try:
            deadline = time.time() + 300
            while server.target(target)["status"] != "solved":
                if client.poll() is not None or time.time() > deadline:
                    raise Failed("the client never found %s - see client.log" % planted)
                time.sleep(0.2)
        finally:
            client.kill()
            client.wait()
            log.close()
        said = read_log(workdir)

        users = sorted(r[0] for r in server.rows("SELECT hostname FROM users WHERE username = 'w'"))
        check(users == ["test", "test/gpu1"], "each worker registered as a client of its own: %s" % users)
        completed = dict(server.rows("SELECT u.hostname, COUNT(*) FROM ranges r JOIN users u ON u.id = r.assigned_user_id "
                                     "WHERE r.status = 'completed' GROUP BY u.hostname"))
        check(completed.get("test", 0) > 0 and completed.get("test/gpu1", 0) > 0, "both searched ranges of the target: %s" % completed)
        check(server.target(target)["found_filename"] == "REZ\\" + planted + ".WAV", "and one of them found it")
        check("searching on 2 GPUs at once with the cpu backend" in said, "the client said it searches with two workers")
        written = sorted(f for f in os.listdir(os.path.join(workdir, "matches")) if f.startswith("matches-"))
        check(written == ["matches-shared-gpu1.txt", "matches-shared.txt"], "each worker wrote a matches file of its own: %s" % written)
        for tag in ("[coordinator gpu0] starting range", "[coordinator gpu1] starting range", "[coordinator gpu0] registered",
                    "[coordinator gpu1] registered"):
            check(tag in said, "said with its worker's tag: " + tag)
    finally:
        server.stop()


def phase2(server_binary, client_binary, workdir):
    """A quit hands back both workers' ranges."""
    print("--- phase 2: quitting ---", flush=True)
    server = Server(server_binary, workdir, 100_000_000_000)
    try:
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        target = server.create("long", alphabet, 8, "ZZZZZZZZ")
        client, log = start_client(client_binary, workdir, server)
        try:
            deadline = time.time() + 120
            while len(server.rows("SELECT id FROM ranges WHERE status = 'in_progress'")) < 2:
                if client.poll() is not None or time.time() > deadline:
                    raise Failed("the two workers never both had a range - see client.log")
                time.sleep(0.2)
            held = dict(server.rows("SELECT r.id, u.hostname FROM ranges r JOIN users u ON u.id = r.assigned_user_id WHERE r.status = 'in_progress'"))
            check(sorted(held.values()) == ["test", "test/gpu1"], "each worker has a range in hand: %s" % held)
            # Long enough to have searched some of it.
            time.sleep(2)
            client.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
            try:
                code = client.wait(timeout=60)
            except subprocess.TimeoutExpired:
                raise Failed("the client didn't quit within a minute of being told to")
        finally:
            if client.poll() is None:
                client.kill()
                client.wait()
            log.close()
        said = read_log(workdir)
        check(code == 0, "the client quit cleanly (exit code %d)" % code)
        check(not server.rows("SELECT id FROM ranges WHERE status = 'in_progress'"), "no range is left in hand")
        for range_id in held:
            # What it searched is completed; the rest is a range of its own,
            # to be handed out again.
            (status, end), = server.rows("SELECT status, end_index FROM ranges WHERE id = ?", range_id)
            rest = server.rows("SELECT status FROM ranges WHERE start_index = ? AND id != ?", end, range_id)
            check(status == "completed" and rest == [("pending",)], "range %d was handed back past where it got: %s, the rest %s" % (range_id, status, rest))
        for range_id, hostname in held.items():
            tag = "[coordinator gpu%s]" % ("1" if hostname.endswith("/gpu1") else "0")
            check("%s range %d: told the coordinator" % (tag, range_id) in said, "%s told the server how far range %d got" % (tag, range_id))
        check(server.target(target)["status"] == "active", "the target is still to be searched")
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--coordinator", required=True)
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

    workdir = os.path.abspath("coordinator_workers_test")
    try:
        for phase in (phase1, phase2):
            shutil.rmtree(workdir, ignore_errors=True)
            os.makedirs(workdir)
            phase(server_binary, client_binary, workdir)
    except Failed as failure:
        print("FAILED: %s" % failure, flush=True)
        return 1
    print("ALL CHECKS PASSED")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
