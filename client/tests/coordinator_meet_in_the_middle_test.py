#!/usr/bin/env python3
"""Meet-in-the-middle targets end to end: the real coordinator server (../coordinator)
and this client in coordinator mode, talking over HTTP on localhost.

Phase 1 - the memory a client will give: a meet-in-the-middle target whose table is
bigger than one client will allow and another says nothing about. Neither
gets any work from it, while a third, which will give enough, gets it - and
each says at registration what it's offering, which the server keeps. (A
table of four characters of 43, about 30 MB - enough to tell the three apart
without a test allocating gigabytes.)

Phase 2 - finding the directory: a meet-in-the-middle target for a planted name, over
an alphabet small enough to walk in a moment. The client claims it, rewinds
the tails, finds the directory, and the server records the find, solves the
target and shows it all on the dashboard.

Phase 3 - a quit part way: a meet-in-the-middle target whose range is hours of work.
The client is told to quit (SIGTERM, or Ctrl+Break on Windows) once it has
got somewhere, and its quit report has to carry how far it got as a tail
index - the server splits the range there and hands the rest out, which the
next claim takes up from exactly that tail.

Phase 4 - a canary: a server that hands out nothing but canaries, and a
meet-in-the-middle target whose own directory is nowhere in the alphabet it searches,
so that it stays active and unsolved. The client's first claim is a real
range of it, which measures its tails a second; from then on it's handed
canaries of that target, and it has to find the directory planted in each -
the server recording the canary as found.

    coordinator_meet_in_the_middle_test.py --client <namebreak> --coordinator <coordinator dir> [--backend cpu]

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

# The 43 characters of a real search, and a small one for the target whose
# directory is meant to be found in a moment. Both in ascending order, as the
# server stores a custom alphabet.
BIG_ALPHABET = " ()-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\\_"
SMALL_ALPHABET = "ADEPSTU"


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


def mpq_state(text, offset, state=(0x7FED7FED, 0xEEEEEEEE)):
    """Storm's hash state after `text` (already uppercase, with '\\'), from
    `state`: 0x100 is hash A's table, 0x200 hash B's. The state after a
    directory is what a meet-in-the-middle target is given."""
    seed1, seed2 = state
    for ch in text.encode("ascii"):
        seed1 = (TABLE[offset + ch] ^ (seed1 + seed2)) & 0xFFFFFFFF
        seed2 = (ch + seed1 + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
    return seed1, seed2


def hex32(value):
    return "0x%08X" % value


def meet_in_the_middle_settings(prefix, directory, suffix, min_length, max_length, table_length):
    """What a target for `prefix` + `directory` + `suffix` is made with: the
    hash state after the directory, and the whole name's two hashes - as an
    operator who had recovered them would give them."""
    a = mpq_state(prefix + directory, 0x100)
    b = mpq_state(prefix + directory, 0x200)
    return {
        "hash_a_hex": hex32(mpq_state(prefix + directory + suffix, 0x100)[0]),
        "hash_b_hex": hex32(mpq_state(prefix + directory + suffix, 0x200)[0]),
        "meet_in_the_middle": {
            "meeting_point_a_seed1_hex": hex32(a[0]), "meeting_point_a_seed2_hex": hex32(a[1]),
            "meeting_point_b_seed1_hex": hex32(b[0]), "meeting_point_b_seed2_hex": hex32(b[1]),
            "min_length": min_length, "max_length": max_length, "table_length": table_length,
        },
    }


class Failed(Exception):
    pass


def check(ok, what):
    if not ok:
        raise Failed(what)
    print("  ok: " + what, flush=True)


class Server:
    # Every server started, to stop whatever happens.
    started = []

    def __init__(self, binary, workdir, port=None, log_name="server.log", extra_env=None):
        if port is None:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        self.binary, self.workdir, self.port = binary, workdir, port
        self.extra_env = extra_env or {}
        self.url = "http://127.0.0.1:%d" % self.port
        env = dict(os.environ)
        env.update({
            "ADMIN_TOKEN": ADMIN_TOKEN,
            "DATABASE_URL": "sqlite://" + os.path.join(workdir, "coordinator.db"),
            "BIND_ADDR": "127.0.0.1:%d" % self.port,
            "CANARY_PROBABILITY": "0",
            # A range of at least ten billion tails: phase 3's is then hours
            # of work, long enough to quit part way, and the small target's
            # is all of its own in one.
            "MIN_CHUNK_CANDIDATES": "10000000000",
            "RUST_LOG": "warn",
        })
        env.update(self.extra_env)
        self.log = open(os.path.join(workdir, log_name), "w")
        self.process = subprocess.Popen([binary], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        Server.started.append(self)
        for _ in range(100):
            try:
                self.get("/api/v1/status")
                return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        raise Failed("the server didn't start - see " + log_name)

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

    def get(self, path):
        return self.request("GET", path)

    def admin(self, method, path, body=None):
        return self.request(method, path, body=body, admin=True)

    def create(self, name, settings, **extra):
        body = {"name": name, "prefix": "", "suffix": "\\BWUNIN.PIF"}
        body.update(settings)
        body.update(extra)
        return self.admin("POST", "/api/v1/admin/targets", body)["target_id"]

    def target(self, target_id):
        return next(t for t in self.get("/api/v1/dashboard")["targets"] if t["id"] == target_id)

    def ranges(self, target_id):
        return self.target(target_id)["ranges"]

    def rows(self, query, *params):
        """Straight out of the database - for what no endpoint shows (a
        canary's result, a range's own tail indices)."""
        with sqlite3.connect(os.path.join(self.workdir, "coordinator.db")) as db:
            return db.execute(query, params).fetchall()

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()

    def restarted(self, extra_env=None, log_name="server-restarted.log"):
        """The same server again - same port, same database - with other
        settings, which are read at startup."""
        self.stop()
        return Server(self.binary, self.workdir, self.port, log_name, extra_env)


class Client:
    # Every client started, to stop whatever happens.
    started = []

    def __init__(self, binary, workdir, server, backend, name, meet_in_the_middle_table_memory="1G"):
        self.name = name
        self.matches = os.path.join(workdir, name + "-matches")
        config = os.path.join(workdir, name + ".conf")
        with open(config, "w") as f:
            f.write("mode = coordinator\n")
            f.write("matches_dir = %s\n" % self.matches)
            f.write("backend = %s\n" % backend)
            f.write("check_for_updates = false\n")
            f.write("[coordinator]\n")
            f.write("server_url = %s\n" % server.url)
            f.write("username = %s\n" % name)
            f.write("hostname = test\n")
            f.write("poll_interval_secs = 1\n")
            if meet_in_the_middle_table_memory is not None:
                f.write("meet_in_the_middle_table_memory = %s\n" % meet_in_the_middle_table_memory)
        self.output_path = os.path.join(workdir, name + ".log")
        self.output = open(self.output_path, "w")
        # On Windows, a process group of its own, for quit()'s Ctrl+Break to
        # go to it alone.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen([binary, "--config", config], stdin=subprocess.DEVNULL, stdout=self.output, stderr=subprocess.STDOUT,
                                        cwd=workdir, creationflags=flags)
        Client.started.append(self)

    def quit(self):
        """SIGTERM: the client tells the server how far it got, and exits.
        Windows has no SIGTERM to send - Popen would terminate the process
        outright - so it's Ctrl+Break there, a quit to the client too."""
        self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
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


def wait_for(what, condition, client, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        if not client.alive():
            raise Failed("the client stopped while waiting for " + what + ":\n" + client.text()[-3000:])
        time.sleep(0.2)
    raise Failed("timed out waiting for " + what + ":\n" + client.text()[-3000:])


def registered(server, name):
    """What the server kept of a client's declared meet-in-the-middle table memory -
    None until it has registered."""
    rows = server.rows("SELECT meet_in_the_middle_table_bytes FROM users WHERE username = ?", name)
    return rows[0][0] if rows else None


def phase1(server, client_binary, workdir, backend):
    """The memory a client will give decides whether it's offered the work."""
    print("--- phase 1: the memory a client will give ---", flush=True)
    # Four characters of 43 is about 30 MB of table.
    target = server.create("meet-in-the-middle-table-memory", meet_in_the_middle_settings("", "SETUPDATA", "\\BWUNIN.PIF", 9, 9, 4), alphabet=BIG_ALPHABET,
                           priority=10)
    table_bytes = server.target(target)["meet_in_the_middle"]["table_bytes"]
    check((16 << 20) < table_bytes < (64 << 20), "its table is about 30 MB: %d bytes" % table_bytes)

    # Too little, and nothing said at all: no work from it, and no range of
    # it ever carved. (A client's own output is block-buffered into its log,
    # so what it said is read once it has exited.)
    for name, memory, expected in [("stingy", "16M", 16 << 20), ("silent", None, None)]:
        client = Client(client_binary, workdir, server, backend, name, meet_in_the_middle_table_memory=memory)
        wait_for(name + " to register", lambda: server.rows("SELECT 1 FROM users WHERE username = ?", name), client, timeout=120)
        check(registered(server, name) == expected, "the server keeps what %s will give a table: %s" % (name, registered(server, name)))
        time.sleep(2)  # long enough to have claimed, had it been offered anything
        check(not server.ranges(target), "a client that %s gets none of its work" % ("won't give the table's memory" if memory else "says nothing"))
        client.quit()
        said = "16 MB for a meet-in-the-middle table" if memory else "no meet-in-the-middle work"
        check(said in client.text(), "and says so when it registers: " + said)

    # One that will give enough gets it, and stops once it has - leaving this
    # target out of the way of the phases below.
    giving = Client(client_binary, workdir, server, backend, "giving", meet_in_the_middle_table_memory="1G")
    wait_for("a claim of it", lambda: bool(server.ranges(target)), giving, timeout=180)
    check(registered(server, "giving") == (1 << 30), "a client that will give the memory gets the work")
    giving.quit()
    server.admin("PATCH", "/api/v1/admin/targets/%d" % target, {"status": "paused"})


def phase2(server, client_binary, workdir, backend):
    """The planted directory is found, and the server records it."""
    print("--- phase 2: finding the directory ---", flush=True)
    directory, suffix = "SETUPDAT", "\\BWUNIN.PIF"
    # Seven characters, eight of them a directory, five in the table: 343
    # tails, a moment's work either way.
    target = server.create("meet-in-the-middle-find", meet_in_the_middle_settings("", directory, suffix, 8, 8, 5), alphabet=SMALL_ALPHABET, priority=5,
                           prune_symbol_runs=True, prune_adjacent_backslashes=True)
    client = Client(client_binary, workdir, server, backend, "finder")
    wait_for("the find", lambda: server.target(target)["status"] == "solved", client, timeout=180)

    shown = server.target(target)
    check(shown["found_filename"] == directory + suffix, "the server records the name: " + str(shown["found_filename"]))
    check("BOTH HASHES MATCH: " + directory + suffix in client.text(), "and the client said so too")
    check(shown["meet_in_the_middle"]["table_length"] == 5 and shown["meet_in_the_middle"]["tail_count"] == "343", "the dashboard shows the table and the tails")
    range_row = shown["ranges"][0]
    check(range_row["candidate_len"] == 8 and range_row["first_candidate"].startswith("?????"), "its range is tails of one length: " + range_row["first_candidate"])
    check(os.path.isfile(os.path.join(client.matches, "matches-meet-in-the-middle-find.txt")), "the client wrote the match to its own file")
    client.quit()


def phase3(server, client_binary, workdir, backend):
    """A quit part way through a range carries the tail it got to."""
    print("--- phase 3: a quit part way ---", flush=True)
    # 43 characters, 11 of them a directory, 4 in the table: ten billion
    # tails a range, hours of work on a CPU and minutes on a GPU.
    target = server.create("meet-in-the-middle-quit", meet_in_the_middle_settings("", "SETUPDATXYZ", "\\BWUNIN.PIF", 11, 11, 4), alphabet=BIG_ALPHABET, priority=1)
    client = Client(client_binary, workdir, server, backend, "quitter")
    wait_for("a claim", lambda: bool(server.ranges(target)), client, timeout=180)
    first = server.ranges(target)[0]
    # Once it has looked something up, it has progress to report.
    wait_for("the walk to get going", lambda: "tails," in client.text(), client, timeout=180)
    check(client.quit() == 0, "the client quits cleanly with a range in hand")
    check("looked up every tail numbered below" in client.text(), "its quit report says how far the walk got")

    ranges = sorted(server.ranges(target), key=lambda r: r["id"])
    check(len(ranges) == 2, "the server split the range there: %d rows" % len(ranges))
    searched, rest = ranges[0], ranges[1]
    check(searched["status"] == "completed" and rest["status"] == "pending", "the searched part is kept, the rest handed back")
    rows = server.rows("SELECT start_index, end_index FROM ranges WHERE target_id = ? ORDER BY id", target)
    check(rows[0][1] == rows[1][0] and rows[0][1] > 0, "the two meet at the tail it got to (%d)" % rows[0][1])
    # The dashboard writes a count as a decimal string (it can be past 2^53).
    check(rows[1][1] == int(first["candidate_count"]), "and the rest reaches the end of what was claimed")

    # The next claim takes up from exactly there.
    again = Client(client_binary, workdir, server, backend, "resumer")
    wait_for("the rest to be claimed", lambda: any(r["status"] == "in_progress" for r in server.ranges(target)), again, timeout=180)
    taken = server.rows("SELECT start_index FROM ranges WHERE target_id = ? AND status = 'in_progress'", target)
    check(taken and taken[0][0] == rows[0][1], "from the tail the first client stopped at")
    again.quit()
    server.admin("PATCH", "/api/v1/admin/targets/%d" % target, {"status": "paused"})


def phase4(server, client_binary, workdir, backend):
    """A canary of a meet-in-the-middle target: the client has to find the directory
    planted in it."""
    print("--- phase 4: a canary ---", flush=True)
    measured = server.rows("SELECT ema_meet_in_the_middle_rate_per_sec FROM users WHERE ema_meet_in_the_middle_rate_per_sec IS NOT NULL")
    check(bool(measured), "a completed range measured a client's tails a second: %s" % (measured[0][0] if measured else None))

    # Nothing but canaries from here on - the probability is read at startup.
    server = server.restarted({"CANARY_PROBABILITY": "1", "CANARY_SECONDS": "2"})
    # A real meeting point, of a directory made of characters this target's
    # alphabet hasn't - so nothing it searches can be the file, and it stays
    # active for its canaries to copy. Eight characters, five of them the
    # table: 512 tails a range.
    suffix = "\\BWUNIN.PIF"
    alphabet = "BCFGHIJK"
    target = server.create("meet-in-the-middle-canary-template", meet_in_the_middle_settings("", "SETUPDAT", suffix, 8, 8, 5), alphabet=alphabet, priority=7)
    client = Client(client_binary, workdir, server, backend, "canaried")
    # Its first claim is real work (a canary needs a measured rate), which
    # measures it; the claims after that are canaries.
    wait_for("the client's own rate to be measured",
             lambda: bool(server.rows("SELECT 1 FROM users WHERE username = 'canaried' AND ema_meet_in_the_middle_rate_per_sec IS NOT NULL")),
             client, timeout=240)
    check(server.target(target)["status"] == "active", "its own directory is nowhere in the alphabet it searches, so it's still active")
    # Its result is 'pending' from the moment it's handed out, so what's
    # waited for is one the client has answered for.
    wait_for("a canary to be handed out and resolved",
             lambda: bool(server.rows("SELECT result FROM canaries WHERE result <> 'pending'")), client, timeout=240)

    planted, result = server.rows("SELECT filename, result FROM canaries WHERE result <> 'pending'")[0]
    check(result == "found", "the canary was found: %s" % planted)
    check("virtual-canary" in client.text(), "the client searched it as ordinary work")
    directory = planted[:-len(suffix)]
    check(planted.endswith(suffix) and len(directory) == 8, "its planted name is a directory of the target's length: " + directory)
    check(all(c in alphabet for c in directory), "made of the target's own alphabet")
    client.quit()
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--coordinator", required=True)
    # The cpu backend: phase 3 needs a range to take long enough to quit it
    # part way, and a GPU's table has to fit in the device's own memory,
    # which a test machine's may not.
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

    workdir = os.path.abspath("coordinator_meet_in_the_middle_test")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    server = Server(server_binary, workdir)
    try:
        phase1(server, client_binary, workdir, args.backend)
        phase2(server, client_binary, workdir, args.backend)
        phase3(server, client_binary, workdir, args.backend)
        server = phase4(server, client_binary, workdir, args.backend)
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
