#!/usr/bin/env python3
"""Mutation testing of the GPU backends' search kernels.

Builds copies of the client in which a backend's kernel is broken on purpose -
each mutation a small bug that would make a search silently miss (or invent)
candidates - and checks that every one of them is caught by each of the
three checks that guard the search, on its own:

  self-test          the known-answer test createBackend runs on every
                     backend (src/backends/self_test.h)
  integration        tests/search_integration_test.cpp
  stress             tests/search_stress_test.cpp

In the copies, createBackend's own self-test is switched off, so that the
integration and stress tests are judged by themselves; the self-test is run
separately instead. Two kinds of run check the experiment itself: the
unmodified code must pass all three checks (otherwise nothing here means
anything, and the script stops), and so must the mutations marked harmless -
changes that must *not* change what gets searched.

One backend per run (--backend): cuda (the default), hip (the same code,
compiled with HIP), opencl or metal (on a Mac), each with its own list of
mutations (--list).
Needs the GPU the backend runs on, its toolchain or driver, and CMake. It
copies CMakeLists.txt, src/ and tests/ as they are on disk into a work
directory, so it tests uncommitted changes too and never touches build/.
The CUDA list takes about seven minutes on a laptop i9-12900H and RTX 3080
Ti.

Run from anywhere:  python3 client/tests/mutation_test.py [--backend opencl] [options]
or, from a configured build:  cmake --build --preset default --target run_mutation_test
                              (or run_mutation_test_opencl)

Exit status: 0 if every mutation was caught by all three checks and every
control passed them, 1 if not, 2 if the experiment couldn't be run.

A mutation is an exact string replacement in one file. When the code it
targets changes, the string no longer matches and the script says so -
update the mutation to the new code rather than dropping it.
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

CLIENT = Path(__file__).resolve().parent.parent
KERNEL = "src/backends/cuda/cuda_backend.cu"
FILTER = "src/backends/common/lowbits_filter.cpp"
CL_KERNEL = "src/backends/opencl/search.cl"
CL_HOST = "src/backends/opencl/opencl_backend.cpp"
MTL_KERNEL = "src/backends/metal/search.metal"
MTL_HOST = "src/backends/metal/metal_backend.mm"


@dataclass
class Mutation:
    name: str
    what: str
    # (file, old, new): `old` must occur exactly once in `file`.
    edits: list
    # "caught": every check must fail. "harmless": every check must pass.
    expect: str = "caught"


# The table builder is shared by every backend that has the filter.
WRONG_TARGET = Mutation("wrongtarget", "the table built for the wrong target",
                        [(FILTER, "if (((seed1[b] ^ constants.targetHashA) & kLowBitsFilterHashMask) == 0)",
                          "if (((seed1[b] ^ (constants.targetHashA ^ 1)) & kLowBitsFilterHashMask) == 0)")])

CUDA_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(KERNEL, "mask &= (uint64_t(1) << AlphabetSize) - 1;", "mask &= 0xFFFFFFFFull;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(KERNEL, "bufs.filterTable[lowBitsFilterIndex(seed1, seed2)]", "bufs.filterTable[lowBitsFilterIndex(seed2, seed1)]")]),
    Mutation("lastrow", "a launch's last row one candidate short",
             [(KERNEL, "mask &= (uint64_t(1) << lastRowEndK) - 1;", "mask &= (uint64_t(1) << (lastRowEndK - 1)) - 1;")]),
    Mutation("firstrow", "a launch's first row one candidate short",
             [(KERNEL, "mask &= ~uint64_t(0) << firstRowStartK;", "mask &= ~uint64_t(0) << (firstRowStartK + 1);")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(KERNEL, "if (d == firstRowD)", "if (false)"), (KERNEL, "if (d == lastRowD)", "if (false)")]),
    Mutation("skiplast", "the loop over a row's flagged candidates stops one early",
             [(KERNEL, "while (mask != 0) {", "while ((mask & (mask - 1)) != 0) {")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(KERNEL, "#pragma unroll\n                for (int i = 0; i < SuffixLen; ++i)",
               "#pragma unroll\n                for (int i = 0; i + 1 < SuffixLen; ++i)")]),
    WRONG_TARGET,
    # Row groups and their chunks.
    Mutation("chunkend", "every chunk's last row skipped",
             [(KERNEL, "int dEnd = (int) ((chunk + 1) * AlphabetSize / kChunks);", "int dEnd = (int) ((chunk + 1) * AlphabetSize / kChunks) - 1;")]),
    Mutation("chunkstart", "every chunk's first row skipped",
             [(KERNEL, "int dBegin = (int) (chunk * AlphabetSize / kChunks);", "int dBegin = (int) (chunk * AlphabetSize / kChunks) + 1;")]),
    Mutation("firstclip", "the range cut one row short at its start",
             [(KERNEL, "dBegin = (int) (firstRow - groupStart);", "dBegin = (int) (firstRow - groupStart) + 1;")]),
    Mutation("lastclip", "the range cut one row short at its end",
             [(KERNEL, "dEnd = (int) ((uint64_t) lastRow + 1 - groupStart);", "dEnd = (int) ((uint64_t) lastRow - groupStart);")]),
    Mutation("firstrowd", "the first row's cut applied to the wrong row",
             [(KERNEL, "? (int) (firstRow % AlphabetSize) : -1;", "? (int) (firstRow % AlphabetSize) + 1 : -1;")]),
    Mutation("lastrowd", "the last row's cut applied to the wrong row",
             [(KERNEL, "? (int) (lastRow % AlphabetSize) : -1;", "? (int) (lastRow % AlphabetSize) - 1 : -1;")]),
    Mutation("groupdigits", "one character too few of a row group hashed (trailing length 5)",
             [(KERNEL, "case 3: hashRowDigits<AlphabetSize, 3>(group, group1, group2, sKey, sOrd); break;",
               "case 3: hashRowDigits<AlphabetSize, 2>(group, group1, group2, sKey, sOrd); break;")]),
    Mutation("roword", "the wrong character in a row's own hash step",
             [(KERNEL, "mpqStep(seed1, seed2, sKey[d], sOrd[d]);", "mpqStep(seed1, seed2, sKey[d], sOrd[dBegin]);")]),
    Mutation("chunkcount", "a launch's last chunk never searched",
             [(KERNEL, "(lastRow / AlphabetSize - firstGroup + 1) * kChunks;", "(lastRow / AlphabetSize - firstGroup + 1) * kChunks - 1;")]),
    Mutation("lastgroup", "a launch's last row group never searched, when there's more than one",
             [(KERNEL, "(lastRow / AlphabetSize - firstGroup + 1) * kChunks;",
               "(lastRow / AlphabetSize - firstGroup + (lastRow / AlphabetSize > firstGroup ? 0 : 1)) * kChunks;")]),
    # The kernel works out its chunks itself, so how many threads are
    # launched must not change what gets searched.
    Mutation("hostthreads", "one thread too few launched", expect="harmless",
             edits=[(KERNEL, "const uint64_t threads = groups * kChunksPerGroup<AlphabetC::value>;",
                     "const uint64_t threads = groups * kChunksPerGroup<AlphabetC::value> - 1;")]),
]

OPENCL_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(CL_KERNEL, "mask &= (1UL << ALPHABET_SIZE) - 1UL;", "mask &= 0xFFFFFFFFUL;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(CL_KERNEL, "ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];", "ulong mask = filterTable[FILTER_INDEX(seed2, seed1)];")]),
    Mutation("indexshift", "the kernel's table index shifting seed2's bits one too far",
             [(CL_KERNEL, "(((seed2) & FILTER_STATE_MASK) << FILTER_BITS))", "(((seed2) & FILTER_STATE_MASK) << (FILTER_BITS + 1)))")]),
    Mutation("filterbits", "the kernel compiled for a table one bit narrower",
             [(CL_HOST, '" -DFILTER_BITS=" + std::to_string(kLowBitsFilterBits) +', '" -DFILTER_BITS=" + std::to_string(kLowBitsFilterBits - 1) +')]),
    Mutation("lastrow", "a batch's last row one candidate short",
             [(CL_KERNEL, "mask &= (1UL << lastRowEndK) - 1UL;", "mask &= (1UL << (lastRowEndK - 1)) - 1UL;")]),
    Mutation("firstrow", "a batch's first row one candidate short",
             [(CL_KERNEL, "mask &= ~0UL << firstRowStartK;", "mask &= ~0UL << (firstRowStartK + 1);")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(CL_KERNEL, "if (d == firstRowD)", "if (false)"), (CL_KERNEL, "if (d == lastRowD)", "if (false)")]),
    Mutation("skiplast", "the loop over a row's flagged candidates stops one early",
             [(CL_KERNEL, "while (mask != 0) {", "while ((mask & (mask - 1UL)) != 0) {")]),
    Mutation("wrongbit", "a flagged bit taken for the character after it",
             [(CL_KERNEL, "const int k = 63 - (int) clz(mask & (0UL - mask));", "const int k = 64 - (int) clz(mask & (0UL - mask));")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(CL_KERNEL, "for (int i = 0; i < SUFFIX_LEN; ++i)\n                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);",
               "for (int i = 0; i + 1 < SUFFIX_LEN; ++i)\n                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);")]),
    WRONG_TARGET,
    # Row groups and their chunks.
    Mutation("chunkend", "every chunk's last row skipped",
             [(CL_KERNEL, "int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP);", "int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP) - 1;")]),
    Mutation("chunkstart", "every chunk's first row skipped",
             [(CL_KERNEL, "int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP);", "int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP) + 1;")]),
    Mutation("firstclip", "the range cut one row short at its start",
             [(CL_KERNEL, "dBegin = (int) (firstRow - groupStart);", "dBegin = (int) (firstRow - groupStart) + 1;")]),
    Mutation("lastclip", "the range cut one row short at its end",
             [(CL_KERNEL, "dEnd = (int) ((ulong) lastRow + 1 - groupStart);", "dEnd = (int) ((ulong) lastRow - groupStart);")]),
    Mutation("firstrowd", "the first row's cut applied to the wrong row",
             [(CL_KERNEL, "? (int) (firstRow % ALPHABET_SIZE) : -1;", "? (int) (firstRow % ALPHABET_SIZE) + 1 : -1;")]),
    Mutation("lastrowd", "the last row's cut applied to the wrong row",
             [(CL_KERNEL, "? (int) (lastRow % ALPHABET_SIZE) : -1;", "? (int) (lastRow % ALPHABET_SIZE) - 1 : -1;")]),
    Mutation("groupdigits", "one character too few of a row group hashed",
             [(CL_KERNEL, "for (int i = 0; i < TRAILING_LEN - 2; ++i)\n            MPQ_STEP(group1, group2, sKey[digit[i]], sOrd[digit[i]]);",
               "for (int i = 0; i + 1 < TRAILING_LEN - 2; ++i)\n            MPQ_STEP(group1, group2, sKey[digit[i]], sOrd[digit[i]]);")]),
    Mutation("roword", "the wrong character in a row's own hash step",
             [(CL_KERNEL, "MPQ_STEP(seed1, seed2, sKey[d], sOrd[d]);", "MPQ_STEP(seed1, seed2, sKey[d], sOrd[dBegin]);")]),
    Mutation("chunkcount", "a batch's last chunk never searched",
             [(CL_KERNEL, "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;",
               "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP - 1;")]),
    Mutation("lastgroup", "a batch's last row group never searched, when there's more than one",
             [(CL_KERNEL, "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;",
               "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + (lastRow / ALPHABET_SIZE > firstGroup ? 0 : 1)) * CHUNKS_PER_GROUP;")]),
    # The kernel works out its chunks itself, so the global work size must
    # not change what gets searched - neither half as many work-items (but at
    # least a work-group) nor a work-group too many.
    Mutation("fewergroups", "half as many work-items launched", expect="harmless",
             edits=[(CL_HOST, "const size_t global = (chunks + local - 1) / local * local;",
                     "const size_t global = ((chunks + 1) / 2 + local - 1) / local * local;")]),
    Mutation("extragroup", "a whole work-group too many launched", expect="harmless",
             edits=[(CL_HOST, "const size_t global = (chunks + local - 1) / local * local;",
                     "const size_t global = (chunks + 2 * local - 1) / local * local;")]),
]

# The Metal ones can only be run on a Mac (--backend metal).
METAL_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(MTL_KERNEL, "mask &= (1ul << ALPHABET_SIZE) - 1ul;", "mask &= 0xFFFFFFFFul;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(MTL_KERNEL, "ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];", "ulong mask = filterTable[FILTER_INDEX(seed2, seed1)];")]),
    Mutation("indexshift", "the kernel's table index shifting seed2's bits one too far",
             [(MTL_KERNEL, "(((seed2) & FILTER_STATE_MASK) << FILTER_BITS))", "(((seed2) & FILTER_STATE_MASK) << (FILTER_BITS + 1)))")]),
    Mutation("filterbits", "the kernel compiled for a table one bit narrower",
             [(MTL_HOST, '@"FILTER_BITS": @(kLowBitsFilterBits),', '@"FILTER_BITS": @(kLowBitsFilterBits - 1),')]),
    Mutation("lastrow", "a batch's last row one candidate short",
             [(MTL_KERNEL, "mask &= (1ul << args.lastRowEndK) - 1ul;", "mask &= (1ul << (args.lastRowEndK - 1)) - 1ul;")]),
    Mutation("firstrow", "a batch's first row one candidate short",
             [(MTL_KERNEL, "mask &= ~0ul << args.firstRowStartK;", "mask &= ~0ul << (args.firstRowStartK + 1);")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(MTL_KERNEL, "if (t == 0)\n        mask &=", "if (false)\n        mask &="),
              (MTL_KERNEL, "if (t == args.rowCount - 1)\n        mask &=", "if (false)\n        mask &=")]),
    Mutation("skiplast", "the loop over a row's flagged candidates stops one early",
             [(MTL_KERNEL, "while (mask != 0) {", "while ((mask & (mask - 1ul)) != 0) {")]),
    Mutation("wrongbit", "a flagged bit in the low half taken for the character after it",
             [(MTL_KERNEL, "(low != 0) ? int(ctz(low)) : 32 + int(ctz(high));", "(low != 0) ? int(ctz(low)) + 1 : 32 + int(ctz(high));")]),
    Mutation("highhalf", "a flagged bit in the high half taken for the character before it",
             [(MTL_KERNEL, "(low != 0) ? int(ctz(low)) : 32 + int(ctz(high));", "(low != 0) ? int(ctz(low)) : 31 + int(ctz(high));")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(MTL_KERNEL, "for (int i = 0; i < SUFFIX_LEN; ++i)\n            MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);",
               "for (int i = 0; i + 1 < SUFFIX_LEN; ++i)\n            MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);")]),
    Mutation("rowdigits", "one character too few of a row hashed",
             [(MTL_KERNEL, "for (int i = 0; i < TRAILING_LEN - 1; ++i)\n        MPQ_STEP(seed1, seed2, sKey[digit[i]], sOrd[digit[i]]);",
               "for (int i = 0; i + 1 < TRAILING_LEN - 1; ++i)\n        MPQ_STEP(seed1, seed2, sKey[digit[i]], sOrd[digit[i]]);")]),
    WRONG_TARGET,
    # Threads past the batch's last row return at once, so launching a whole
    # threadgroup too many must not change what gets searched.
    Mutation("extragroup", "a whole threadgroup too many launched", expect="harmless",
             edits=[(MTL_HOST, "const NSUInteger threadgroups = (rows.rowCount + threadgroupSize - 1) / threadgroupSize;",
                     "const NSUInteger threadgroups = (rows.rowCount + threadgroupSize - 1) / threadgroupSize + 1;")]),
]

# Per backend: how to build it, what its createBackend name is, and its mutations.
BACKENDS = {
    "cuda": (["-DNAMEBREAK_GPU=cuda", "-DNAMEBREAK_OPENCL=OFF"], CUDA_MUTATIONS),
    "hip": (["-DNAMEBREAK_GPU=hip", "-DNAMEBREAK_OPENCL=OFF"], CUDA_MUTATIONS),
    "opencl": (["-DNAMEBREAK_GPU=none", "-DNAMEBREAK_OPENCL=ON"], OPENCL_MUTATIONS),
    "metal": (["-DNAMEBREAK_GPU=none", "-DNAMEBREAK_OPENCL=OFF", "-DNAMEBREAK_METAL=ON"], METAL_MUTATIONS),
}

CONTROL = Mutation("none", "the code as it is", [], expect="harmless")

# Runs the self-test on its own: createBackend's is switched off in the copies.
SELF_TEST_RUNNER = r'''// Written by tests/mutation_test.py into its copies of the client.
#include <cstdio>
#include <memory>
#include <string>
#include "backends/backends.h"
#include "backends/self_test.h"
int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<SearchBackend> backend = createBackend(argc > 1 ? argv[1] : "", error);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return 77;
    }
    std::string why;
    if (selfTestBackend(*backend, why)) {
        printf("self-test passed\n");
        return 0;
    }
    printf("self-test FAILED: %s\n", why.c_str());
    return 1;
}
'''

TARGETS = ["search_integration_test", "search_stress_test", "mutation_self_test_runner"]
CHECKS = [
    ("self-test", "mutation_self_test_runner"),
    ("integration", "search_integration_test"),
    ("stress", "search_stress_test"),
]


class ExperimentError(Exception):
    pass


@dataclass
class CheckResult:
    passed: bool
    summary: str


@dataclass
class Outcome:
    mutation: Mutation
    checks: dict = field(default_factory=dict)  # check name -> CheckResult
    error: str = ""                              # build failure etc.

    def as_expected(self):
        if self.error or len(self.checks) != len(CHECKS):
            return False
        want_pass = self.mutation.expect == "harmless"
        return all(r.passed == want_pass for r in self.checks.values())


print_lock = threading.Lock()


def say(message):
    with print_lock:
        print(message, flush=True)


def replace_exactly_once(path, old, new, what):
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise ExperimentError(f"{what}: expected to find this exactly once in {path.name}, found it {count} times:\n    {old!r}\n"
                              "The code it targets has probably changed - update the mutation in tests/mutation_test.py.")
    path.write_text(text.replace(old, new))


def prepare_base(work):
    """A copy of the client with createBackend's self-test switched off and the self-test runner added."""
    base = work / "base"
    base.mkdir()
    shutil.copy2(CLIENT / "CMakeLists.txt", base)
    for sub in ("src", "tests"):
        shutil.copytree(CLIENT / sub, base / sub, ignore=shutil.ignore_patterns("__pycache__"))
    backends = base / "src/backends/backends.cpp"
    text = backends.read_text()
    if text.count("selfTestBackend(*backend, why)") != 2:
        raise ExperimentError("couldn't find createBackend's two calls to selfTestBackend in src/backends/backends.cpp - "
                              "update prepare_base() in tests/mutation_test.py")
    backends.write_text(text.replace("selfTestBackend(*backend, why)", "true"))
    (base / "tests/mutation_self_test_runner.cpp").write_text(SELF_TEST_RUNNER)
    with open(base / "CMakeLists.txt", "a") as cmake:
        cmake.write("\n# Added by tests/mutation_test.py.\n"
                    "namebreak_test_executable(mutation_self_test_runner tests/mutation_self_test_runner.cpp namebreak_search)\n")
    return base


def build(work, base, mutation, args):
    """A copy of `base` with `mutation` applied, built. Returns its build directory."""
    src = work / mutation.name
    shutil.copytree(base, src)
    for file, old, new in mutation.edits:
        replace_exactly_once(src / file, old, new, f"mutation {mutation.name}")
    build_dir = src / "build"
    configure = ["cmake", "-S", str(src), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DNAMEBREAK_METAL=OFF",
                 "-DNAMEBREAK_NETWORK=OFF", "-DNAMEBREAK_TEST_VARIANTS=OFF"] + BACKENDS[args.backend][0]
    configure += args.cmake_arg
    compile_ = ["cmake", "--build", str(build_dir), "-j", str(args.jobs), "--target"] + TARGETS
    log = src / "build.log"
    with open(log, "w") as out:
        for command in (configure, compile_):
            if subprocess.run(command, stdout=out, stderr=subprocess.STDOUT).returncode != 0:
                return None
    return build_dir


def summarize(check, log_text, capture_text, exit_code, timed_out):
    if timed_out:
        return "timed out"
    if check == "self-test":
        line = next((l for l in log_text.splitlines() if l.startswith("self-test")), "")
        if "INTERNAL ERROR" in log_text:
            line = log_text[log_text.index("INTERNAL ERROR"):].splitlines()[0]
        return line[:110] or f"exit {exit_code}"
    if check == "integration":
        m = re.search(r"(\d+) case\(s\) run in [\d.]+s, (\d+) failure\(s\)", log_text)
        if m:
            return f"{m.group(2)} of {m.group(1)} cases failed"
    else:
        m = re.search(r"(\d+) case\(s\), \d+ candidates, (\d+) hits verified, [\d.]+s, (\d+) failure\(s\)", log_text)
        if m:
            return f"{m.group(3)} of {m.group(1)} cases failed ({m.group(2)} hits verified)"
    # Stopped before its summary - e.g. by the per-search table check, whose
    # message lands in the test's output capture.
    for text in (log_text, capture_text):
        if "INTERNAL ERROR" in text:
            return "stopped: " + text[text.index("INTERNAL ERROR"):].splitlines()[0][:90]
    return f"exit {exit_code}, no summary"


def run_checks(build_dir, mutation, args):
    backend = args.backend
    results = {}
    for check, executable in CHECKS:
        run_dir = build_dir.parent / f"run-{check}"
        run_dir.mkdir(exist_ok=True)
        log_path = run_dir / "output.log"
        timed_out = False
        with open(log_path, "wb") as out:
            try:
                code = subprocess.run([str(build_dir / "tests" / executable), "--backend", backend] if check != "self-test"
                                      else [str(build_dir / "tests" / executable), backend],
                                      cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        if code == 77:
            raise ExperimentError(f"the {backend} backend can't run on this machine ({check}, {mutation.name}) - see {log_path}")
        log_text = log_path.read_bytes().decode("utf-8", "replace")
        capture = run_dir / ".capture.tmp"
        capture_text = capture.read_bytes().decode("utf-8", "replace") if capture.exists() else ""
        results[check] = CheckResult(passed=(code == 0), summary=summarize(check, log_text, capture_text, code, timed_out))
    return results


def describe(outcome):
    lines = [f"{outcome.mutation.name}: {outcome.mutation.what} (expected: {outcome.mutation.expect})"]
    if outcome.error:
        lines.append(f"    ERROR: {outcome.error}")
    for check, _ in CHECKS:
        r = outcome.checks.get(check)
        if r:
            lines.append(f"    {check:<12} {'passed' if r.passed else 'FAILED':<7} {r.summary}")
    lines.append("    -> " + ("as expected" if outcome.as_expected() else "NOT AS EXPECTED"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for the whole story.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="cuda", help="the backend to build and test (default cuda)")
    parser.add_argument("--only", help="comma-separated mutation names to run (the unmodified code always runs first)")
    parser.add_argument("--list", action="store_true", help="list the mutations and exit")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="parallel compile jobs per build")
    parser.add_argument("--parallel", type=int, default=3, help="mutations checked at the same time (default 3)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds before one check counts as hung (default 900)")
    parser.add_argument("--work-dir", help="where to put the copies (default: a new temporary directory)")
    parser.add_argument("--keep", action="store_true", help="keep the work directory even if everything went as expected")
    parser.add_argument("--cmake-arg", action="append", default=[], help="extra CMake argument, e.g. -DCMAKE_CUDA_ARCHITECTURES=86")
    args = parser.parse_args()

    mutations = BACKENDS[args.backend][1]
    if args.list:
        for m in mutations:
            print(f"{m.name:<12} {m.expect:<9} {m.what}")
        return 0
    if args.only:
        wanted = [name.strip() for name in args.only.split(",")]
        unknown = [name for name in wanted if name not in {m.name for m in mutations}]
        if unknown:
            print(f"unknown mutation(s): {', '.join(unknown)} (see --list)", file=sys.stderr)
            return 2
        mutations = [m for m in mutations if m.name in wanted]

    work = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="namebreak-mutation-"))
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        print(f"the work directory {work} isn't empty", file=sys.stderr)
        return 2
    say(f"work directory: {work}")
    start = time.monotonic()
    outcomes = []
    try:
        base = prepare_base(work)
        # The unmodified code first: if it doesn't pass, nothing else means anything.
        say("building and checking the unmodified code ...")
        control = Outcome(CONTROL)
        build_dir = build(work, base, CONTROL, args)
        if build_dir is None:
            raise ExperimentError(f"the unmodified code doesn't build - see {work / CONTROL.name / 'build.log'}")
        control.checks = run_checks(build_dir, CONTROL, args)
        say(describe(control))
        if not control.as_expected():
            raise ExperimentError("the unmodified code doesn't pass all three checks, so the experiment can't tell anything")
        outcomes.append(control)

        # Each mutation is built while earlier ones are checked on the GPU.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            pending = []
            for mutation in mutations:
                say(f"building {mutation.name} ...")
                outcome = Outcome(mutation)
                build_dir = build(work, base, mutation, args)
                if build_dir is None:
                    outcome.error = f"doesn't compile - see {work / mutation.name / 'build.log'}"
                    say(describe(outcome))
                    outcomes.append(outcome)
                    continue

                def check(outcome=outcome, build_dir=build_dir):
                    outcome.checks = run_checks(build_dir, outcome.mutation, args)
                    say(describe(outcome))
                    return outcome
                pending.append(pool.submit(check))
            for future in pending:
                outcomes.append(future.result())
    except ExperimentError as e:
        print(f"\nThe experiment couldn't be run: {e}\n(work directory kept: {work})", file=sys.stderr)
        return 2

    minutes = (time.monotonic() - start) / 60
    unexpected = [o for o in outcomes if not o.as_expected()]
    print(f"\n{'mutation':<12} {'expected':<9} {'self-test':<10} {'integration':<12} {'stress':<10} result")
    for o in outcomes:
        cells = ["-" if o.error else ("passed" if o.checks[c].passed else "failed") for c, _ in CHECKS]
        print(f"{o.mutation.name:<12} {o.mutation.expect:<9} {cells[0]:<10} {cells[1]:<12} {cells[2]:<10} "
              f"{'ok' if o.as_expected() else 'NOT AS EXPECTED'}")
    print(f"\n{len(outcomes)} run(s) in {minutes:.1f} minutes: {len(outcomes) - len(unexpected)} as expected, {len(unexpected)} not.")
    if unexpected:
        print("Not as expected: " + ", ".join(o.mutation.name for o in unexpected) +
              f" - a check that lets a bug through, or a mutation that isn't harmless. Work directory kept: {work}")
        return 1
    if args.keep:
        print(f"Work directory kept: {work}")
    else:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
