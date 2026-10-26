#!/usr/bin/env python3
"""Tunes the CUDA or HIP backend's compile-time launch settings for this
machine's GPU, and builds a namebreak client with the fastest.

The CUDA/HIP backend compiles its kernels ahead of time, with its launch
settings as template arguments and launch bounds (src/backends/cuda/tuning.h),
so `namebreak --autotune` - which tunes the OpenCL and Metal backends from
config.conf - can't change them. This script does it the slow way instead:
it builds search_bench and dictionary_bench with each candidate setting
(CMake's NAMEBREAK_TUNING), times them on the GPU, and keeps the fastest -
the same way `--autotune` decides (see src/common/autotune.h): one setting at
a time from the defaults, the setting it stands at timed before and after
the others, a value taken only if more than 2% faster, and the result kept
only if it beats the defaults by more than 2% when timed again. Then it
builds the client itself with the result. See README.md next to this file
for what it needs and how to run it.

Python 3.8 or later, standard library only; Linux and Windows.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CLIENT = Path(__file__).resolve().parents[2]
WINDOWS = platform.system() == "Windows"
EXE = ".exe" if WINDOWS else ""

# The settings tuned, the values tried for each, and how each reads its
# default out of src/backends/cuda/tuning.h (the last match - the non-Apple
# one where there are two). The row search's first, then the dictionary
# search's, each timed with its own benchmark.
ROW_AXES = [
    # 13 rows a thread is 4 chunks of a 49-character row group, 25 two, 64 one whatever the alphabet.
    ("NAMEBREAK_ROWS_PER_THREAD", [7, 13, 25, 64], r"#define NAMEBREAK_ROWS_PER_THREAD (\d+)"),
    ("NAMEBREAK_BATCHES_PER_LAUNCH", [4, 8, 16, 32], r"#define NAMEBREAK_BATCHES_PER_LAUNCH (\d+)"),
    ("NAMEBREAK_GPU_WINDOW_CHARS", [4, 5, 6], r"#define NAMEBREAK_GPU_WINDOW_CHARS (\d+)"),
    ("NAMEBREAK_THREADS_PER_BLOCK", [64, 128, 256, 512], r"#define NAMEBREAK_THREADS_PER_BLOCK (\d+)"),
]
DICTIONARY_AXES = [
    ("NAMEBREAK_DICTIONARY_WORDS_PER_THREAD", [8, 16, 32, 64, 128], r"kDictionaryWordsPerThread = dictionaryWordsPerThreadOr\((\d+)\)"),
    ("NAMEBREAK_DICTIONARY_THREADS_PER_BLOCK", [64, 128, 256, 512], r"kDictionaryThreadsPerBlock = dictionaryThreadsPerBlockOr\((\d+)\)"),
]

MIN_GAIN = 0.02  # less is noise
ROUNDS = 2
HEAT_UP_SECONDS = 30  # a GPU that starts cool runs faster until it's warm - a laptop's up to three times as fast
SEARCH_BENCH_RANGE = 28.8e9  # search_bench's candidates at --scale 1


class Failure(Exception):
    """Something the user has to fix - printed without a traceback."""


def say(text=""):
    print(text, flush=True)


def run(command, cwd=None, quiet=False):
    """Runs `command`; its output goes to the terminal unless `quiet`, when
    it's returned (and shown only if it fails)."""
    if not quiet:
        say("$ " + " ".join(str(c) for c in command))
    result = subprocess.run([str(c) for c in command], cwd=cwd, text=True, errors="replace",
                            stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None)
    if result.returncode != 0:
        if quiet and result.stdout:
            say(result.stdout[-4000:])
        raise Failure(f"'{' '.join(str(c) for c in command)}' failed (exit code {result.returncode})")
    return result.stdout or ""


def defaults():
    text = (CLIENT / "src/backends/cuda/tuning.h").read_text()
    values = {}
    for name, _, pattern in ROW_AXES + DICTIONARY_AXES:
        found = re.findall(pattern, text)
        if not found:
            raise Failure(f"couldn't find {name}'s default in src/backends/cuda/tuning.h - update ROW_AXES/DICTIONARY_AXES in {__file__}")
        values[name] = int(found[-1])
    return values


def label(name):
    """NAMEBREAK_ROWS_PER_THREAD as rows_per_thread."""
    return name[len("NAMEBREAK_"):].lower()


def describe(config, base):
    changed = [f"{label(name)} {value}" for name, value in config.items() if value != base[name]]
    return ", ".join(changed) or "the defaults"


# --- Checking the tools -------------------------------------------------------

def check_tools(args):
    if sys.version_info < (3, 8):
        raise Failure("this script needs Python 3.8 or later")
    if not shutil.which("cmake"):
        raise Failure("CMake isn't on the PATH - see README.md next to this script")
    version = re.search(r"(\d+)\.(\d+)", run(["cmake", "--version"], quiet=True))
    if not version or (int(version[1]), int(version[2])) < (3, 24):
        raise Failure(f"CMake 3.24 or later is needed, this one is {version[0] if version else 'unknown'} - "
                      "on Linux, `pip install cmake` gets a new one")
    if args.gpu == "cuda":
        if not shutil.which("nvcc") and not os.environ.get("CUDA_PATH") and not Path("/usr/local/cuda/bin/nvcc").exists():
            raise Failure("the CUDA Toolkit (nvcc) isn't installed, or isn't on the PATH - see README.md next to this script")
        if not shutil.which("nvidia-smi"):
            raise Failure("nvidia-smi isn't on the PATH: is the NVIDIA driver installed?")
    elif not Path(args.rocm).exists() and not shutil.which("hipcc"):
        raise Failure(f"ROCm (or the HIP SDK) isn't installed at {args.rocm} - see README.md next to this script, or pass --rocm <dir>")


def check_gpu_idle(args):
    """Asks before going on if something else is keeping the GPU busy: it
    would make every measurement meaningless. Busy, not merely open - a
    paused namebreak keeps its GPU context but leaves the GPU idle."""
    if args.gpu != "cuda" or args.yes:
        return
    samples = []
    try:
        for _ in range(5):
            output = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=30).stdout.split()
            samples.append(max(int(value) for value in output))
            time.sleep(1)
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    busy = sum(samples) / len(samples)
    if busy < 10:
        return
    say(f"Something else keeps the GPU {busy:.0f}% busy, which makes the measurements meaningless - stop it (or pause it) first.")
    try:
        answer = input("Go on anyway? [y/N] ")
    except EOFError:
        raise Failure("stopped (pass --yes to go on without asking)")
    if answer.strip().lower() != "y":
        raise Failure("stopped")


def release_version(args):
    """The release tag the source is checked out at, for the build's
    NAMEBREAK_VERSION - so the update check knows what it has - or dev."""
    if args.version:
        return args.version
    try:
        tag = subprocess.run(["git", "describe", "--tags", "--exact-match"], cwd=CLIENT, capture_output=True, text=True).stdout.strip()
    except OSError:
        tag = ""
    return tag if re.fullmatch(r"v\d{4}-\d{2}-\d{2}(\.\d+)?", tag) else "dev"


def vcpkg_toolchain():
    for root in (os.environ.get("VCPKG_ROOT"), os.environ.get("VCPKG_INSTALLATION_ROOT"), "C:/vcpkg"):
        if root and (Path(root) / "scripts/buildsystems/vcpkg.cmake").exists():
            return Path(root) / "scripts/buildsystems/vcpkg.cmake"
    return None


def configure_arguments(args, network):
    """What every configure of this machine's builds passes to CMake."""
    arguments = ["-DCMAKE_BUILD_TYPE=Release", f"-DNAMEBREAK_GPU={args.gpu}", "-DNAMEBREAK_TEST_VARIANTS=OFF",
                 f"-DNAMEBREAK_NETWORK={'ON' if network else 'OFF'}"]
    if WINDOWS:
        arguments += ["-G", "Visual Studio 17 2022", "-A", "x64"]
        if args.gpu == "cuda" and os.environ.get("CUDA_PATH"):
            arguments += ["-T", f"cuda={os.environ['CUDA_PATH']}"]
        if network:
            toolchain = vcpkg_toolchain()
            if not toolchain:
                raise Failure("the client's coordinator mode needs libcurl, which on Windows comes from vcpkg - see README.md next to this "
                              "script (set VCPKG_ROOT to where it is), or pass --no-network for a client without coordinator mode")
            arguments += [f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", "-DVCPKG_TARGET_TRIPLET=x64-windows"]
    elif shutil.which("ninja"):
        arguments += ["-G", "Ninja"]
    if args.gpu == "cuda":
        arguments.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.arch or 'native'}")
    else:
        if Path(args.rocm, "llvm/bin/clang++").exists():
            arguments += [f"-DCMAKE_HIP_COMPILER={Path(args.rocm, 'llvm/bin/clang++')}", f"-DCMAKE_PREFIX_PATH={args.rocm}"]
        if args.arch:
            arguments.append(f"-DCMAKE_HIP_ARCHITECTURES={args.arch}")
    return arguments + args.cmake_arg


def built(build_dir, subdir, name):
    """A program the build made: build_dir/subdir/name, or under Release/
    with a multi-config generator (Visual Studio)."""
    for candidate in (build_dir / subdir / (name + EXE), build_dir / subdir / "Release" / (name + EXE)):
        if candidate.exists():
            return candidate
    raise Failure(f"the build made no {name}{EXE} in {build_dir / subdir}")


# --- Building and timing the trials -------------------------------------------

class Trials:
    """Builds the benchmarks for a configuration (kept, so the setting a
    search stands at is timed again without building it again), and times
    them."""

    def __init__(self, args, work):
        self.args = args
        self.work = work
        self.build_dir = work / "build"
        self.run_dir = work / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.programs = {}
        self.configured = False
        self.scale = 20
        self.candidates = 5_000_000_000

    def benchmarks(self, config):
        key = tuple(sorted(config.items()))
        if key in self.programs:
            return self.programs[key]
        tuning = ";".join(f"{name}={value}" for name, value in sorted(config.items()))
        arguments = [] if self.configured else configure_arguments(self.args, network=False) + ["-DNAMEBREAK_OPENCL=OFF"]
        say(f"  building with {describe(config, self.args.base)} ...")
        started = time.monotonic()
        run(["cmake", "-S", CLIENT, "-B", self.build_dir, *arguments, f"-DNAMEBREAK_TUNING={tuning}"], quiet=True)
        self.configured = True
        run(["cmake", "--build", self.build_dir, "--config", "Release", "--parallel", "--target", "search_bench", "dictionary_bench"], quiet=True)
        kept = self.work / "programs" / str(len(self.programs))
        kept.mkdir(parents=True)
        programs = tuple(shutil.copy2(built(self.build_dir, "tests", name), kept) for name in ("search_bench", "dictionary_bench"))
        say(f"  built in {time.monotonic() - started:.0f} s")
        self.programs[key] = programs
        return programs

    def time(self, program, arguments, pattern):
        """The rate (G candidates/s) `program` reports, or 0 if it fails."""
        result = subprocess.run([str(program), "--backend", self.args.gpu, *arguments], cwd=self.run_dir,
                                capture_output=True, text=True, errors="replace")
        found = re.search(pattern, result.stdout)
        if result.returncode != 0 or not found:
            lines = (result.stderr or result.stdout).strip().splitlines()
            say(f"    (failed: {lines[-1]})" if lines else "    (failed)")
            return 0.0
        return float(found[1])

    def row_rate(self, config, scale=None):
        search_bench = self.benchmarks(config)[0]
        return self.time(search_bench, ["--prune", "none", "--scale", str(scale or self.scale)], r"throughput: ([\d.]+) G")

    def dictionary_rate(self, config, candidates=None):
        dictionary_bench = self.benchmarks(config)[1]
        return self.time(dictionary_bench, ["--max-words", "3", "--candidates", str(candidates or self.candidates)],
                         r" - ([\d.]+) G candidates/s")


def tune(axes, start, measure):
    """One setting at a time, as tuneAxes (src/common/autotune.h) does."""
    current = dict(start)
    for _ in range(ROUNDS):
        changed = False
        for name, values, _ in axes:
            incumbent = current[name]

            def measure_with(value):
                rate = measure({**current, name: value})
                say(f"  {label(name)} {value}: " + (f"{rate:.1f} G candidates/s" if rate > 0 else "can't be used"))
                return rate

            first = measure_with(incumbent)
            rates = [(value, measure_with(value)) for value in values if value != incumbent]
            stands = (first + measure_with(incumbent)) / 2
            best, best_rate = max(rates, key=lambda pair: pair[1])
            if best_rate > stands * (1 + MIN_GAIN):
                current[name] = best
                changed = True
                say(f"  -> {label(name)} {best}")
            else:
                say(f"  -> {label(name)} stays {incumbent}")
        if not changed:
            break
    return current


def confirm(what, measure, base, tuned):
    """The defaults and `tuned` timed twice more each, in turn: `tuned` if it
    was more than MIN_GAIN faster, else the defaults."""
    default_rate = tuned_rate = 0.0
    for _ in range(2):
        default_rate += measure(base) / 2
        tuned_rate += measure(tuned) / 2
    better = tuned_rate > default_rate * (1 + MIN_GAIN)
    say(f"  {what}: defaults {default_rate:.1f}, tuned {tuned_rate:.1f} G candidates/s"
        + ("" if better else " - not enough faster, keeping the defaults"))
    return (tuned if better else base), default_rate, (tuned_rate if better else default_rate)


# --- The whole run --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tunes the CUDA/HIP backend's launch settings for this GPU and builds a client with them. "
                                                 "See README.md next to this script.")
    parser.add_argument("--gpu", choices=["cuda", "hip"], help="the backend to tune (default: cuda with an NVIDIA driver, else hip)")
    parser.add_argument("--arch", help="the GPU architecture to build for (default: CUDA's 'native', HIP's what ROCm says)")
    parser.add_argument("--seconds", type=float, default=3, help="about how long each timed search takes (default 3)")
    parser.add_argument("--output", type=Path, default=CLIENT / "build-tuned", help="where the tuned client is built (default client/build-tuned)")
    parser.add_argument("--work", type=Path, help="where the trial builds go (default: a temporary directory, removed afterwards)")
    parser.add_argument("--no-network", action="store_true", help="build the client without coordinator mode (and without libcurl)")
    parser.add_argument("--version", help="the release the source is, for the update check (default: the git tag it's checked out at, or dev)")
    parser.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm"), help="where ROCm is (default /opt/rocm)")
    parser.add_argument("--cmake-arg", action="append", default=[], help="one more argument for every CMake configure (repeatable)")
    parser.add_argument("--yes", action="store_true", help="don't ask before tuning on a GPU something else keeps busy")
    args = parser.parse_args()
    if not args.gpu:
        args.gpu = "cuda" if shutil.which("nvidia-smi") else "hip"

    check_tools(args)
    args.base = defaults()
    version = release_version(args)
    # Fails now, rather than after the tuning, if the client's own build can't be configured.
    configure_arguments(args, network=not args.no_network)
    check_gpu_idle(args)

    work = args.work or Path(tempfile.mkdtemp(prefix="namebreak-autotune-"))
    say(f"Tuning the {args.gpu} backend on this GPU, building in {work}: up to about 40 builds and 90 timed searches,\n"
        "which takes half an hour to an hour. Nothing else should use the GPU meanwhile. Ctrl+C stops it.\n")
    try:
        trials = Trials(args, work)
        base = args.base

        say("row search:")
        rate = trials.row_rate(base)
        if rate <= 0:
            raise Failure(f"search_bench can't search on the {args.gpu} backend with its defaults (see above)")
        # Heats the GPU up, and sizes the trials at the speed it then has.
        heat_scale = max(1, round(rate * 1e9 * HEAT_UP_SECONDS / SEARCH_BENCH_RANGE))
        say(f"  heating the GPU up for about {HEAT_UP_SECONDS} s ...")
        rate = trials.row_rate(base, heat_scale) or rate
        trials.scale = max(1, round(rate * 1e9 * args.seconds / SEARCH_BENCH_RANGE))
        row = tune(ROW_AXES, base, trials.row_rate)
        row, row_before, row_after = confirm("row search", trials.row_rate, base, row)

        say("\ndictionary search:")
        rate = trials.dictionary_rate(row)
        if rate <= 0:
            raise Failure(f"dictionary_bench can't search on the {args.gpu} backend (see above)")
        trials.candidates = max(1_000_000, round(rate * 1e9 * args.seconds))
        tuned = tune(DICTIONARY_AXES, row, trials.dictionary_rate)
        tuned, dictionary_before, dictionary_after = confirm("dictionary search", trials.dictionary_rate, row, tuned)
    except KeyboardInterrupt:
        say("\nStopped - nothing built.")
        return 1
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)

    changed = {name: value for name, value in tuned.items() if value != base[name]}
    say(f"\nrow search {row_before:.1f} -> {row_after:.1f}, dictionary search {dictionary_before:.1f} -> {dictionary_after:.1f} G candidates/s")
    say("The defaults were fastest." if not changed else "Fastest: " + describe(tuned, base))

    say(f"\nBuilding the client into {args.output} ...")
    tuning = ";".join(f"{name}={value}" for name, value in sorted(changed.items()))
    run(["cmake", "-S", CLIENT, "-B", args.output, *configure_arguments(args, network=not args.no_network),
         f"-DNAMEBREAK_TUNING={tuning}", f"-DNAMEBREAK_VERSION={version}"])
    targets = ["namebreak"] + (["namebreak-gui"] if WINDOWS and not args.no_network else [])
    run(["cmake", "--build", args.output, "--config", "Release", "--parallel", "--target", *targets])
    programs = [built(args.output, ".", target) for target in targets]
    (args.output / "tuning.txt").write_text(
        f"namebreak {version}, {args.gpu} backend, tuned by scripts/autotune/autotune.py on {time.strftime('%Y-%m-%d')}\n"
        f"NAMEBREAK_TUNING={tuning}\n"
        f"row search {row_before:.1f} -> {row_after:.1f} G candidates/s, dictionary search {dictionary_before:.1f} -> {dictionary_after:.1f}\n")
    say("\nDone. The tuned client: " + ", ".join(str(p) for p in programs))
    say("Use it in place of the release's: run it from the directory with your config.conf as before"
        + (" (keep the .dll files next to the .exe files)." if WINDOWS else "."))
    say(f"What it was tuned with is in {args.output / 'tuning.txt'}. Rebuild (run this again) when you update the source.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as failure:
        print(f"autotune: {failure}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(1)
