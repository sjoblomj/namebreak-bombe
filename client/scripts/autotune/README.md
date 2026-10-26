# Tuning the CUDA or HIP client for your GPU

The released client's CUDA and HIP backends were tuned on one GPU, an RTX
3080 Ti Laptop. Yours may search faster with other launch settings. Those
settings are compiled into the program, so `namebreak --autotune` (which
tunes the OpenCL and Metal backends from `config.conf`) can't change them.
`autotune.py` does it by building the client itself:

1. It builds the search benchmarks with each candidate setting and times
   them on your GPU: the row search first (rows per thread, batches per
   launch, the GPU window, threads per block), then the dictionary search
   (words per thread, threads per block).
2. It keeps the fastest, the way `--autotune` does. Each setting is tried
   in turn from the defaults, and a value is taken only if it is more than
   2% faster. At the end the defaults and the result are timed again, and
   the result is kept only if it is still more than 2% faster.
3. It builds a client with what it found, into `client/build-tuned/`.

That client behaves exactly like the release, coordinator mode included.
It still runs its self-tests at every start, so a setting can't make it
miss a name, only change how fast it searches. Expect half an hour to an
hour: each candidate setting is a build of the CUDA code, about 25 seconds
to a minute, then a few seconds of searching. On a laptop, plug it in.

Nothing else should use the GPU while it runs. Stop (or pause, with `p`)
any namebreak that's running, and don't game meanwhile: anything else on
the GPU makes the measurements meaningless.

Only Linux and Windows. On a Mac the client uses Metal, which
`namebreak --autotune` tunes without building anything.

## 1. Get the source code

Get the source of the release you run. The releases page lists them, and
this script is in those newer than v2026-10-22:

```sh
git clone https://github.com/sjoblomj/namebreak-bombe.git
cd namebreak-bombe
git checkout vYYYY-MM-DD        # the tag of the release you run
```

Or download the release's "Source code (zip)" from
<https://github.com/sjoblomj/namebreak-bombe/releases> and unpack it. A
checkout at a release's tag makes the tuned client know which release it
is, so it still says when a newer one is out. From a zip, pass
`--version vYYYY-MM-DD` (the release's tag) to the script for the same.

## 2. Install what it needs

### Linux, NVIDIA (CUDA)

- The NVIDIA driver. `nvidia-smi` should list your GPU.
- The CUDA Toolkit, 12.x: from <https://developer.nvidia.com/cuda-downloads>
  or your distribution (`nvidia-cuda-toolkit` on Debian/Ubuntu, `cuda` on
  Arch and Fedora with RPM Fusion). `nvcc --version` should work. If it
  isn't on the PATH, add `/usr/local/cuda/bin`.
- A C++ compiler, CMake 3.24 or later, libcurl's headers and Python 3.8 or
  later:
  - Debian/Ubuntu: `sudo apt install build-essential cmake libcurl4-openssl-dev python3 git`.
    If `cmake --version` says less than 3.24 (Ubuntu 22.04's does), use
    `pip install cmake` (or `pipx install cmake`).
  - Fedora: `sudo dnf install gcc-c++ cmake libcurl-devel python3 git`
  - Arch: `sudo pacman -S base-devel cmake curl python git`
- Optional: Ninja (`ninja-build` / `ninja`) builds a little faster, and the
  script uses it if it's there.

### Linux, AMD (HIP)

- ROCm 6.x, installed as AMD describes for your distribution
  (<https://rocm.docs.amd.com>). The script expects it in `/opt/rocm`; pass
  `--rocm <dir>` if it's elsewhere.
- The same compiler, CMake, libcurl and Python as above.

### Windows, NVIDIA (CUDA)

Install these in this order:

1. **Visual Studio 2022** (the free Community edition, or the "Build
   Tools for Visual Studio 2022") from <https://visualstudio.microsoft.com/downloads/>,
   with the **Desktop development with C++** workload. It includes CMake.
2. **The CUDA Toolkit 12.x** from <https://developer.nvidia.com/cuda-downloads>.
   Install it after Visual Studio, so that it adds its Visual Studio
   integration.
3. **Python 3** from <https://www.python.org/downloads/>. Tick "Add
   python.exe to PATH".
4. **Git** from <https://git-scm.com/download/win>.
5. **libcurl**, through vcpkg, in a command prompt:

   ```bat
   git clone https://github.com/microsoft/vcpkg C:\vcpkg
   C:\vcpkg\bootstrap-vcpkg.bat
   C:\vcpkg\vcpkg install curl opencl --triplet x64-windows
   setx VCPKG_ROOT C:\vcpkg
   ```

   The script finds vcpkg in `C:\vcpkg` or wherever `VCPKG_ROOT` says. To
   skip vcpkg, pass `--no-network`. The client is then built without
   coordinator mode, which you'd only want for searching on your own.

Then open a new **"x64 Native Tools Command Prompt for VS 2022"** (from the
Start menu), so that CMake, the compiler and the new `VCPKG_ROOT` are all
found.

## 3. Run it

From the source directory (the one with `client/` in it), with nothing else
on the GPU:

```sh
python3 client/scripts/autotune/autotune.py
```

On Windows: `py client\scripts\autotune\autotune.py`.

It says what it measures as it goes. Ctrl+C stops it, and nothing is built
then. At the end it prints where the tuned client is:

- Linux: `client/build-tuned/namebreak`
- Windows: `client\build-tuned\Release\namebreak.exe` and
  `namebreak-gui.exe`, with the `.dll` files they need next to them.

Use it in place of the release's, run from the directory with your
`config.conf` as before. On Windows, copy the `.dll` files along with the
`.exe` files. `client/build-tuned/tuning.txt` says what it was tuned with.
When you update to a newer release, check out its tag and run the script
again.

The defaults may well turn out fastest; then the client it builds is the
release's, built on your machine.

### Options

| Option | |
|---|---|
| `--gpu cuda` / `--gpu hip` | Which backend to tune. By default `cuda` if there's an NVIDIA driver, else `hip`. |
| `--arch <arch>` | The GPU architecture to build for. By default CUDA's `native` (the GPU in this machine) and, for HIP, what ROCm reports. |
| `--seconds <n>` | About how long each timed search takes (default 3). More is steadier, and slower. |
| `--output <dir>` | Where the tuned client is built (default `client/build-tuned`). |
| `--work <dir>` | Where the trial builds go, kept afterwards. By default a temporary directory, removed at the end. |
| `--no-network` | Build the client without coordinator mode, so libcurl isn't needed. |
| `--version <tag>` | The release the source is, for the update check. By default the tag it's checked out at, else `dev`. |
| `--cmake-arg <arg>` | One more argument for every CMake configure, e.g. `--cmake-arg=-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`. Repeatable. |
| `--yes` | Don't ask before tuning on a GPU that something else keeps busy. |

## How it works

The settings are the macros in `src/backends/cuda/tuning.h`, which CMake's
`NAMEBREAK_TUNING` sets for the search library:

```sh
cmake -S client -B build -DNAMEBREAK_TUNING="NAMEBREAK_ROWS_PER_THREAD=25;NAMEBREAK_GPU_WINDOW_CHARS=6"
```

The script builds `search_bench` and `dictionary_bench` this way into one
trial build directory (without the tests' extra configurations, OpenCL or
coordinator mode) and keeps a copy of each pair, so that a setting timed
again isn't built again. The two benchmarks are the real search code:

- `search_bench --prune none --scale <n>` times the real 49-character
  alphabet, ten characters long, nothing pruned.
- `dictionary_bench --max-words 3 --candidates <n>` times english-1, up to
  three words a candidate, with an encryption key.

Before the first trial, it searches for about 30 seconds with the defaults
to heat the GPU up, since a cool GPU searches faster until it's warm. The
trials are then sized to take `--seconds` each.

The GPU window is measured without pruning. In a real search a wider window
leaves the CPU a little less to prune (see the client README's
"`prune_symbol_runs`, ..." section), so a window that wins by only a few
percent here may come out even in practice.
