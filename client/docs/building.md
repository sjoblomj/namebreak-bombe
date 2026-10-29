# Building namebreak

Requires CMake (3.24 or later) and libcurl, plus the CUDA Toolkit (`nvcc`)
for the `default` preset, which builds the CUDA backend. Without one - on a
Mac, or for OpenCL and the CPU only - use the `portable` preset instead
(`cmake --preset portable` and so on, into `build-portable/`), and for AMD
GPUs the `hip` preset (see `NAMEBREAK_GPU` below). The build is described
by `CMakeLists.txt`, and `CMakePresets.json` has those configurations -
CLion and Visual Studio pick them up directly.

```sh
cmake --preset default            # configure into build/
cmake --build --preset default -j "$(nproc)"   # builds build/namebreak, and the tests in build/tests/
ctest --preset default            # runs the tests
```

See [Tests and benchmarks](testing.md) for what the tests cover.

`-j` sets how many files are compiled at once - on a Mac, `-j "$(sysctl -n
hw.ncpu)"`. The presets leave it out on purpose: with make, a bare `-j` has
no limit, and the dozen builds of the search library the tests use then
start hundreds of compilers at once, enough to run out of memory. Without
`-j`, make compiles one file at a time; Ninja and Visual Studio pick a number
themselves.

On a machine with a GPU, add `-DNAMEBREAK_REQUIRE_GPU=ON` to the first
command, so that tests fail rather than skip if the driver loses the GPU
(see the options below).

Options, passed to the first command as `-D<option>=<value>`:

| Option | Default | Meaning |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `86` | The GPU's compute capability, without the dot (`nvidia-smi --query-gpu=compute_cap --format=csv`), or `native` |
| `NAMEBREAK_NETWORK` | `ON` | `OFF` leaves out coordinator mode, the update check and the libcurl dependency |
| `NAMEBREAK_VERSION` | `dev` | The release this build is, e.g. `v2026-09-26` - what `--version` prints, and what the update check and the coordinator compare. The release workflow sets it to the tag it builds |
| `NAMEBREAK_GPU` | `cuda` | `hip` builds the HIP backend for AMD GPUs instead (needs ROCm or the HIP SDK; the `hip` preset, into `build-hip/`, with `CMAKE_HIP_ARCHITECTURES` e.g. `gfx1100` - unset, CMake asks ROCm). `none` builds without either, leaving the CPU backends and OpenCL - or use the `portable` preset, which builds into `build-portable/` |
| `NAMEBREAK_METAL` | `AUTO` | The Metal backend: `AUTO` builds it on macOS (Xcode's command line tools are all it needs), `OFF` leaves it out |
| `NAMEBREAK_OPENCL` | `AUTO` | The OpenCL backend: `AUTO` builds it if OpenCL's headers and loader library are found (on Debian/Ubuntu: `opencl-headers ocl-icd-opencl-dev`; on Windows e.g. vcpkg's `opencl`), `ON` insists, `OFF` leaves it out |
| `NAMEBREAK_BENCH_WINDOW` | *(empty)* | A different GPU window for `search_bench` only, e.g. `6` |
| `NAMEBREAK_TUNING` | *(empty)* | The CUDA/HIP backend's compile-time launch settings (`src/backends/cuda/tuning.h`), `;`-separated, e.g. `NAMEBREAK_ROWS_PER_THREAD=25;NAMEBREAK_GPU_WINDOW_CHARS=6` - what [`scripts/autotune/`](../scripts/autotune/README.md) finds the fastest of for a GPU |
| `NAMEBREAK_TEST_VARIANTS` | `ON` | `OFF` leaves out the tests' extra tuning configurations (see [Tests and benchmarks](testing.md)), which multiply the build time and only matter on a machine with the GPUs to run them on. The release builds turn them off |
| `NAMEBREAK_REQUIRE_GPU` | `OFF` | `ON` makes `ctest` fail, rather than skip, the tests of a backend that can't run on the machine - for a machine with the GPUs the build is for, where that means a broken driver (see [Backends](backends.md#backends)) |

On Windows, use a Visual Studio developer prompt (MSVC is the host compiler
the CUDA Toolkit supports there) and point CMake at a libcurl, e.g. vcpkg's:
`cmake --preset default -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`.
That also builds the GUI, `namebreak-gui.exe`. There's no Windows machine
with CUDA to verify this on; the CPU-only build has been cross-compiled with
MinGW-w64, and its tests pass under Wine.

Every backend searches an alphabet of any size from 1 to `MAX_ALPHABET_SIZE`
(63, as a row's candidates are a 64-bit mask, one bit each). The CUDA
kernel takes the size at runtime, dividing by it with
multipliers worked out on the host, the way a compiler divides by a
constant - except for 40, 42 and 43 (`CompiledAlphabetSizes` in
`src/backends/cuda/cuda_backend.cu`), compiled in: a size the compiler
knows saves it a few instructions a row (see
[PERFORMANCE.md](../PERFORMANCE.md)).

## Releases

Pushing a version tag (`git tag v2026-09-26 && git push origin v2026-09-26`)
runs `.github/workflows/client-release.yml`, which builds the client for Linux
(CUDA + OpenCL + CPU, and a separate AMD/HIP build), Windows (CUDA + OpenCL +
CPU, and the GUI) and macOS on Apple Silicon (Metal + OpenCL + CPU), runs the
tests on each, and attaches the archives to a GitHub release for the tag. A
build that fails is left out and the release is created as a draft instead.
Tags must be the release date, `vYYYY-MM-DD`, with `.1`, `.2`, ... added for
a later release the same day: that's how releases are ordered, both by the
client's update check and by the coordinator's minimum release. Each build
is made with `NAMEBREAK_VERSION` set to the tag. The
workflow can also be started by hand from the Actions tab, to get the same
builds (as `dev`) without a release. The runners have no GPUs, so only the CPU
backends' tests actually run there.
