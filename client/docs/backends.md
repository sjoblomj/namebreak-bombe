# Source layout and backends

| Directory | What's in it |
|---|---|
| `src/engine/` | The search itself: `runSearch()`, candidate/bound arithmetic, hashing on the CPU; the dictionary search, `runDictionarySearch()`, with its word lists; and the meet-in-the-middle search, `runMeetInTheMiddleSearch()`, with the hash run backwards |
| `src/backends/` | What the search runs its batches on (see Backends below) |
| `src/common/` | The config file, and the few OS-specific helpers (terminal, hostname) |
| `src/net/` | The coordinator client: HTTP, the wire protocol, the claim/heartbeat loop, and a dictionary target's word lists and the basenames waiting to be sent (left out by `NAMEBREAK_NETWORK=OFF`) |
| `src/cli/` | The console program's `main()` |
| `src/gui/win32/` | The Windows GUI (Windows only) |
| `tests/` | Correctness tests and benchmarks (`ctest`, `run_search_bench`), and the mutation experiment (`mutation_test.py`, `run_mutation_test`) |
| `scripts/` | The generator for the GUI's embedded icon |
| `data/` | `english-1`, the dictionary compiled into the program |

Includes are written relative to `src/` (e.g. `#include "engine/search.h"`).

## Backends

`runSearch()` (`src/engine/search.cpp`) walks the search space and does
everything that isn't hashing - validation, bounds, the leading/trailing
split, pruning, pause/abort, writing matches. It hands each chunk of
candidates to a *backend* (`SearchBackend`, `src/engine/backend.h`), which
hashes them and returns the hits. `src/backends/backends.cpp` lists the
backends a build has, most preferred first; unless told otherwise (see
[Modes](../README.md#modes)) the program uses the first one that can run on the machine,
so a GPU build still works on a machine without that GPU:

- `cuda/` - the CUDA kernels and the code that launches them, with [the
  lookup filter](lookup-filter.md)
  (`common/lowbits_filter.h`) and row groups split into chunks: about
  1,450-1,700 G candidates/s on the RTX 3080 Ti Laptop, depending on how
  hot it is.
- `hip` - AMD GPUs: `cuda/`'s code compiled with ROCm's HIP instead, which
  accepts CUDA's kernel syntax as is; `gpu_runtime.h` maps the handful of
  CUDA runtime calls onto HIP's. So it has everything the CUDA backend has,
  lookup filter and row groups included. Not built by default, and not yet
  tested on an AMD GPU: it has only been run through HIP's NVIDIA mapping,
  on an NVIDIA GPU, where every test passes, stress tests included (see
  [PERFORMANCE.md](../PERFORMANCE.md) for what that does and doesn't cover).
- `metal/` - the Mac's GPU (Apple Silicon, or an Intel Mac's), through
  Metal: the OpenCL kernel, lookup filter, row groups and all, in Metal's
  shading language (`search.metal`), compiled by Metal at runtime the same way; the
  host side is Objective-C++ (`metal_backend.mm`). On a Mac, use the
  `portable` preset. Not yet run on a Mac: its kernel has only been tested
  on the CPU, compiled as C++ against a stand-in for Metal's standard
  library and run one GPU thread at a time behind a C++ transcription of
  `metal_backend.mm`, where every test configuration, the stress tests and
  the kernel's mutation experiment pass; `metal_backend.mm` itself has
  never been compiled. Its dictionary kernel (`dictionary.metal`) likewise:
  run on the CPU, each threadgroup's threads around its barrier, behind a
  C++ transcription of the `.mm`'s dictionary search, where the dictionary
  self-test and every `dictionary-search` configuration pass, and planted
  kernel bugs fail them. And its meet-in-the-middle kernel
  (`meet_in_the_middle.metal`) the same way: compiled for each search's
  alphabet and rules as Metal would, behind a C++ transcription of the
  `.mm`'s meet-in-the-middle search, where the meet-in-the-middle self-test
  and every `meet-in-the-middle-search` configuration pass, and the
  kernel's mutations fail them.
- `opencl/` - any GPU with an OpenCL driver: AMD, Intel (including
  integrated ones) and NVIDIA, with nothing but the vendor's regular driver
  installed. CUDA's kernel ported (`search.cl`), lookup filter, row groups,
  16 leading values per launch and all - with a whole row group per
  work-item, which suits it best; it's
  compiled by the driver at runtime, once per alphabet size, suffix length
  and trailing length, so it takes an alphabet of any size - drivers cache
  the result, but a new combination's first search starts a moment later.
  About 1,300-1,460 G candidates/s on the RTX 3080 Ti Laptop, depending on
  how hot it is - six to seven times what it did before it got the filter, and
  close to the CUDA backend.
- `cpu/` - every core of the CPU, with the same row trick and [lookup
  filter](lookup-filter.md) as the GPU
  kernels: a row's shared characters hashed once, a table lookup, and only
  the few candidates it flags hashed in full. The rows that differ only in
  their last shared character go on from the same state, which it takes
  once, and hashes them eight at a time - and the candidates the filter
  lets through eight at a time too. On an x86-64 CPU with AVX2, both are
  done with AVX2, whatever the build's flags, used only when the CPU has it
  (`-DNAMEBREAK_CPU_SIMD=0` forces the plain loops). It searches 16 leading
  values per call, their rows shared out among the threads as they go.
  About 59-74 G candidates/s on a laptop i9-12900H, 20 threads (it was
  about 10.6 hashing every candidate with AVX2, before the filter) - far
  below its GPU, but it lets any machine contribute. Every build has it.
- `reference/` - one thread, every candidate hashed from scratch. Far too
  slow for real searches, but simple enough to be obviously right: it's what
  the others are held to. Every build has it.

`ctest` runs the end-to-end tests against every backend in the build; a
test whose backend can't run on the machine is reported as skipped (one
whose backend fails its self-test on the machine fails). Skipped tests
still leave `ctest` at "100% tests passed", so a GPU the driver has lost -
CUDA after a laptop's suspend, say - would turn every GPU test green without
running any: on a machine with the GPUs, configure with
`-DNAMEBREAK_REQUIRE_GPU=ON`, and those tests fail instead. A new
backend implements `SearchBackend`, gets a directory under `src/backends/`
and an entry in `backends.cpp` and `CMakeLists.txt`, and is covered by the
same tests: they take a `--backend <name>` argument.
