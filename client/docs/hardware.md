# Choosing and tuning the hardware

The `[hardware]` section says which parts of the machine namebreak searches
with, and how - in every mode. Every key is optional.

```ini
[hardware]
gpus = 0, 2
cpu_threads = 8
```

`gpus` picks the GPUs, by the numbers `namebreak --list-gpus` shows for
each backend (the same card can have another number in another backend):

```
$ namebreak --list-gpus
cuda:
  0: NVIDIA GeForce RTX 3080 Ti Laptop GPU
opencl:
  0: Intel(R) Iris(R) Xe Graphics (integrated) - not selected
  1: NVIDIA GeForce RTX 3080 Ti Laptop GPU
```

- `auto` (the default): every discrete GPU of the backend that searches -
  or its integrated one, if it has nothing else. An integrated GPU shares
  the CPU's power and memory, and the CPU prepares the discrete GPUs' work,
  so next to a discrete GPU it would mostly slow that one down.
- `all`: every one, integrated or not.
- GPU numbers separated by commas, e.g. `0, 2`: those.
- `none`: no GPU at all - the cpu backend searches.

The CUDA and OpenCL backends number the GPUs in the order they sit on the
PCI bus (as `nvidia-smi` does), so a number stays the same card from one
run to the next, through driver updates too - until a card is added, taken
out or moved to another slot. OpenCL GPUs whose driver doesn't say where
they sit come after those, in the order the drivers list them; Metal lists
the system's default GPU first. When a GPU `gpus` picks by number isn't the
card its `gpu<N>.name` (below) names, the search says so, and
`--list-gpus` marks it.

`coordinator` mode searches on every GPU it picks at once: one worker each,
with a backend of its own, registered with the coordinator as a client of
its own - the machine's `hostname` for GPU 0, `<hostname>/gpu<N>` for GPU N,
so that the server sizes each GPU's ranges by how fast that GPU searches -
each claiming, searching and reporting ranges of its own, and writing its
own matches file (`matches-<target>-gpu<N>.txt` for GPU N). They pause,
resume and quit together; *finish the current range, then pause* waits for
every GPU's range. Their messages say which GPU's they are
(`[coordinator gpu1] starting range ...`), and the Windows GUI shows each
in a cell of its own - target, status line, progress bar and search log -
laid out in a grid, or, with more GPUs than fit on the screen (or more than
three by three), listed down the left with the chosen one's cell beside
the list. A GPU whose backend can't be made, or fails its self-test, is
left out, and the others search.
The other modes search one range on one GPU, the first one `gpus` picks,
and say so when it picks more. Each GPU's worker builds a meet-in-the-middle
target's table of its own, so `meet_in_the_middle_table_memory` is per GPU
(see [Meet-in-the-middle mode](meet-in-the-middle.md#in-coordinator-mode)).

`cpu_threads` is how many threads the cpu backend searches with - by
default one per hardware thread, every core.

The OpenCL and Metal backends compile their kernels when a search starts,
so how they cut up the work can be set here too. An unset key is the
backend's default - measured on one RTX 3080 Ti Laptop, so another GPU may
well do better with other values. None of them changes what a search
finds; the backend still runs its self-tests with them before it searches.
The start of a search shows what it uses (`OpenCL launches: ...`). The CUDA
backend compiles its kernels ahead of time, with these values built into
them (`src/backends/cuda/tuning.h`), so it takes none of them: to tune
those, [`scripts/autotune/`](../scripts/autotune/README.md) builds the client
with each setting on the machine and keeps the fastest.

| Key | Default (OpenCL / Metal) | Range | What it is |
|---|---|---|---|
| `rows_per_thread` | 64 / 25 | 1-64 | About how many rows of a row group one work-item searches; 64 is a whole group, whatever the alphabet. |
| `batches_per_launch` | 16 / - | 1-32 | How many batches one launch searches at most - OpenCL only, Metal launches one. |
| `window_chars` | 5 / 5 | 1-6 | How many trailing characters the GPU enumerates (see [The GPU doesn't brute-force the whole candidate](design.md#the-gpu-doesnt-brute-force-the-whole-candidate)). |
| `work_group_size` | 256 / 256 | 64-1024 | Work-items (threads) per work-group (threadgroup) of a row search - fewer if the device allows fewer. |
| `dictionary_words_per_thread` | 32 / 32 | 1-256 | How many cells (words) of a batch one work-item hashes in a dictionary search. |
| `dictionary_work_group_size` | 256 / 256 | 1-1024 | Work-items per work-group of a dictionary search - fewer if the device allows fewer. |
| `dictionary_tails_per_cell` | 32 / 32 | 1-256 | How many tails a dictionary cell has (with `tails` only). |

Written as they are, they're for every GPU. `gpu<N>.<key>` is GPU N's own,
over those - for a machine with two different cards, say - and
`gpu<N>.name` the GPU they're meant for: should another card have number N
now, it doesn't get them, and the search says so.

```ini
[hardware]
window_chars = 6
gpu1.name = AMD Radeon RX 7900 XTX
gpu1.rows_per_thread = 25
```

Rather than trying values by hand, `namebreak --autotune` (with
`--config <file>`, and `--backend opencl|metal` to tune one other than the
one a search would use) measures them on each GPU `gpus` picks, one after
the other, and writes the fastest to `[hardware]` as each GPU's own, with
its name. A GPU of the same name as one already measured gets its values
without being measured itself. On a machine that searches with CUDA or HIP
it says so, and points at `scripts/autotune/`. It times a
search like `search_bench`'s (the real 49-character alphabet, nothing
pruned) for the row search keys, and one like `dictionary_bench`'s
(english-1, with a key) for the dictionary ones, about two seconds a trial
once half a minute of searching has heated the GPU up (a laptop's searches
up to three times as fast while it's cool), each key in turn from the
defaults. It then times the defaults and its result again, and keeps the
result only if it was more than 2% faster: the keys that differ from the
defaults, taking out any equal to them, so a later release's defaults
still apply - unless the section sets that key for every GPU, which the
default is then written over. It doesn't touch `dictionary_tails_per_cell`,
which only searches with tails would show, or any other line of the file.
It takes a few minutes a GPU; anything else on the GPU meanwhile - a
running namebreak, a game - makes its measurements meaningless. Ctrl+C
stops it without writing anything. Two caveats: it measures the window
without pruning, though a wider one leaves the CPU less to prune (see
[`prune_symbol_runs`, ...](pruning.md#prune_symbol_runs-prune_unopened_brackets-and-max_backslash_count-only-run-on-the-cpu)),
and a work-group size beyond what the device allows runs as the largest it
does.

In the Windows GUI, the setup dialog's **Hardware...** button (at first
run, or from the tray menu's `Settings...`) opens all of `[hardware]` in a
window of its own. It offers the CPU, with its thread count, or the GPUs
of the backend that searches, each with a checkbox - never both. An
integrated GPU next to a discrete one starts unchecked, as `gpus = auto`
leaves it, and says why; one that is the machine's only GPU starts checked.
With OpenCL or Metal, each GPU's launch settings are there too, to change
by hand, each with its default beside it, and shown in the system's
selection colors while it isn't the default; every value's `?` says what
it is. Under the GPUs is the memory a coordinator run gives
meet-in-the-middle work (see [below](meet-in-the-middle.md#in-coordinator-mode)). OK writes only
what was changed - `gpus` only if the choice of GPUs changed, a GPU's
`gpu<N>.*` keys only if one of its values did, with a value that is the
default taken out. Whatever doesn't fit on the screen scrolls.

Its **Optimize for hardware** button does what `--autotune` does, on the
GPUs checked, with a progress bar, an estimate of the time left, what it's
measuring, and a Cancel button, and shows what it found. A search already
running is paused while it measures and resumes afterwards. What it saves,
and what the Hardware window's OK saves, is kept whether the setup dialog
is then closed with OK or Cancel. The launch settings are used:

- at first run, as soon as the search starts;
- in a coordinator run already going, after the current range finishes:
  the client makes its backend again with the new settings, and self-tests
  it, before it claims the next one (and carries on with the old backend,
  saying so, should that fail) - each GPU's worker after its own range;
- in a local search already going, once namebreak is restarted.

Which GPUs search, or the CPU, is used as soon as the search starts at
first run, and otherwise once namebreak is restarted.

On a machine that searches with CUDA or HIP, it says why it can't, and
offers to open `scripts/autotune/`'s page.
