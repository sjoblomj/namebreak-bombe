NVCC    := nvcc
CXXSTD  := c++17
OPT     := -O3

# Target GPU architecture, as a compute capability (see `nvidia-smi --query-gpu=compute_cap
# --format=csv`). Defaults to sm_86 (confirmed for this machine's RTX 3080 Ti Laptop GPU) -
# override on the command line for a different GPU, e.g.: make ARCH=sm_75
ARCH    := sm_86

TARGET  := namebreak
# platform.cpp is always built: it isn't networking code - it holds the OS-
# specific terminal/keypress helpers behind the pause key (used in every mode),
# alongside the hostname/username lookups the coordinator client wants.
SOURCES := namebreak.cu cpu-utils.cpp config.cpp platform.cpp
HEADERS_CORE := cpu-utils.h constants.h search.h hash_kernels.cuh
HEADERS := cpu-utils.h constants.h search.h config.h hash_kernels.cuh platform.h

# Set to 0 to build without any networking: no coordinator client, HTTP or
# protocol code, no libcurl dependency, and `coordinator` mode is unavailable
# at runtime (the bounded and continuous modes are unaffected): make NETWORK=0
NETWORK ?= 1

# Windows here means a native build: MSVC (cl.exe) as nvcc's host compiler +
# the CUDA Toolkit for Windows - NVIDIA's actually-supported combo, as
# opposed to MinGW (which nvcc doesn't officially support as a host
# compiler, even though the plain C++ files happen to cross-compile fine
# with it - see platform.cpp). Run this from a Git Bash or MSYS2 shell with
# cl.exe on PATH (e.g. a "x64 Native Tools Command Prompt" that's also had
# one of those sourced) - a bare cmd.exe/PowerShell `make` isn't supported,
# since `rm` in `clean` below needs a real shell providing it. $(OS) is set
# to "Windows_NT" by the Windows environment itself (cmd.exe/PowerShell/Git
# Bash/MSYS2 all inherit it), so this doesn't rely on `uname` existing.
#
# NOTE: unlike the rest of this Makefile, the Windows branch below is
# untested - there's no Windows+CUDA machine available to build-verify it
# on. The source itself (namebreak.cu and friends) has been confirmed to
# actually cross-compile clean against real Windows headers (MinGW-w64,
# see platform.cpp/coordinator_runner.cpp) - it's specifically this
# Makefile's flag translation that's unverified. If something here doesn't
# line up with your actual toolchain/curl install, the two CURL_* variables
# below are the most likely thing to need adjusting.
ifeq ($(OS),Windows_NT)
    EXE_EXT := .exe
    # libcurl has no universal system location on Windows the way it does on
    # Linux (where it's just always on the default include/library search
    # path) - point these at wherever yours is, e.g. a vcpkg install:
    #   make CURL_INC=C:/vcpkg/installed/x64-windows/include \
    #        CURL_LIB_DIR=C:/vcpkg/installed/x64-windows/lib
    # CURL_LIB is the library name to link (no extension) - vcpkg's default
    # triplet produces libcurl.lib, hence "libcurl" here rather than Linux's
    # bare "curl" (-lcurl there finds libcurl.so).
    CURL_INC     ?=
    CURL_LIB_DIR ?=
    CURL_LIB     ?= libcurl
    # No pthread to link against on Windows - std::thread is backed by
    # native Windows threads there, unlike glibc/MinGW's POSIX threads.
    NETWORK_LDFLAGS = $(if $(CURL_LIB_DIR),-L$(CURL_LIB_DIR)) -l$(CURL_LIB)
else
    EXE_EXT :=
    CURL_INC     ?=
    CURL_LIB_DIR ?=
    NETWORK_LDFLAGS = -lcurl
    # std::thread (the pause key listener, in every mode) needs libpthread on
    # older glibc - not just the networked build.
    LDFLAGS += -lpthread
endif

ifeq ($(NETWORK),1)
SOURCES += coordinator_runner.cpp coordinator_client.cpp http_client.cpp protocol.cpp
HEADERS += coordinator_runner.h coordinator_client.h http_client.h protocol.h
NVCC_FLAGS += -DNAMEBREAK_WITH_NETWORK $(if $(CURL_INC),-I$(CURL_INC))
LDFLAGS += $(NETWORK_LDFLAGS)
endif

# A little native Win32 GUI wrapper around coordinator mode (gui_win32.cpp) -
# Windows-only (Win32 API), so `make gui`/this target simply doesn't exist on
# the Linux branch above. Always built with networking (it's coordinator-only,
# unaffected by NETWORK=0 above) and with namebreak.cu's own console main()
# excluded via -DNAMEBREAK_NO_MAIN - the same flag/precedent the test
# executables further down already use to get runSearch()/the CUDA kernels
# without a duplicate main() symbol. /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup
# gives the process no console while still letting gui_win32.cpp use an
# ordinary main() instead of WinMain() - see that file's own header comment.
# NOTE: like the Windows OS branch above, this is untested - no Windows+CUDA
# machine is available to build-verify it on.
ifeq ($(OS),Windows_NT)
GUI_TARGET  := namebreak-gui
GUI_SOURCES := gui_win32.cpp namebreak.cu cpu-utils.cpp config.cpp platform.cpp \
               coordinator_runner.cpp coordinator_client.cpp http_client.cpp protocol.cpp
GUI_LDFLAGS := $(NETWORK_LDFLAGS) -lcomctl32 -lshell32 -Xlinker /SUBSYSTEM:WINDOWS -Xlinker /ENTRY:mainCRTStartup

.PHONY: gui
gui: $(GUI_TARGET)$(EXE_EXT)

$(GUI_TARGET)$(EXE_EXT): $(GUI_SOURCES) $(HEADERS) logo_bmp.h icon_bmp.h
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN -DNAMEBREAK_WITH_NETWORK $(if $(CURL_INC),-I$(CURL_INC)) \
	    -o $(GUI_TARGET)$(EXE_EXT) $(GUI_SOURCES) $(GUI_LDFLAGS)
endif

.PHONY: all verbose clean test

all: $(TARGET)$(EXE_EXT)

$(TARGET)$(EXE_EXT): $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) $(NVCC_FLAGS) -o $(TARGET)$(EXE_EXT) $(SOURCES) $(LDFLAGS)

# Same build, but with ptxas' per-kernel register/spill/stack-frame report - useful when
# checking whether the kernel still fits comfortably within occupancy limits.
verbose: $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) $(NVCC_FLAGS) --ptxas-options=-v -o $(TARGET)$(EXE_EXT) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)$(EXE_EXT) $(GUI_TARGET)$(EXE_EXT) tests/prefix_hasher_test tests/prefix_hasher_bench tests/search_bench $(INTEGRATION_TESTS) tests/search_overflow_test
	rm -rf tests/.testrun

# Pure-CPU correctness check for IncrementalPrefixHasher (cpu-utils.h/.cpp) -
# no GPU involved, plain g++ build.
tests/prefix_hasher_test: tests/prefix_hasher_test.cpp cpu-utils.cpp cpu-utils.h constants.h
	g++ -std=$(CXXSTD) -O2 -o tests/prefix_hasher_test tests/prefix_hasher_test.cpp cpu-utils.cpp

# Pure-CPU benchmark isolating IncrementalPrefixHasher's own speedup (O(1)
# amortized vs. from-scratch rehashing per leadingIdx step) from the separate
# GPU-window-size question - no GPU involved.
tests/prefix_hasher_bench: tests/prefix_hasher_bench.cpp cpu-utils.cpp cpu-utils.h constants.h
	g++ -std=$(CXXSTD) -O2 -o tests/prefix_hasher_bench tests/prefix_hasher_bench.cpp cpu-utils.cpp

prefix_hasher_bench: tests/prefix_hasher_bench
	./tests/prefix_hasher_bench

# End-to-end correctness check for runSearch() itself (namebreak.cu, linked
# in directly, main() excluded via -DNAMEBREAK_NO_MAIN) against a CPU
# brute-force reference - see the file for why this is a separate concern
# from prefix_hasher_test above. Built without NETWORK (doesn't need it).
#
# The same test source is built several times with different compile-time
# tuning values (constants.h), because they change which code paths a
# search exercises - every case derives its geometry from those values:
#   (default)        the shipped configuration
#   _small_launch    3 rows per launch: forces many launches per leading value,
#                    so the chunk boundaries are actually exercised
#   _window2         a 2-character GPU window (5 rows per launch): a much
#                    longer CPU-side leading part, and a shallower row decode
#   _window6         the widest supported window: 5 prefix digits per row and
#                    trailing indices beyond 32 bits
INTEGRATION_DEPS := tests/search_integration_test.cu namebreak.cu cpu-utils.cpp $(HEADERS_CORE)
INTEGRATION_TESTS := tests/search_integration_test tests/search_integration_test_small_launch \
                     tests/search_integration_test_window2 tests/search_integration_test_window6

tests/search_integration_test: $(INTEGRATION_DEPS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN -o $@ tests/search_integration_test.cu namebreak.cu cpu-utils.cpp

tests/search_integration_test_small_launch: $(INTEGRATION_DEPS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN -DNAMEBREAK_ROWS_PER_LAUNCH=3 -o $@ tests/search_integration_test.cu namebreak.cu cpu-utils.cpp

tests/search_integration_test_window2: $(INTEGRATION_DEPS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN -DNAMEBREAK_GPU_WINDOW_CHARS=2 -DNAMEBREAK_ROWS_PER_LAUNCH=5 -o $@ tests/search_integration_test.cu namebreak.cu cpu-utils.cpp

tests/search_integration_test_window6: $(INTEGRATION_DEPS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN -DNAMEBREAK_GPU_WINDOW_CHARS=6 -o $@ tests/search_integration_test.cu namebreak.cu cpu-utils.cpp

# What runSearch() does when a launch has more hashA hits than MAX_MATCHES can
# record - built with MAX_MATCHES=1 so a single hashA collision pair overflows it.
tests/search_overflow_test: tests/search_overflow_test.cu namebreak.cu cpu-utils.cpp $(HEADERS_CORE)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN -DMAX_MATCHES=1 -o $@ tests/search_overflow_test.cu namebreak.cu cpu-utils.cpp

# Each configuration takes about a minute to compile (the search kernel is
# instantiated per alphabet size and suffix length) - `make -j test` builds
# them in parallel.
test: tests/prefix_hasher_test $(INTEGRATION_TESTS) tests/search_overflow_test
	./tests/prefix_hasher_test
	mkdir -p tests/.testrun
	cd tests/.testrun && ../search_integration_test
	cd tests/.testrun && ../search_integration_test_small_launch
	cd tests/.testrun && ../search_integration_test_window2
	cd tests/.testrun && ../search_integration_test_window6
	cd tests/.testrun && ../search_overflow_test

# Real-code throughput check - times the actual runSearch() (not a proxy
# kernel) over a ~28.8B-candidate range. WINDOW controls
# namebreak.cu's gpuWindowChars for this build only (make WINDOW=6
# search_bench), to re-sweep that tuning value against real hardware without
# hand-editing the source - see constants.h. Unset, the shipped default is used.
# No dependency tracking on WINDOW itself (matches this Makefile's existing
# NETWORK/etc. convention) - combine with `make clean` first if switching it.
WINDOW ?=
tests/search_bench: tests/search_bench.cu namebreak.cu cpu-utils.cpp hash_kernels.cuh cpu-utils.h constants.h search.h
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -DNAMEBREAK_NO_MAIN $(if $(WINDOW),-DNAMEBREAK_GPU_WINDOW_CHARS=$(WINDOW)) -o tests/search_bench tests/search_bench.cu namebreak.cu cpu-utils.cpp

search_bench: tests/search_bench
	mkdir -p tests/.testrun
	cd tests/.testrun && ../search_bench
