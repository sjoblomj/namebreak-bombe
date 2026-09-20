NVCC    := nvcc
CXXSTD  := c++17
OPT     := -O3

# Target GPU architecture, as a compute capability (see `nvidia-smi --query-gpu=compute_cap
# --format=csv`). Defaults to sm_86 (confirmed for this machine's RTX 3080 Ti Laptop GPU) -
# override on the command line for a different GPU, e.g.: make ARCH=sm_75
ARCH    := sm_86

# Everything this Makefile produces goes under build/: the binaries at its top
# level, the tests in build/tests/, and object files in build/obj/<ARCH>/
# <config>/ - one directory per set of compile flags (see "Build
# configurations" below), so switching between them never mixes objects
# compiled with different flags.
BUILD   := build
OBJ     := $(BUILD)/obj/$(ARCH)
TESTRUN := $(BUILD)/testrun

# Set to 0 to build without any networking: no coordinator client, HTTP or
# protocol code, no libcurl dependency, and `coordinator` mode is unavailable
# at runtime (the bounded and continuous modes are unaffected): make NETWORK=0
NETWORK ?= 1

# Windows here means a native build: MSVC (cl.exe) as nvcc's host compiler +
# the CUDA Toolkit for Windows - NVIDIA's actually-supported combo, as
# opposed to MinGW (which nvcc doesn't officially support as a host
# compiler, even though the plain C++ files happen to cross-compile fine
# with it - see src/common/platform.cpp). Run this from a Git Bash or MSYS2
# shell with cl.exe on PATH (e.g. a "x64 Native Tools Command Prompt" that's
# also had one of those sourced) - a bare cmd.exe/PowerShell `make` isn't
# supported, since `rm`/`mkdir -p` below need a real shell providing them.
# $(OS) is set to "Windows_NT" by the Windows environment itself
# (cmd.exe/PowerShell/Git Bash/MSYS2 all inherit it), so this doesn't rely on
# `uname` existing.
#
# NOTE: unlike the rest of this Makefile, the Windows branch below is
# untested - there's no Windows+CUDA machine available to build-verify it
# on. The source itself has been confirmed to actually cross-compile clean
# against real Windows headers (MinGW-w64, see src/common/platform.cpp and
# src/net/coordinator_runner.cpp) - it's specifically this Makefile's flag
# translation that's unverified. If something here doesn't line up with your
# actual toolchain/curl install, the two CURL_* variables below are the most
# likely thing to need adjusting.
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
    # nvcc's dependency-file generation is only relied on with gcc as the host
    # compiler; on Windows, `make clean` after editing a header.
    DEPFLAGS :=
else
    EXE_EXT :=
    CURL_INC     ?=
    CURL_LIB_DIR ?=
    NETWORK_LDFLAGS = -lcurl
    # std::thread (the pause key listener, in every mode) needs libpthread on
    # older glibc - not just the networked build.
    LDFLAGS += -lpthread
    # Writes a .d file next to each object listing the headers it included,
    # so editing a header rebuilds exactly what depends on it.
    DEPFLAGS := -MMD -MP
endif

COMPILE := $(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -Isrc

# --- Sources ---------------------------------------------------------------
#
# src/engine    the search itself: runSearch(), candidate/bound arithmetic, hashing
# src/backends  the GPU code (CUDA)
# src/common    config file, OS-specific helpers
# src/net       the coordinator client (NETWORK=1 only)
# src/cli       the console program's main()
# src/gui       the Windows GUI's main() and windows (Windows only)
ENGINE_SRC := src/engine/mpq_hash.cpp src/engine/candidate.cpp
CUDA_SRC   := src/backends/cuda/cuda_backend.cu
# platform.cpp is always built: it isn't networking code - it holds the OS-
# specific terminal/keypress helpers behind the pause key (used in every mode),
# alongside the hostname/username lookups the coordinator client wants.
COMMON_SRC := src/common/config.cpp src/common/platform.cpp
NET_SRC    := src/net/coordinator_runner.cpp src/net/coordinator_client.cpp src/net/http_client.cpp src/net/protocol.cpp
CLI_SRC    := src/cli/main.cpp
GUI_SRC    := src/gui/win32/gui_win32.cpp

# What the search needs, and nothing else - all the tests link.
SEARCH_SRC := $(ENGINE_SRC) $(CUDA_SRC)

APP_SRC := $(CLI_SRC) $(SEARCH_SRC) $(COMMON_SRC)
ifeq ($(NETWORK),1)
APP_SRC += $(NET_SRC)
APP_CONFIG := app
else
APP_CONFIG := app-nonet
endif

# --- Build configurations --------------------------------------------------
#
# Each configuration is a set of extra compile flags; every source is
# compiled separately per configuration it's used in, into
# $(OBJ)/<config>/<source path>.o. The tests use several (see below),
# because the flags change which code paths a search exercises.
NETWORK_CFLAGS := -DNAMEBREAK_WITH_NETWORK $(if $(CURL_INC),-I$(CURL_INC))
CFLAGS_app          := $(NETWORK_CFLAGS)
CFLAGS_app-nonet    :=
CFLAGS_gui          := $(NETWORK_CFLAGS)
CFLAGS_test         :=
CFLAGS_small_launch := -DNAMEBREAK_ROWS_PER_LAUNCH=3
CFLAGS_window2      := -DNAMEBREAK_GPU_WINDOW_CHARS=2 -DNAMEBREAK_ROWS_PER_LAUNCH=5
CFLAGS_window6      := -DNAMEBREAK_GPU_WINDOW_CHARS=6
CFLAGS_overflow     := -DMAX_MATCHES=1
# WINDOW controls the GPU window for search_bench only (make WINDOW=6
# search_bench), to re-sweep that tuning value against real hardware without
# hand-editing the source - see src/engine/constants.h. Unset, the shipped
# default is used.
WINDOW ?=
BENCH_CONFIG := bench$(if $(WINDOW),-window$(WINDOW))
CFLAGS_$(BENCH_CONFIG) := $(if $(WINDOW),-DNAMEBREAK_GPU_WINDOW_CHARS=$(WINDOW))

CONFIGS := app app-nonet gui test small_launch window2 window6 overflow $(BENCH_CONFIG)

# $(call objs,<config>,<sources>)
objs = $(patsubst %,$(OBJ)/$(1)/%.o,$(2))

define CONFIG_RULES
$(OBJ)/$(1)/%.cu.o: %.cu
	@mkdir -p $$(@D)
	$(COMPILE) $$(CFLAGS_$(1)) $(DEPFLAGS) -c -o $$@ $$<
$(OBJ)/$(1)/%.cpp.o: %.cpp
	@mkdir -p $$(@D)
	$(COMPILE) $$(CFLAGS_$(1)) $(DEPFLAGS) -c -o $$@ $$<
endef
$(foreach c,$(CONFIGS),$(eval $(call CONFIG_RULES,$(c))))

# --- The program -------------------------------------------------------------

TARGET := $(BUILD)/namebreak$(EXE_EXT)
APP_LDFLAGS := $(LDFLAGS) $(if $(filter 1,$(NETWORK)),$(NETWORK_LDFLAGS))

.PHONY: all verbose clean test search_bench prefix_hasher_bench

all: $(TARGET)

$(TARGET): $(call objs,$(APP_CONFIG),$(APP_SRC)) $(BUILD)/app-config
	$(COMPILE) -o $@ $(filter %.o,$^) $(APP_LDFLAGS)

# Holds the name of the configuration build/namebreak was last linked from,
# and is only rewritten when that changes - so switching NETWORK relinks it,
# even though the other configuration's objects may all be older than it.
$(BUILD)/app-config: FORCE
	@mkdir -p $(@D)
	@echo '$(APP_CONFIG)' | cmp -s - $@ || echo '$(APP_CONFIG)' > $@

.PHONY: FORCE
FORCE:

# Compiles the CUDA backend with ptxas' per-kernel register/spill/stack-frame
# report - useful when checking whether the kernel still fits comfortably
# within occupancy limits. Nothing is written but a throwaway object file.
verbose:
	@mkdir -p $(OBJ)
	$(COMPILE) $(CFLAGS_$(APP_CONFIG)) --ptxas-options=-v -c -o $(OBJ)/verbose.o $(CUDA_SRC)

clean:
	rm -rf $(BUILD)

# A little native Win32 GUI wrapper (src/gui/win32) - Windows-only (Win32
# API), so `make gui`/this target simply doesn't exist on the Linux branch
# above. Always built with networking (unaffected by NETWORK=0 above) and
# without the console main() (src/cli/main.cpp) - the GUI has its own.
# /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup gives the process no console while
# still letting the GUI use an ordinary main() instead of WinMain() - see its
# own header comment.
# NOTE: like the Windows OS branch above, this is untested - no Windows+CUDA
# machine is available to build-verify it on.
ifeq ($(OS),Windows_NT)
GUI_TARGET  := $(BUILD)/namebreak-gui$(EXE_EXT)
GUI_LDFLAGS := $(NETWORK_LDFLAGS) -lcomctl32 -lshell32 -Xlinker /SUBSYSTEM:WINDOWS -Xlinker /ENTRY:mainCRTStartup

.PHONY: gui
gui: $(GUI_TARGET)

$(GUI_TARGET): $(call objs,gui,$(GUI_SRC) $(SEARCH_SRC) $(COMMON_SRC) $(NET_SRC))
	$(COMPILE) -o $@ $^ $(GUI_LDFLAGS)
endif

# --- Tests and benchmarks ----------------------------------------------------
#
# Every GPU test runs the real runSearch() - it calls fopen("matches.txt", "a")
# relative to the current directory, so they're run from $(TESTRUN) to keep
# them away from any real matches.txt.

# Pure-CPU correctness check for IncrementalPrefixHasher
# (src/engine/mpq_hash.h) - no GPU involved.
$(BUILD)/tests/prefix_hasher_test: $(call objs,test,tests/prefix_hasher_test.cpp $(ENGINE_SRC))
	@mkdir -p $(@D)
	$(COMPILE) -o $@ $^

# Pure-CPU benchmark isolating IncrementalPrefixHasher's own speedup (O(1)
# amortized vs. from-scratch rehashing per leadingIdx step) from the separate
# GPU-window-size question - no GPU involved.
$(BUILD)/tests/prefix_hasher_bench: $(call objs,test,tests/prefix_hasher_bench.cpp $(ENGINE_SRC))
	@mkdir -p $(@D)
	$(COMPILE) -o $@ $^

prefix_hasher_bench: $(BUILD)/tests/prefix_hasher_bench
	./$<

# End-to-end correctness check for runSearch() itself against a CPU
# brute-force reference - see the file for why this is a separate concern
# from prefix_hasher_test above. Built without NETWORK (doesn't need it).
#
# The same test source is built in several configurations (see "Build
# configurations" above), because the tuning values in
# src/engine/constants.h change which code paths a search exercises - every
# case derives its geometry from those values:
#   test             the shipped configuration
#   small_launch     3 rows per launch: forces many launches per leading value,
#                    so the chunk boundaries are actually exercised
#   window2          a 2-character GPU window (5 rows per launch): a much
#                    longer CPU-side leading part, and a shallower row decode
#   window6          the widest supported window: 5 prefix digits per row and
#                    trailing indices beyond 32 bits
INTEGRATION_CONFIGS := test small_launch window2 window6
INTEGRATION_TESTS := $(patsubst %,$(BUILD)/tests/search_integration_test-%,$(INTEGRATION_CONFIGS))

define INTEGRATION_RULE
$(BUILD)/tests/search_integration_test-$(1): $(call objs,$(1),tests/search_integration_test.cu $(SEARCH_SRC))
	@mkdir -p $$(@D)
	$(COMPILE) -o $$@ $$^
endef
$(foreach c,$(INTEGRATION_CONFIGS),$(eval $(call INTEGRATION_RULE,$(c))))

# What runSearch() does when a launch has more hashA hits than MAX_MATCHES can
# record - built with MAX_MATCHES=1 so a single hashA collision pair overflows it.
$(BUILD)/tests/search_overflow_test: $(call objs,overflow,tests/search_overflow_test.cu $(SEARCH_SRC))
	@mkdir -p $(@D)
	$(COMPILE) -o $@ $^

# The CUDA backend takes about a minute to compile per configuration (the
# search kernel is instantiated per alphabet size and suffix length) -
# `make -j test` builds them in parallel.
test: $(BUILD)/tests/prefix_hasher_test $(INTEGRATION_TESTS) $(BUILD)/tests/search_overflow_test
	./$(BUILD)/tests/prefix_hasher_test
	mkdir -p $(TESTRUN)
	cd $(TESTRUN) && $(CURDIR)/$(BUILD)/tests/search_integration_test-test
	cd $(TESTRUN) && $(CURDIR)/$(BUILD)/tests/search_integration_test-small_launch
	cd $(TESTRUN) && $(CURDIR)/$(BUILD)/tests/search_integration_test-window2
	cd $(TESTRUN) && $(CURDIR)/$(BUILD)/tests/search_integration_test-window6
	cd $(TESTRUN) && $(CURDIR)/$(BUILD)/tests/search_overflow_test

# Real-code throughput check - times the actual runSearch() (not a proxy
# kernel) over a ~28.8B-candidate range. See WINDOW above.
$(BUILD)/tests/search_bench$(if $(WINDOW),-window$(WINDOW)): $(call objs,$(BENCH_CONFIG),tests/search_bench.cu $(SEARCH_SRC))
	@mkdir -p $(@D)
	$(COMPILE) -o $@ $^

search_bench: $(BUILD)/tests/search_bench$(if $(WINDOW),-window$(WINDOW))
	mkdir -p $(TESTRUN)
	cd $(TESTRUN) && $(CURDIR)/$<

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
