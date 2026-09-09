NVCC    := nvcc
CXXSTD  := c++17
OPT     := -O3

# Target GPU architecture, as a compute capability (see `nvidia-smi --query-gpu=compute_cap
# --format=csv`). Defaults to sm_86 (confirmed for this machine's RTX 3080 Ti Laptop GPU) -
# override on the command line for a different GPU, e.g.: make ARCH=sm_75
ARCH    := sm_86

TARGET  := namebreak
SOURCES := namebreak.cu cpu-utils.cpp config.cpp
HEADERS := cpu-utils.h constants.h search.h config.h

# Set to 0 to build without the coordinator client (no libcurl dependency,
# `coordinator` mode unavailable at runtime): make NETWORK=0
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
    NETWORK_LDFLAGS = -lcurl -lpthread
endif

ifeq ($(NETWORK),1)
SOURCES += coordinator_runner.cpp coordinator_client.cpp http_client.cpp protocol.cpp platform.cpp
HEADERS += coordinator_runner.h coordinator_client.h http_client.h protocol.h platform.h
NVCC_FLAGS += -DNAMEBREAK_WITH_NETWORK $(if $(CURL_INC),-I$(CURL_INC))
LDFLAGS += $(NETWORK_LDFLAGS)
endif

.PHONY: all verbose clean

all: $(TARGET)$(EXE_EXT)

$(TARGET)$(EXE_EXT): $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) $(NVCC_FLAGS) -o $(TARGET)$(EXE_EXT) $(SOURCES) $(LDFLAGS)

# Same build, but with ptxas' per-kernel register/spill/stack-frame report - useful when
# checking whether the kernel still fits comfortably within occupancy limits.
verbose: $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) $(NVCC_FLAGS) --ptxas-options=-v -o $(TARGET)$(EXE_EXT) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)$(EXE_EXT)
