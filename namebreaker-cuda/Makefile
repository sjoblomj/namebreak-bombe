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
ifeq ($(NETWORK),1)
SOURCES += coordinator_runner.cpp coordinator_client.cpp http_client.cpp protocol.cpp platform.cpp
HEADERS += coordinator_runner.h coordinator_client.h http_client.h protocol.h platform.h
NVCC_FLAGS += -DNAMEBREAK_WITH_NETWORK
LDFLAGS += -lcurl -lpthread
endif

.PHONY: all verbose clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Same build, but with ptxas' per-kernel register/spill/stack-frame report - useful when
# checking whether the kernel still fits comfortably within occupancy limits.
verbose: $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) $(NVCC_FLAGS) --ptxas-options=-v -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)
