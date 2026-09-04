NVCC    := nvcc
CXXSTD  := c++17
OPT     := -O3

# Target GPU architecture, as a compute capability (see `nvidia-smi --query-gpu=compute_cap
# --format=csv`). Defaults to sm_86 (confirmed for this machine's RTX 3080 Ti Laptop GPU) -
# override on the command line for a different GPU, e.g.: make ARCH=sm_75
ARCH    := sm_86

TARGET  := namebreak
SOURCES := namebreak.cu cpu-utils.cpp
HEADERS := cpu-utils.h constants.h

.PHONY: all verbose clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) -o $(TARGET) $(SOURCES)

# Same build, but with ptxas' per-kernel register/spill/stack-frame report - useful when
# checking whether the kernel still fits comfortably within occupancy limits.
verbose: $(SOURCES) $(HEADERS)
	$(NVCC) -std=$(CXXSTD) $(OPT) -arch=$(ARCH) --ptxas-options=-v -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)
