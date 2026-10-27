#ifndef NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H
#define NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "backends/gpu_device.h"
#include "backends/gpu_tuning.h"
#include "engine/backend.h"

// Searches on the GPU numbered `device` in openClDevices() - AMD, Intel or
// NVIDIA, with nothing but the vendor's regular driver installed - with the
// same lookup filter as the CUDA backend (backends/common/lowbits_filter.h),
// launched as `tuning` says (openClTuningDefaults() where it says nothing).
// Null, with `error` set, if there's no such GPU.
std::unique_ptr<SearchBackend> makeOpenClBackend(int device, const GpuTuning& tuning, std::string& error);

// The GPUs every OpenCL platform offers, numbered across them in the order
// they sit on the PCI bus, if their drivers say (cl_khr_pci_bus_info) - so
// that a number stays the same card as drivers are installed and removed -
// then the rest in the order the platforms list them. A GPU that two
// platforms both offer (Mesa's rusticl and AMD's own driver, say) is listed
// once, by the first, if the drivers say where it sits; if not, it's listed
// by both.
std::vector<GpuDevice> openClDevices();

// What the backend uses for any value a GpuTuning leaves unset - every one
// set.
GpuTuning openClTuningDefaults();

#endif // NAMEBREAK_BACKENDS_OPENCL_OPENCL_BACKEND_H
