#ifndef NAMEBREAK_BACKENDS_GPU_DEVICE_H
#define NAMEBREAK_BACKENDS_GPU_DEVICE_H

#include <string>

// One GPU a backend can search on (see gpuDevices, backends.h).
struct GpuDevice {
    // The backend's own number for it, from 0 - what [hardware]'s `gpus`
    // and `gpu<N>.` keys (see HardwareConfig, backends.h) and
    // `namebreak --list-gpus` call it. The same card can have another
    // number in another backend.
    int index = 0;
    std::string name;
    // Whether it's built into the CPU, sharing its memory and power - left
    // out by default when there's a discrete GPU too (see HardwareConfig).
    bool integrated = false;
};

#endif // NAMEBREAK_BACKENDS_GPU_DEVICE_H
