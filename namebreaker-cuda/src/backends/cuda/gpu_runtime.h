#ifndef NAMEBREAK_BACKENDS_CUDA_GPU_RUNTIME_H
#define NAMEBREAK_BACKENDS_CUDA_GPU_RUNTIME_H

// The GPU runtime cuda_backend.cu is written against: CUDA's - or, when it's
// compiled with HIP for AMD GPUs (NAMEBREAK_HIP, see NAMEBREAK_GPU=hip in
// CMakeLists.txt), HIP's, which has the same calls under hip* names. The
// kernels themselves (__global__, __constant__, <<<...>>>, atomicAdd, ...)
// need no mapping: HIP accepts CUDA's syntax as is.
#ifdef NAMEBREAK_HIP
#include <hip/hip_runtime.h>

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
// HIP takes a __device__/__constant__ variable's address, via HIP_SYMBOL,
// where CUDA takes the variable itself.
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, ...) hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), __VA_ARGS__)

#define NAMEBREAK_GPU_RUNTIME_NAME "HIP"
#else
#include <cuda_runtime.h>

#define NAMEBREAK_GPU_RUNTIME_NAME "CUDA"
#endif

#endif // NAMEBREAK_BACKENDS_CUDA_GPU_RUNTIME_H
