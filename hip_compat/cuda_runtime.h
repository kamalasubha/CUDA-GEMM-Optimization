// Maps the CUDA runtime API used by this project onto HIP.
// This directory is on the include path only when building with USE_HIP=ON.
#ifndef HIP_COMPAT_CUDA_RUNTIME_H
#define HIP_COMPAT_CUDA_RUNTIME_H

#include <hip/hip_runtime.h>

#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#define cudaDataType_t hipDataType
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F

#endif // HIP_COMPAT_CUDA_RUNTIME_H
