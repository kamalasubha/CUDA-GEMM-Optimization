// Maps the cuBLAS API used by this project onto hipBLAS.
#ifndef HIP_COMPAT_CUBLAS_V2_H
#define HIP_COMPAT_CUBLAS_V2_H

#include <hipblas/hipblas.h>

#include "cuda_runtime.h"

#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmAlgo_t hipblasGemmAlgo_t
#define cublasGetStatusString hipblasStatusToString
#define cublasHandle_t hipblasHandle_t
#define cublasSetStream hipblasSetStream
#define cublasStatus_t hipblasStatus_t

// cublasGemmEx accepts a cudaDataType_t compute type through its C++
// overload; hipblasGemmEx takes a hipblasComputeType_t.
inline hipblasComputeType_t hip_compat_compute_type(hipDataType data_type)
{
    switch (data_type)
    {
        case HIP_R_16F:
            return HIPBLAS_COMPUTE_16F;
        case HIP_R_64F:
            return HIPBLAS_COMPUTE_64F;
        default:
            return HIPBLAS_COMPUTE_32F;
    }
}

inline hipblasStatus_t
cublasGemmEx(hipblasHandle_t handle, hipblasOperation_t transa,
             hipblasOperation_t transb, int m, int n, int k, void const* alpha,
             void const* A, hipDataType Atype, int lda, void const* B,
             hipDataType Btype, int ldb, void const* beta, void* C,
             hipDataType Ctype, int ldc, hipDataType computeType,
             hipblasGemmAlgo_t algo)
{
    return hipblasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                         B, Btype, ldb, beta, C, Ctype, ldc,
                         hip_compat_compute_type(computeType), algo);
}

#endif // HIP_COMPAT_CUBLAS_V2_H
