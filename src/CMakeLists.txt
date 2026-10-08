cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Algorithms VERSION 0.0.1 LANGUAGES CXX)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)
if(USE_HIP)
    enable_language(HIP)
else()
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_HIP)
    # rocWMMA requires C++17.
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    # nvcc optimizes device code even when no build type is set; clang does
    # not, so match that default for the HIP build.
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
    endif()
    find_package(hipblas REQUIRED)
    # rocWMMA is header-only. Its CMake package config also requires OpenMP,
    # which this project does not use, so only its headers are located.
    find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)
else()
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
endif()

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
file(GLOB CUDA_GEMM_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

# Add all the source files in the current directory to build the library
add_library(
    cuda_gemm 
    SHARED 
    cuda_gemm_utils.cu 
    00_non_coalesced_global_memory_access.cu
    01_coalesced_global_memory_access.cu
    02_2d_block_tiling.cu
    02_2d_block_tiling_vectorized_memory_access.cu
    03_2d_block_tiling_1d_thread_tiling.cu
    03_2d_block_tiling_1d_thread_tiling_vectorized_memory_access.cu
    04_2d_block_tiling_2d_thread_tiling.cu
    04_2d_block_tiling_2d_thread_tiling_vectorized_memory_access.cu
    05_2d_block_tiling_2d_thread_tiling_matrix_transpose.cu
    05_2d_block_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access.cu
    06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose.cu
    06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access.cu
    06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access_double_buffered.cu
    07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma.cu
    07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma_vectorized_memory_access.cu
    07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma_vectorized_memory_access_double_buffered.cu
)

# Add the include directory of the library to the include directories of the project
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

if(USE_HIP)
    file(GLOB CUDA_GEMM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    set_source_files_properties(${CUDA_GEMM_SOURCES} PROPERTIES LANGUAGE HIP)
    # The headers in hip_compat map the CUDA headers included by the sources
    # onto their HIP and ROCm library equivalents.
    target_include_directories(cuda_gemm BEFORE PUBLIC ${CMAKE_SOURCE_DIR}/hip_compat)
    target_include_directories(cuda_gemm PUBLIC ${ROCWMMA_INCLUDE_DIR})
    set_target_properties(cuda_gemm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
else()
    # Set the CUDA architecture to compile the code for
    # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
    set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES native)
endif()
install(TARGETS cuda_gemm DESTINATION lib)
install(FILES ${CUDA_GEMM_HEADERS} DESTINATION include)

if(USE_HIP)
    set(CUDA_GEMM_BLAS_LIBRARY roc::hipblas)
else()
    set(CUDA_GEMM_BLAS_LIBRARY CUDA::cublas)
endif()

add_executable(profile_cuda_gemm_fp32 profile_cuda_gemm_fp32.cu)
target_link_libraries(profile_cuda_gemm_fp32 cuda_gemm ${CUDA_GEMM_BLAS_LIBRARY})

add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm ${CUDA_GEMM_BLAS_LIBRARY})

if(USE_HIP)
    set_target_properties(profile_cuda_gemm_fp32 profile_cuda_gemm_fp16 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
else()
    set_target_properties(profile_cuda_gemm_fp32 PROPERTIES CUDA_ARCHITECTURES native)
    set_target_properties(profile_cuda_gemm_fp16 PROPERTIES CUDA_ARCHITECTURES native)
endif()
