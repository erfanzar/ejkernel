cmake_minimum_required(VERSION 3.21)
project(ejkernel_quantized_matmul LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EJKERNEL_CUDA_ARCH "" CACHE STRING "CUDA SM architecture (e.g. 80, 90, 100, 110, 120)")
set(EJKERNEL_CUDA_ARCHS "" CACHE STRING "CUDA SM architectures (e.g. 80;90;100;110;120)")
set(EJKERNEL_JAX_FFI_INCLUDE "" CACHE STRING "Path to JAX FFI include dir")

if(NOT EJKERNEL_CUDA_ARCHS AND NOT EJKERNEL_CUDA_ARCH)
  message(FATAL_ERROR "Set EJKERNEL_CUDA_ARCHS or EJKERNEL_CUDA_ARCH (e.g. -DEJKERNEL_CUDA_ARCH=80)")
endif()

if(NOT EJKERNEL_JAX_FFI_INCLUDE)
  message(FATAL_ERROR "EJKERNEL_JAX_FFI_INCLUDE must be set (path to jax/ffi include)")
endif()

set(QMM_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(CUTLASS_ROOT "${REPO_ROOT}/csrc/cutlass")
set(CUTLASS_INCLUDE "${CUTLASS_ROOT}/include")
file(GLOB QMM_DEQUANT_SOURCES "${QMM_SRC_DIR}/qmm_dequant_*.cu")
list(FILTER QMM_DEQUANT_SOURCES EXCLUDE REGEX "qmm_dequant_affine_bits(2|3|5|6|7)_.*\\.cu$")
set(QMM_SOURCES
  "${QMM_SRC_DIR}/qmm_cuda.cu"
  ${QMM_DEQUANT_SOURCES}
)

if(EJKERNEL_CUDA_ARCHS)
  set(_arch_list ${EJKERNEL_CUDA_ARCHS})
else()
  set(_arch_list ${EJKERNEL_CUDA_ARCH})
endif()

foreach(arch IN LISTS _arch_list)
  set(target_name "ejkernel_qmm_cuda_sm${arch}")

  add_library(${target_name} SHARED ${QMM_SOURCES})

  set_target_properties(${target_name} PROPERTIES
    CUDA_ARCHITECTURES "${arch}"
    OUTPUT_NAME "ejkernel_qmm_cuda_sm${arch}"
  )

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  target_include_directories(${target_name} PRIVATE
    "${QMM_SRC_DIR}"
    "${EJKERNEL_JAX_FFI_INCLUDE}"
    "${CUTLASS_INCLUDE}"
  )

  target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cublas
                                               CUDA::cublasLt)
endforeach()
