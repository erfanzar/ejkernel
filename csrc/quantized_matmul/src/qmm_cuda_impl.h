// Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "qmm_dequant_dispatch.h"
#include "qmm_dequant_kernels.h"
#include "xla/ffi/api/ffi.h"

namespace {

using xla::ffi::AnyBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Result;
using xla::ffi::ScratchAllocator;
using xla::ffi::Span;

constexpr int kBits4 = 4;
constexpr int kBits8 = 8;

inline const char *CublasErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "CUBLAS_STATUS_UNKNOWN";
  }
}

__global__ void convert_f32_to_f16(const float *in, half *out, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = ToHalf(in[idx]);
  }
}

__global__ void convert_bf16_to_f16(const __nv_bfloat16 *in, half *out,
                                    int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = ToHalf(ToFloat(in[idx]));
  }
}

inline bool IsFiniteInt64(int64_t v) {
  return v >= 0 && v <= std::numeric_limits<int32_t>::max();
}

inline Error MakeInvalid(std::string msg) {
  return Error::InvalidArgument(std::move(msg));
}

inline Error MakeInternal(const std::string &msg) {
  return Error::Internal(msg);
}

inline Error CheckCuda(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return Error::Success();
  }
  std::string msg = std::string(what) + ": " + cudaGetErrorString(err);
  return MakeInternal(msg);
}

inline Error CheckCublas(cublasStatus_t status, const char *what) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Error::Success();
  }
  std::string msg = std::string(what) + ": " + CublasErrorString(status);
  return MakeInternal(msg);
}

inline Error CheckCublasLt(cublasStatus_t status, const char *what) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Error::Success();
  }
  std::string msg = std::string(what) + ": " + CublasErrorString(status);
  return MakeInternal(msg);
}

inline cublasComputeType_t GetCublasComputeType() {
  static int cached = -1;
  static cublasComputeType_t cached_type = CUBLAS_COMPUTE_32F;
  if (cached != -1) {
    return cached_type;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_COMPUTE");
  if (!env) {
    cached = 1;
    cached_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    return cached_type;
  }
  if (env[0] == 't' || env[0] == 'T') {
    cached = 1;
    cached_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    return cached_type;
  }
  if (env[0] == 'f' || env[0] == 'F') {
    cached = 1;
    cached_type = CUBLAS_COMPUTE_32F_FAST_16F;
    return cached_type;
  }
  if (env[0] == 'b' || env[0] == 'B') {
    cached = 1;
    cached_type = CUBLAS_COMPUTE_32F_FAST_16BF;
    return cached_type;
  }
  cached = 1;
  cached_type = CUBLAS_COMPUTE_32F;
  return cached_type;
}

inline cublasMath_t GetCublasMathMode(cublasComputeType_t compute_type) {
  if (compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) {
    return CUBLAS_TF32_TENSOR_OP_MATH;
  }
  return CUBLAS_TENSOR_OP_MATH;
}

enum class GemmBackend {
  kCublas = 0,
  kCublasLt = 1,
  kCutlass = 2,
  kCutlassW4A16 = 3,
  kCutlassTuned = 4,
};

inline GemmBackend GetGemmBackend() {
  static int cached = -1;
  static GemmBackend cached_backend = GemmBackend::kCublasLt;
  if (cached != -1) {
    return cached_backend;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_GEMM");
  if (!env) {
    cached = 1;
    cached_backend = GemmBackend::kCublasLt;
    return cached_backend;
  }
  std::string mode(env);
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
  if (mode == "cublaslt") {
    cached_backend = GemmBackend::kCublasLt;
  } else if (mode == "cutlass") {
    cached_backend = GemmBackend::kCutlass;
  } else if (mode == "cutlass_tuned" || mode == "cutlasstuned") {
    cached_backend = GemmBackend::kCutlassTuned;
  } else if (mode == "cutlass_w4a16" || mode == "cutlassw4a16") {
    cached_backend = GemmBackend::kCutlassW4A16;
  } else {
    cached_backend = GemmBackend::kCublas;
  }
  cached = 1;
  return cached_backend;
}

inline bool StrictGemmBackend() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_STRICT");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

inline size_t GetCublasLtWorkspaceLimit() {
  static int cached = -1;
  static size_t cached_bytes = 0;
  if (cached != -1) {
    return cached_bytes;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_CUBLASLT_WS_MB");
  if (!env) {
    cached = 1;
    cached_bytes = static_cast<size_t>(32) * 1024 * 1024;
    return cached_bytes;
  }
  char *end = nullptr;
  long value = std::strtol(env, &end, 10);
  if (end == env || value <= 0) {
    cached = 1;
    cached_bytes = static_cast<size_t>(32) * 1024 * 1024;
    return cached_bytes;
  }
  cached = 1;
  cached_bytes = static_cast<size_t>(value) * 1024 * 1024;
  return cached_bytes;
}

inline bool UseFusedQmm() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_FUSED");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

__device__ __forceinline__ uint32_t LoadPackedQ(const uint32_t *row, int64_t n,
                                                int bits) {
  if (bits == 4) {
    int64_t word_idx = n >> 3;
    int32_t shift = static_cast<int32_t>(n & 7) * 4;
    return (row[word_idx] >> shift) & 0xFu;
  }
  if (bits == 8) {
    int64_t word_idx = n >> 2;
    int32_t shift = static_cast<int32_t>(n & 3) * 8;
    return (row[word_idx] >> shift) & 0xFFu;
  }

  int64_t bit_offset = n * bits;
  int64_t word_idx = bit_offset >> 5;
  int32_t shift = static_cast<int32_t>(bit_offset & 31);
  uint32_t low_word = row[word_idx];
  int32_t low_bits = (shift + bits > 32) ? (32 - shift) : bits;
  int32_t high_bits = bits - low_bits;
  uint32_t low_mask = (uint32_t(1) << low_bits) - 1u;
  uint32_t low = (low_word >> shift) & low_mask;
  uint32_t high = 0;
  if (high_bits > 0) {
    uint32_t high_mask = (uint32_t(1) << high_bits) - 1u;
    high = row[word_idx + 1] & high_mask;
  }
  return low | (high << low_bits);
}

__device__ __forceinline__ float ToFloat(uint8_t v) {
  return static_cast<float>(v);
}

template <typename XType>
__device__ __forceinline__ half LoadXAsHalf(const XType *ptr) {
  return ToHalf(ToFloat(*ptr));
}

template <>
__device__ __forceinline__ half LoadXAsHalf<half>(const half *ptr) {
  return *ptr;
}

template <typename ScaleT, typename BiasT>
__device__ __forceinline__ float DequantValue(uint32_t q, const ScaleT *scales,
                                              const BiasT *biases, int64_t k,
                                              int64_t g, int64_t n_groups,
                                              int64_t mode) {
  int64_t idx = k * n_groups + g;
  if (mode == 0) {
    float scale = ToFloat(scales[idx]);
    float bias = ToFloat(biases[idx]);
    return static_cast<float>(q) * scale + bias;
  }
  if (mode == 1) {
    float scale = ToFloat(scales[idx]);
    return kNF4Table[q] * scale;
  }
  if (mode == 2) {
    uint8_t exp_u8 = static_cast<uint8_t>(scales[idx]);
    int8_t exp = static_cast<int8_t>(exp_u8);
    float scale = exp2f(static_cast<float>(exp));
    return kE2M1Table[q] * scale;
  }
  if (mode == 3) {
    uint8_t exp_u8 = static_cast<uint8_t>(scales[idx]);
    int8_t exp = static_cast<int8_t>(exp_u8);
    float scale = exp2f(static_cast<float>(exp));
    return kE4M3Table[q] * scale;
  }
  if (mode == 4) {
    uint8_t code = static_cast<uint8_t>(scales[idx]);
    float scale = kE4M3Table[code];
    return kE2M1Table[q] * scale;
  }
  uint8_t code = static_cast<uint8_t>(scales[idx]);
  float scale = kE4M3Table[code];
  return kE4M3Table[q] * scale;
}

template <typename XType, typename ScaleT, typename BiasT>
__global__ void QmmFusedKernel(const XType *x, const uint32_t *wq,
                               const ScaleT *scales, const BiasT *biases,
                               float *out, int64_t M, int64_t N, int64_t K,
                               int64_t n_words, int64_t n_groups,
                               int64_t group_size, int bits, int64_t mode) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;

  int block_m = static_cast<int>(blockIdx.y) * BM;
  int block_n = static_cast<int>(blockIdx.x) * BN;

  __shared__ half sh_a[BM * BK];
  __shared__ half sh_b[BK * BN];

  int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
  int num_threads = static_cast<int>(blockDim.x * blockDim.y);

  int base_row = block_m + static_cast<int>(threadIdx.y) * 4;
  int base_col = block_n + static_cast<int>(threadIdx.x) * 4;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int64_t k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += num_threads) {
      int a_row = idx / BK;
      int a_col = idx - a_row * BK;
      int64_t g_m = static_cast<int64_t>(block_m + a_row);
      int64_t g_k = k0 + a_col;
      half val = ToHalf(0.0f);
      if (g_m < M && g_k < K) {
        val = LoadXAsHalf(x + g_m * K + g_k);
      }
      sh_a[idx] = val;
    }

    for (int idx = tid; idx < BK * BN; idx += num_threads) {
      int b_row = idx / BN;
      int b_col = idx - b_row * BN;
      int64_t g_k = k0 + b_row;
      int64_t g_n = static_cast<int64_t>(block_n + b_col);
      half val = ToHalf(0.0f);
      if (g_k < K && g_n < N) {
        const uint32_t *wq_row = wq + g_k * n_words;
        uint32_t q = LoadPackedQ(wq_row, g_n, bits);
        int64_t g = g_n / group_size;
        float w_val = DequantValue(q, scales, biases, g_k, g, n_groups, mode);
        val = ToHalf(w_val);
      }
      sh_b[idx] = val;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int a_row0 = static_cast<int>(threadIdx.y) * 4;
      int b_col0 = static_cast<int>(threadIdx.x) * 4;

      float a0 = __half2float(sh_a[(a_row0 + 0) * BK + k]);
      float a1 = __half2float(sh_a[(a_row0 + 1) * BK + k]);
      float a2 = __half2float(sh_a[(a_row0 + 2) * BK + k]);
      float a3 = __half2float(sh_a[(a_row0 + 3) * BK + k]);

      float b0 = __half2float(sh_b[k * BN + (b_col0 + 0)]);
      float b1 = __half2float(sh_b[k * BN + (b_col0 + 1)]);
      float b2 = __half2float(sh_b[k * BN + (b_col0 + 2)]);
      float b3 = __half2float(sh_b[k * BN + (b_col0 + 3)]);

      acc[0][0] += a0 * b0;
      acc[0][1] += a0 * b1;
      acc[0][2] += a0 * b2;
      acc[0][3] += a0 * b3;
      acc[1][0] += a1 * b0;
      acc[1][1] += a1 * b1;
      acc[1][2] += a1 * b2;
      acc[1][3] += a1 * b3;
      acc[2][0] += a2 * b0;
      acc[2][1] += a2 * b1;
      acc[2][2] += a2 * b2;
      acc[2][3] += a2 * b3;
      acc[3][0] += a3 * b0;
      acc[3][1] += a3 * b1;
      acc[3][2] += a3 * b2;
      acc[3][3] += a3 * b3;
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int64_t g_m = static_cast<int64_t>(base_row + i);
    if (g_m >= M) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int64_t g_n = static_cast<int64_t>(base_col + j);
      if (g_n >= N) {
        continue;
      }
      out[g_m * N + g_n] = acc[i][j];
    }
  }
}

template <typename XType, typename ScaleT, typename BiasT>
Error LaunchQmmFusedKernel(const XType *x, const uint32_t *wq,
                           const ScaleT *scales, const BiasT *biases,
                           float *out, int64_t M, int64_t N, int64_t K,
                           int64_t n_words, int64_t n_groups,
                           int64_t group_size, int bits, int64_t mode,
                           cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(static_cast<uint32_t>((N + 63) / 64),
            static_cast<uint32_t>((M + 63) / 64));
  QmmFusedKernel<XType, ScaleT, BiasT><<<grid, block, 0, stream>>>(
      x, wq, scales, biases, out, M, N, K, n_words, n_groups, group_size,
      bits, mode);
  return CheckCuda(cudaPeekAtLastError(), "fused qmm kernel launch");
}

struct CublasHandleCache {
  cublasHandle_t handle = nullptr;
  int device = -1;

  ~CublasHandleCache() {
    if (handle) {
      cublasDestroy(handle);
    }
  }

  Error Get(cublasHandle_t *out, cudaStream_t stream) {
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return MakeInternal("Failed to query current CUDA device.");
    }
    if (handle == nullptr || device != current_device) {
      if (handle) {
        cublasDestroy(handle);
        handle = nullptr;
      }
      if (Error err = CheckCublas(cublasCreate(&handle), "cublasCreate");
          err.failure()) {
        return err;
      }
      device = current_device;
      if (Error err = CheckCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH),
                                  "cublasSetMathMode");
          err.failure()) {
        return err;
      }
    }
    if (Error err = CheckCublas(cublasSetStream(handle, stream),
                                "cublasSetStream");
        err.failure()) {
      return err;
    }
    *out = handle;
    return Error::Success();
  }
};

static thread_local CublasHandleCache g_cublas_cache;

struct CublasLtHandleCache {
  cublasLtHandle_t handle = nullptr;
  int device = -1;

  ~CublasLtHandleCache() {
    if (handle) {
      cublasLtDestroy(handle);
    }
  }

  Error Get(cublasLtHandle_t *out) {
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return MakeInternal("Failed to query current CUDA device.");
    }
    if (handle == nullptr || device != current_device) {
      if (handle) {
        cublasLtDestroy(handle);
        handle = nullptr;
      }
      if (Error err = CheckCublasLt(cublasLtCreate(&handle), "cublasLtCreate");
          err.failure()) {
        return err;
      }
      device = current_device;
    }
    *out = handle;
    return Error::Success();
  }
};

static thread_local CublasLtHandleCache g_cublaslt_cache;

Error RunCublasLtGemm(const half *w_deq, const half *x_half, float *out,
                      int64_t M, int64_t N, int64_t K, cudaStream_t stream,
                      ScratchAllocator &scratch,
                      cublasComputeType_t compute_type) {
  cublasLtHandle_t lt_handle;
  if (Error err = g_cublaslt_cache.Get(&lt_handle); err.failure()) {
    return err;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  cublasOperation_t trans = CUBLAS_OP_N;

  if (Error err = CheckCublasLt(
          cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
      err.failure()) {
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatmulDescSetAttribute(op_desc,
                                         CUBLASLT_MATMUL_DESC_TRANSA, &trans,
                                         sizeof(trans)),
          "cublasLtMatmulDescSetAttribute(TRANSA)");
      err.failure()) {
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatmulDescSetAttribute(op_desc,
                                         CUBLASLT_MATMUL_DESC_TRANSB, &trans,
                                         sizeof(trans)),
          "cublasLtMatmulDescSetAttribute(TRANSB)");
      err.failure()) {
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16F,
                                     static_cast<int64_t>(N),
                                     static_cast<int64_t>(K),
                                     static_cast<int64_t>(N)),
          "cublasLtMatrixLayoutCreate(A)");
      err.failure()) {
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16F,
                                     static_cast<int64_t>(K),
                                     static_cast<int64_t>(M),
                                     static_cast<int64_t>(K)),
          "cublasLtMatrixLayoutCreate(B)");
      err.failure()) {
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32F,
                                     static_cast<int64_t>(N),
                                     static_cast<int64_t>(M),
                                     static_cast<int64_t>(N)),
          "cublasLtMatrixLayoutCreate(C)");
      err.failure()) {
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  if (Error err = CheckCublasLt(
          cublasLtMatmulPreferenceCreate(&preference),
          "cublasLtMatmulPreferenceCreate");
      err.failure()) {
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  size_t workspace_limit = GetCublasLtWorkspaceLimit();
  if (Error err = CheckCublasLt(
          cublasLtMatmulPreferenceSetAttribute(
              preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &workspace_limit, sizeof(workspace_limit)),
          "cublasLtMatmulPreferenceSetAttribute");
      err.failure()) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  cublasLtMatmulHeuristicResult_t results[4] = {};
  int returned = 0;
  if (Error err = CheckCublasLt(
          cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, layout_a,
                                         layout_b, layout_c, layout_c,
                                         preference, 4, results, &returned),
          "cublasLtMatmulAlgoGetHeuristic");
      err.failure()) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  if (returned == 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return MakeInternal("cublasLtMatmulAlgoGetHeuristic returned no algorithms.");
  }

  const auto &algo = results[0].algo;
  size_t workspace_size = results[0].workspaceSize;
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_opt =
        scratch.Allocate(workspace_size, /*alignment=*/256);
    if (!workspace_opt.has_value()) {
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(layout_c);
      cublasLtMatrixLayoutDestroy(layout_b);
      cublasLtMatrixLayoutDestroy(layout_a);
      cublasLtMatmulDescDestroy(op_desc);
      return MakeInternal("Failed to allocate cublasLt workspace.");
    }
    workspace = *workspace_opt;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasStatus_t status = cublasLtMatmul(
      lt_handle, op_desc, &alpha, w_deq, layout_a, x_half, layout_b, &beta,
      out, layout_c, out, layout_c, &algo, workspace, workspace_size, stream);

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(layout_c);
  cublasLtMatrixLayoutDestroy(layout_b);
  cublasLtMatrixLayoutDestroy(layout_a);
  cublasLtMatmulDescDestroy(op_desc);

  return CheckCublasLt(status, "cublasLtMatmul");
}

Error RunCutlassGemm(const half *w_deq, const half *x_half, float *out,
                     int64_t M, int64_t N, int64_t K, cudaStream_t stream,
                     ScratchAllocator &scratch) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                           ElementC, LayoutC,
                                           ElementAccumulator,
                                           cutlass::arch::OpClassTensorOp,
                                           cutlass::arch::Sm80>;

  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M),
                                        static_cast<int>(N),
                                        static_cast<int>(K));

  typename Gemm::Arguments args(
      problem_size,
      {reinterpret_cast<ElementA const *>(x_half), static_cast<int>(K)},
      {reinterpret_cast<ElementB const *>(w_deq), static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {1.0f, 0.0f});

  cutlass::Status can_impl = gemm_op.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    return MakeInvalid("CUTLASS GEMM configuration not supported.");
  }

  size_t workspace_size = gemm_op.get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_opt =
        scratch.Allocate(workspace_size, /*alignment=*/256);
    if (!workspace_opt.has_value()) {
      return MakeInternal("Failed to allocate CUTLASS workspace.");
    }
    workspace = *workspace_opt;
  }

  cutlass::Status status = gemm_op(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return MakeInternal("CUTLASS GEMM failed.");
  }
  return Error::Success();
}

template <typename Gemm>
Error RunCutlassGemmImpl(const half *w_deq, const half *x_half, float *out,
                         int64_t M, int64_t N, int64_t K, cudaStream_t stream,
                         ScratchAllocator &scratch) {
  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M),
                                        static_cast<int>(N),
                                        static_cast<int>(K));

  typename Gemm::Arguments args(
      problem_size,
      {reinterpret_cast<typename Gemm::ElementA const *>(x_half),
       static_cast<int>(K)},
      {reinterpret_cast<typename Gemm::ElementB const *>(w_deq),
       static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {1.0f, 0.0f});

  cutlass::Status can_impl = gemm_op.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    return MakeInvalid("CUTLASS GEMM configuration not supported.");
  }

  size_t workspace_size = gemm_op.get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_opt =
        scratch.Allocate(workspace_size, /*alignment=*/256);
    if (!workspace_opt.has_value()) {
      return MakeInternal("Failed to allocate CUTLASS workspace.");
    }
    workspace = *workspace_opt;
  }

  cutlass::Status status = gemm_op(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return MakeInternal("CUTLASS GEMM failed.");
  }
  return Error::Success();
}

Error RunCutlassGemmTuned(const half *w_deq, const half *x_half, float *out,
                          int64_t M, int64_t N, int64_t K,
                          cudaStream_t stream, ScratchAllocator &scratch) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Epilogue =
      cutlass::epilogue::thread::LinearCombination<
          ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
          ElementAccumulator, ElementAccumulator>;

  using Gemm128x64x64 = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  using Gemm64x128x64 = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 64>,
      cutlass::gemm::GemmShape<32, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  using Gemm128x128x32 = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  if (N >= 8192 && M >= 8192) {
    return RunCutlassGemmImpl<Gemm128x128x32>(w_deq, x_half, out, M, N, K,
                                             stream, scratch);
  }
  if (N >= M) {
    return RunCutlassGemmImpl<Gemm64x128x64>(w_deq, x_half, out, M, N, K,
                                            stream, scratch);
  }
  return RunCutlassGemmImpl<Gemm128x64x64>(w_deq, x_half, out, M, N, K, stream,
                                          scratch);
}

Error RunCutlassWeightOnly(const half *w_deq, const half *x_half, float *out,
                           int64_t M, int64_t N, int64_t K,
                           cudaStream_t stream, ScratchAllocator &scratch) {
  return RunCutlassGemm(w_deq, x_half, out, M, N, K, stream, scratch);
}

struct DequantCache {
  const void *wq = nullptr;
  const void *scales = nullptr;
  const void *biases = nullptr;
  int64_t k = 0;
  int64_t n = 0;
  int64_t n_words = 0;
  int64_t n_groups = 0;
  int64_t group_size = 0;
  int bits = 0;
  int mode = -1;
  int device = -1;
  size_t bytes = 0;
  half *buffer = nullptr;

  ~DequantCache() {
    if (buffer) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }

  void Invalidate() {
    wq = nullptr;
    scales = nullptr;
    biases = nullptr;
    k = 0;
    n = 0;
    n_words = 0;
    n_groups = 0;
    group_size = 0;
    bits = 0;
    mode = -1;
    device = -1;
  }
};

static thread_local DequantCache g_dequant_cache;

inline bool UseDequantCache() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_CACHE");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

Error QuantizedMatmulCuda(AnyBuffer x, AnyBuffer wq, AnyBuffer scales,
                          std::optional<AnyBuffer> biases,
                          Result<AnyBuffer> out, int64_t group_size,
                          int64_t bits, int64_t mode, int64_t transpose,
                          cudaStream_t stream, ScratchAllocator scratch) {
  if (bits < 2 || bits > 8) {
    return MakeInvalid("CUDA quantized_matmul supports bits in [2, 8].");
  }
  if (transpose != 0) {
    return MakeInvalid(
        "CUDA quantized_matmul currently requires transpose=False.");
  }
  if (mode == 0 && !(bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
                     bits == 6 || bits == 7 || bits == 8)) {
    return MakeInvalid(
        "affine mode supports bits in {2,3,4,5,6,7,8} on CUDA.");
  }
  if (mode == 1 && bits != kBits4) {
    return MakeInvalid("nf4 mode requires bits=4 on CUDA.");
  }
  if (mode == 2 && (group_size != 32 || bits != kBits4)) {
    return MakeInvalid("mxfp4 requires group_size=32 and bits=4.");
  }
  if (mode == 3 && (group_size != 32 || bits != kBits8)) {
    return MakeInvalid("mxfp8 requires group_size=32 and bits=8.");
  }
  if (mode == 4 && (group_size != 16 || bits != kBits4)) {
    return MakeInvalid("nvfp4 requires group_size=16 and bits=4.");
  }
  if (mode == 5 && (group_size != 16 || bits != kBits8)) {
    return MakeInvalid("nvfp8 requires group_size=16 and bits=8.");
  }

  Span<const int64_t> x_dims = x.dimensions();
  Span<const int64_t> w_dims = wq.dimensions();
  Span<const int64_t> s_dims = scales.dimensions();
  Span<const int64_t> o_dims = out->dimensions();

  if (x_dims.size() != 2 || w_dims.size() != 2 || s_dims.size() != 2 ||
      o_dims.size() != 2) {
    return MakeInvalid("All inputs/outputs must be rank-2 matrices.");
  }

  int64_t M = x_dims[0];
  int64_t K = x_dims[1];
  int64_t K_w = w_dims[0];
  int64_t N_words = w_dims[1];

  if (K_w != K) {
    return MakeInvalid("Weight K dimension does not match input K.");
  }
  if (group_size <= 0) {
    return MakeInvalid("group_size must be positive.");
  }

  if (s_dims[0] != K) {
    return MakeInvalid("scales shape must be (K, N/group_size).");
  }
  int64_t n_groups = s_dims[1];
  int64_t N = n_groups * group_size;
  int64_t expected_words =
      (static_cast<int64_t>(N) * bits + 31) / 32;
  if (N_words != expected_words) {
    return MakeInvalid("packed weight shape does not match N and bits.");
  }

  if (o_dims[0] != M || o_dims[1] != N) {
    return MakeInvalid("output shape must be (M, N).");
  }

  if (wq.element_type() != xla::ffi::DataType::U32) {
    return MakeInvalid("wq must be uint32 packed codes.");
  }
  if (out->element_type() != xla::ffi::DataType::F32) {
    return MakeInvalid("output must be float32.");
  }

  if (mode == 0) {
    if (!biases.has_value()) {
      return MakeInvalid("affine mode requires biases.");
    }
    Span<const int64_t> b_dims = biases->dimensions();
    if (b_dims.size() != 2 || b_dims[0] != K || b_dims[1] != n_groups) {
      return MakeInvalid("biases shape must match scales shape.");
    }
  } else if (mode == 1) {
    if (biases.has_value()) {
      return MakeInvalid("nf4 mode does not accept biases.");
    }
  } else if (mode == 2 || mode == 3 || mode == 4 || mode == 5) {
    if (biases.has_value()) {
      return MakeInvalid("mxfp/nvfp modes do not accept biases.");
    }
  } else {
    return MakeInvalid("Unsupported mode for CUDA quantized_matmul.");
  }

  if (!IsFiniteInt64(M) || !IsFiniteInt64(N) || !IsFiniteInt64(K)) {
    return MakeInvalid("M/N/K are too large for CUDA GEMM.");
  }

  if (UseFusedQmm()) {
    const uint32_t *wq_ptr =
        static_cast<const uint32_t *>(wq.untyped_data());
    float *out_ptr = out->typed_data<float>();
    auto scales_dtype = scales.element_type();

    if (mode == 0 || mode == 1) {
      if (scales_dtype == xla::ffi::DataType::F32) {
        const float *scales_ptr =
            static_cast<const float *>(scales.untyped_data());
        const float *bias_ptr = nullptr;
        if (mode == 0) {
          bias_ptr = static_cast<const float *>(biases->untyped_data());
        }
        if (x.element_type() == xla::ffi::DataType::F16) {
          return LaunchQmmFusedKernel<half, float, float>(
              static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
              bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, stream);
        }
        if (x.element_type() == xla::ffi::DataType::F32) {
          return LaunchQmmFusedKernel<float, float, float>(
              static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
              bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, stream);
        }
        if (x.element_type() == xla::ffi::DataType::BF16) {
          return LaunchQmmFusedKernel<__nv_bfloat16, float, float>(
              static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
              scales_ptr, bias_ptr, out_ptr, M, N, K, N_words, n_groups,
              group_size, bits, mode, stream);
        }
        return MakeInvalid("x dtype must be float16/float32/bfloat16.");
      }
      if (scales_dtype == xla::ffi::DataType::F16) {
        const half *scales_ptr =
            static_cast<const half *>(scales.untyped_data());
        const half *bias_ptr = nullptr;
        if (mode == 0) {
          bias_ptr = static_cast<const half *>(biases->untyped_data());
        }
        if (x.element_type() == xla::ffi::DataType::F16) {
          return LaunchQmmFusedKernel<half, half, half>(
              static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
              bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, stream);
        }
        if (x.element_type() == xla::ffi::DataType::F32) {
          return LaunchQmmFusedKernel<float, half, half>(
              static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
              bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, stream);
        }
        if (x.element_type() == xla::ffi::DataType::BF16) {
          return LaunchQmmFusedKernel<__nv_bfloat16, half, half>(
              static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
              scales_ptr, bias_ptr, out_ptr, M, N, K, N_words, n_groups,
              group_size, bits, mode, stream);
        }
        return MakeInvalid("x dtype must be float16/float32/bfloat16.");
      }
      if (scales_dtype == xla::ffi::DataType::BF16) {
        const __nv_bfloat16 *scales_ptr =
            static_cast<const __nv_bfloat16 *>(scales.untyped_data());
        const __nv_bfloat16 *bias_ptr = nullptr;
        if (mode == 0) {
          bias_ptr =
              static_cast<const __nv_bfloat16 *>(biases->untyped_data());
        }
        if (x.element_type() == xla::ffi::DataType::F16) {
          return LaunchQmmFusedKernel<half, __nv_bfloat16, __nv_bfloat16>(
              static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
              bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, stream);
        }
        if (x.element_type() == xla::ffi::DataType::F32) {
          return LaunchQmmFusedKernel<float, __nv_bfloat16, __nv_bfloat16>(
              static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
              bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, stream);
        }
        if (x.element_type() == xla::ffi::DataType::BF16) {
          return LaunchQmmFusedKernel<__nv_bfloat16, __nv_bfloat16,
                                      __nv_bfloat16>(
              static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
              scales_ptr, bias_ptr, out_ptr, M, N, K, N_words, n_groups,
              group_size, bits, mode, stream);
        }
        return MakeInvalid("x dtype must be float16/float32/bfloat16.");
      }
      return MakeInvalid(
          "scales dtype must be float32/float16/bfloat16 for affine/nf4.");
    }

    if (scales_dtype == xla::ffi::DataType::U8) {
      const uint8_t *scales_ptr =
          static_cast<const uint8_t *>(scales.untyped_data());
      if (x.element_type() == xla::ffi::DataType::F16) {
        return LaunchQmmFusedKernel<half, uint8_t, uint8_t>(
            static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
            nullptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
            mode, stream);
      }
      if (x.element_type() == xla::ffi::DataType::F32) {
        return LaunchQmmFusedKernel<float, uint8_t, uint8_t>(
            static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
            nullptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
            mode, stream);
      }
      if (x.element_type() == xla::ffi::DataType::BF16) {
        return LaunchQmmFusedKernel<__nv_bfloat16, uint8_t, uint8_t>(
            static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
            scales_ptr, nullptr, out_ptr, M, N, K, N_words, n_groups,
            group_size, bits, mode, stream);
      }
      return MakeInvalid("x dtype must be float16/float32/bfloat16.");
    }

    return MakeInvalid("scales dtype must be uint8 for mxfp/nvfp modes.");
  }

  const uint32_t *wq_ptr = static_cast<const uint32_t *>(wq.untyped_data());
  half *w_deq = nullptr;
  bool need_dequantize = true;
  int current_device = -1;

  if (UseDequantCache()) {
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return MakeInternal("Failed to query current CUDA device.");
    }
    bool match = g_dequant_cache.buffer != nullptr &&
                 g_dequant_cache.device == current_device &&
                 g_dequant_cache.wq == wq_ptr &&
                 g_dequant_cache.scales == scales.untyped_data() &&
                 g_dequant_cache.biases ==
                     (biases.has_value() ? biases->untyped_data() : nullptr) &&
                 g_dequant_cache.k == K && g_dequant_cache.n == N &&
                 g_dequant_cache.n_words == N_words &&
                 g_dequant_cache.n_groups == n_groups &&
                 g_dequant_cache.group_size == group_size &&
                 g_dequant_cache.bits == bits && g_dequant_cache.mode == mode;
    if (match) {
      w_deq = g_dequant_cache.buffer;
      need_dequantize = false;
    } else {
      size_t w_deq_bytes =
          static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(half);
      if (g_dequant_cache.buffer == nullptr ||
          g_dequant_cache.bytes < w_deq_bytes ||
          g_dequant_cache.device != current_device) {
        if (g_dequant_cache.buffer) {
          cudaFree(g_dequant_cache.buffer);
          g_dequant_cache.buffer = nullptr;
          g_dequant_cache.bytes = 0;
        }
        cudaError_t alloc_err =
            cudaMalloc(&g_dequant_cache.buffer, w_deq_bytes);
        if (alloc_err != cudaSuccess) {
          return MakeInternal("Failed to allocate cached dequant buffer.");
        }
        g_dequant_cache.bytes = w_deq_bytes;
        g_dequant_cache.device = current_device;
      }
      g_dequant_cache.wq = wq_ptr;
      g_dequant_cache.scales = scales.untyped_data();
      g_dequant_cache.biases =
          biases.has_value() ? biases->untyped_data() : nullptr;
      g_dequant_cache.k = K;
      g_dequant_cache.n = N;
      g_dequant_cache.n_words = N_words;
      g_dequant_cache.n_groups = n_groups;
      g_dequant_cache.group_size = group_size;
      g_dequant_cache.bits = bits;
      g_dequant_cache.mode = mode;
      w_deq = g_dequant_cache.buffer;
      need_dequantize = true;
    }
  } else {
    size_t w_deq_bytes =
        static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(half);
    auto w_deq_opt = scratch.Allocate(w_deq_bytes, alignof(half));
    if (!w_deq_opt.has_value()) {
      return MakeInternal(
          "Failed to allocate scratch buffer for dequantized weights.");
    }
    void *w_deq_ptr = *w_deq_opt;
    w_deq = reinterpret_cast<half *>(w_deq_ptr);
    need_dequantize = true;
  }

  dim3 block(256);
  int64_t total = K * N;
  int64_t elems_per_block =
      static_cast<int64_t>(block.x) * kDequantElemsPerThread;
  dim3 grid(static_cast<uint32_t>((total + elems_per_block - 1) /
                                  elems_per_block));

  auto scales_dtype = scales.element_type();
  if (need_dequantize) {
    if (mode == 0) {
      auto biases_buf = *biases;
      auto bias_dtype = biases_buf.element_type();
      if (bias_dtype != scales_dtype) {
        return MakeInvalid("biases dtype must match scales dtype.");
      }
      if (scales_dtype == xla::ffi::DataType::F32) {
        switch (bits) {
        case 2:
          ResolveLaunchDequantAffineBits2F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 3:
          ResolveLaunchDequantAffineBits3F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 4:
          ResolveLaunchDequantAffineBits4F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 5:
          ResolveLaunchDequantAffineBits5F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 6:
          ResolveLaunchDequantAffineBits6F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 7:
          ResolveLaunchDequantAffineBits7F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 8:
          ResolveLaunchDequantAffineBits8F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        default:
          return MakeInvalid(
              "affine mode supports bits in {2,3,4,5,6,7,8} on CUDA.");
        }
      } else if (scales_dtype == xla::ffi::DataType::F16) {
        switch (bits) {
        case 2:
          ResolveLaunchDequantAffineBits2F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 3:
          ResolveLaunchDequantAffineBits3F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 4:
          ResolveLaunchDequantAffineBits4F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 5:
          ResolveLaunchDequantAffineBits5F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 6:
          ResolveLaunchDequantAffineBits6F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 7:
          ResolveLaunchDequantAffineBits7F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 8:
          ResolveLaunchDequantAffineBits8F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        default:
          return MakeInvalid(
              "affine mode supports bits in {2,3,4,5,6,7,8} on CUDA.");
        }
      } else if (scales_dtype == xla::ffi::DataType::BF16) {
        switch (bits) {
        case 2:
          ResolveLaunchDequantAffineBits2BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 3:
          ResolveLaunchDequantAffineBits3BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 4:
          ResolveLaunchDequantAffineBits4BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 5:
          ResolveLaunchDequantAffineBits5BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 6:
          ResolveLaunchDequantAffineBits6BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 7:
          ResolveLaunchDequantAffineBits7BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 8:
          ResolveLaunchDequantAffineBits8BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        default:
          return MakeInvalid(
              "affine mode supports bits in {2,3,4,5,6,7,8} on CUDA.");
        }
      } else {
        return MakeInvalid(
            "scales/biases dtype must be float32/float16/bfloat16.");
      }
    } else if (mode == 1) {
      if (scales_dtype == xla::ffi::DataType::F32) {
        ResolveLaunchDequantNf4F32(group_size)(wq_ptr,
                            static_cast<const float *>(scales.untyped_data()),
                            w_deq, K, N, N_words, group_size, n_groups, grid,
                            block, stream);
      } else if (scales_dtype == xla::ffi::DataType::F16) {
        ResolveLaunchDequantNf4F16(group_size)(wq_ptr,
                            static_cast<const half *>(scales.untyped_data()),
                            w_deq, K, N, N_words, group_size, n_groups, grid,
                            block, stream);
      } else if (scales_dtype == xla::ffi::DataType::BF16) {
        ResolveLaunchDequantNf4BF16(group_size)(
            wq_ptr, static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
      } else {
        return MakeInvalid(
            "scales dtype must be float32/float16/bfloat16 for nf4.");
      }
    } else if (mode == 2) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
        return MakeInvalid("mxfp4 requires scales dtype uint8 and bits=4.");
      }
      LaunchDequantMxFp4(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else if (mode == 3) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
        return MakeInvalid("mxfp8 requires scales dtype uint8 and bits=8.");
      }
      LaunchDequantMxFp8(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else if (mode == 4) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
        return MakeInvalid("nvfp4 requires scales dtype uint8 and bits=4.");
      }
      LaunchDequantNvFp4(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else if (mode == 5) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
        return MakeInvalid("nvfp8 requires scales dtype uint8 and bits=8.");
      }
      LaunchDequantNvFp8(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else {
      return MakeInvalid("Unsupported mode for CUDA quantized_matmul.");
    }

    if (Error err = CheckCuda(cudaPeekAtLastError(),
                              "dequant kernel launch");
        err.failure()) {
      return err;
    }
  }

  const half *x_half = nullptr;
  size_t x_elems = static_cast<size_t>(M) * static_cast<size_t>(K);

  if (x.element_type() == xla::ffi::DataType::F16) {
    x_half = static_cast<const half *>(x.untyped_data());
  } else {
    size_t x_half_bytes = x_elems * sizeof(half);
    auto x_half_opt = scratch.Allocate(x_half_bytes, alignof(half));
    if (!x_half_opt.has_value()) {
      return MakeInternal("Failed to allocate scratch buffer for input cast.");
    }
    void *x_half_ptr = *x_half_opt;
    half *x_half_out = reinterpret_cast<half *>(x_half_ptr);
    dim3 cblock(256);
    dim3 cgrid(static_cast<uint32_t>((x_elems + cblock.x - 1) / cblock.x));
    if (x.element_type() == xla::ffi::DataType::F32) {
      convert_f32_to_f16<<<cgrid, cblock, 0, stream>>>(
          static_cast<const float *>(x.untyped_data()), x_half_out,
          static_cast<int64_t>(x_elems));
    } else if (x.element_type() == xla::ffi::DataType::BF16) {
      convert_bf16_to_f16<<<cgrid, cblock, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(x.untyped_data()), x_half_out,
          static_cast<int64_t>(x_elems));
    } else {
      return MakeInvalid("x dtype must be float16/float32/bfloat16.");
    }
    if (Error err =
            CheckCuda(cudaPeekAtLastError(), "input cast kernel launch");
        err.failure()) {
      return err;
    }
    x_half = x_half_out;
  }

  GemmBackend backend = GetGemmBackend();
  cublasComputeType_t compute_type = GetCublasComputeType();

  if (backend == GemmBackend::kCublasLt) {
    Error err =
        RunCublasLtGemm(w_deq, x_half, out->typed_data<float>(), M, N, K,
                        stream, scratch, compute_type);
    if (!err.failure()) {
      return Error::Success();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  } else if (backend == GemmBackend::kCutlass) {
    Error err = RunCutlassGemm(w_deq, x_half, out->typed_data<float>(), M, N,
                               K, stream, scratch);
    if (!err.failure()) {
      return Error::Success();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  } else if (backend == GemmBackend::kCutlassTuned) {
    Error err = RunCutlassGemmTuned(w_deq, x_half, out->typed_data<float>(), M,
                                    N, K, stream, scratch);
    if (!err.failure()) {
      return Error::Success();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  } else if (backend == GemmBackend::kCutlassW4A16) {
    Error err = RunCutlassWeightOnly(w_deq, x_half, out->typed_data<float>(),
                                     M, N, K, stream, scratch);
    if (!err.failure()) {
      return Error::Success();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  }

  cublasHandle_t handle;
  if (Error err = g_cublas_cache.Get(&handle, stream); err.failure()) {
    return err;
  }
  if (Error err =
          CheckCublas(cublasSetMathMode(handle, GetCublasMathMode(compute_type)),
                      "cublasSetMathMode");
      err.failure()) {
    return err;
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  cublasStatus_t gemm_status =
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(N),
                   static_cast<int>(M), static_cast<int>(K), &alpha, w_deq,
                   CUDA_R_16F, static_cast<int>(N), x_half, CUDA_R_16F,
                   static_cast<int>(K), &beta, out->typed_data<float>(),
                   CUDA_R_32F, static_cast<int>(N), compute_type, algo);

  if (Error err = CheckCublas(gemm_status, "cublasGemmEx"); err.failure()) {
    return err;
  }

  return Error::Success();
}

} // namespace

extern "C" XLA_FFI_Error *ejk_qmm_cuda(XLA_FFI_CallFrame *call_frame) {
  static auto handler = xla::ffi::Ffi::Bind()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .OptionalArg<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Attr<int64_t>("group_size")
                            .Attr<int64_t>("bits")
                            .Attr<int64_t>("mode")
                            .Attr<int64_t>("transpose")
                            .Ctx<PlatformStream<cudaStream_t>>()
                            .Ctx<ScratchAllocator>()
                            .To(QuantizedMatmulCuda);
  return handler->Call(call_frame);
}
