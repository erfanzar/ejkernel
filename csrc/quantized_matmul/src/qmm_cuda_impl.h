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

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

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

  size_t w_deq_bytes =
      static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(half);
  auto w_deq_opt = scratch.Allocate(w_deq_bytes, alignof(half));
  if (!w_deq_opt.has_value()) {
    return MakeInternal(
        "Failed to allocate scratch buffer for dequantized weights.");
  }
  void *w_deq_ptr = *w_deq_opt;
  half *w_deq = reinterpret_cast<half *>(w_deq_ptr);
  const uint32_t *wq_ptr = static_cast<const uint32_t *>(wq.untyped_data());

  dim3 block(256);
  int64_t total = K * N;
  dim3 grid(static_cast<uint32_t>((total + block.x - 1) / block.x));

  auto scales_dtype = scales.element_type();
  if (mode == 0) {
    auto biases_buf = *biases;
    auto bias_dtype = biases_buf.element_type();
    if (bias_dtype != scales_dtype) {
      return MakeInvalid("biases dtype must match scales dtype.");
    }
    if (scales_dtype == xla::ffi::DataType::F32) {
      switch (bits) {
      case 2:
        LaunchDequantAffineBits2F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 3:
        LaunchDequantAffineBits3F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 4:
        LaunchDequantAffineBits4F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 5:
        LaunchDequantAffineBits5F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 6:
        LaunchDequantAffineBits6F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 7:
        LaunchDequantAffineBits7F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 8:
        LaunchDequantAffineBits8F32(
            wq_ptr, static_cast<const float *>(scales.untyped_data()),
            static_cast<const float *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      default:
        return MakeInvalid(
            "affine mode supports bits in {2,3,4,5,6,7,8} on CUDA.");
      }
    } else if (scales_dtype == xla::ffi::DataType::F16) {
      switch (bits) {
      case 2:
        LaunchDequantAffineBits2F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 3:
        LaunchDequantAffineBits3F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 4:
        LaunchDequantAffineBits4F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 5:
        LaunchDequantAffineBits5F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 6:
        LaunchDequantAffineBits6F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 7:
        LaunchDequantAffineBits7F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      case 8:
        LaunchDequantAffineBits8F16(
            wq_ptr, static_cast<const half *>(scales.untyped_data()),
            static_cast<const half *>(biases_buf.untyped_data()), w_deq, K, N,
            N_words, group_size, n_groups, grid, block, stream);
        break;
      default:
        return MakeInvalid(
            "affine mode supports bits in {2,3,4,5,6,7,8} on CUDA.");
      }
    } else if (scales_dtype == xla::ffi::DataType::BF16) {
      switch (bits) {
      case 2:
        LaunchDequantAffineBits2BF16(
            wq_ptr,
            static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
        break;
      case 3:
        LaunchDequantAffineBits3BF16(
            wq_ptr,
            static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
        break;
      case 4:
        LaunchDequantAffineBits4BF16(
            wq_ptr,
            static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
        break;
      case 5:
        LaunchDequantAffineBits5BF16(
            wq_ptr,
            static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
        break;
      case 6:
        LaunchDequantAffineBits6BF16(
            wq_ptr,
            static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
        break;
      case 7:
        LaunchDequantAffineBits7BF16(
            wq_ptr,
            static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
        break;
      case 8:
        LaunchDequantAffineBits8BF16(
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
      LaunchDequantNf4F32(wq_ptr,
                          static_cast<const float *>(scales.untyped_data()),
                          w_deq, K, N, N_words, group_size, n_groups, grid,
                          block, stream);
    } else if (scales_dtype == xla::ffi::DataType::F16) {
      LaunchDequantNf4F16(wq_ptr,
                          static_cast<const half *>(scales.untyped_data()),
                          w_deq, K, N, N_words, group_size, n_groups, grid,
                          block, stream);
    } else if (scales_dtype == xla::ffi::DataType::BF16) {
      LaunchDequantNf4BF16(
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

  if (Error err = CheckCuda(cudaPeekAtLastError(), "dequant kernel launch");
      err.failure()) {
    return err;
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

  cublasHandle_t handle;
  if (Error err = CheckCublas(cublasCreate(&handle), "cublasCreate");
      err.failure()) {
    return err;
  }
  if (Error err =
          CheckCublas(cublasSetStream(handle, stream), "cublasSetStream");
      err.failure()) {
    cublasDestroy(handle);
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
                   CUDA_R_32F, static_cast<int>(N), CUBLAS_COMPUTE_32F, algo);

  cublasDestroy(handle);

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
