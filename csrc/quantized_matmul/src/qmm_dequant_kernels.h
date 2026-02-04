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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

constexpr int kDequantElemsPerThread = 4;

static __device__ __constant__ float kNF4Table[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

static __device__ __constant__ float kE2M1Table[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static __device__ __constant__ float kE4M3Table[256] = {
    0.0f,
    0.001953125f,
    0.00390625f,
    0.005859375f,
    0.0078125f,
    0.009765625f,
    0.01171875f,
    0.013671875f,
    0.015625f,
    0.017578125f,
    0.01953125f,
    0.021484375f,
    0.0234375f,
    0.025390625f,
    0.02734375f,
    0.029296875f,
    0.03124999814f,
    0.03515624627f,
    0.03906249627f,
    0.04296874627f,
    0.04687499627f,
    0.05078124627f,
    0.05468749627f,
    0.05859374627f,
    0.0625f,
    0.0703125f,
    0.078125f,
    0.0859375f,
    0.09375f,
    0.1015625f,
    0.109375f,
    0.1171875f,
    0.1249999925f,
    0.1406249851f,
    0.1562499851f,
    0.1718749851f,
    0.1874999851f,
    0.2031249851f,
    0.2187499851f,
    0.2343749851f,
    0.25f,
    0.28125f,
    0.3125f,
    0.34375f,
    0.375f,
    0.40625f,
    0.4375f,
    0.46875f,
    0.5f,
    0.5625f,
    0.625f,
    0.6875f,
    0.75f,
    0.8125f,
    0.875f,
    0.9375f,
    1.0f,
    1.125f,
    1.25f,
    1.375f,
    1.5f,
    1.625f,
    1.75f,
    1.875f,
    2.0f,
    2.25f,
    2.5f,
    2.75f,
    3.0f,
    3.25f,
    3.5f,
    3.75f,
    4.0f,
    4.5f,
    5.0f,
    5.5f,
    6.0f,
    6.5f,
    7.0f,
    7.5f,
    8.0f,
    9.0f,
    10.0f,
    11.0f,
    12.0f,
    13.0f,
    14.0f,
    15.0f,
    16.0f,
    18.0f,
    20.0f,
    22.0f,
    24.0f,
    26.0f,
    28.0f,
    30.0f,
    32.0f,
    36.0f,
    40.0f,
    44.0f,
    48.0f,
    52.0f,
    56.0f,
    60.0f,
    64.0f,
    72.0f,
    80.0f,
    88.0f,
    96.0f,
    104.0f,
    112.0f,
    120.0f,
    128.0f,
    144.0f,
    160.0f,
    176.0f,
    192.0f,
    208.0f,
    224.0f,
    240.0f,
    256.0f,
    288.0f,
    320.0f,
    352.0f,
    384.0f,
    416.0f,
    448.0f,
    0.0f,
    -0.0f,
    -0.001953125f,
    -0.00390625f,
    -0.005859375f,
    -0.0078125f,
    -0.009765625f,
    -0.01171875f,
    -0.013671875f,
    -0.015625f,
    -0.017578125f,
    -0.01953125f,
    -0.021484375f,
    -0.0234375f,
    -0.025390625f,
    -0.02734375f,
    -0.029296875f,
    -0.03124999814f,
    -0.03515624627f,
    -0.03906249627f,
    -0.04296874627f,
    -0.04687499627f,
    -0.05078124627f,
    -0.05468749627f,
    -0.05859374627f,
    -0.0625f,
    -0.0703125f,
    -0.078125f,
    -0.0859375f,
    -0.09375f,
    -0.1015625f,
    -0.109375f,
    -0.1171875f,
    -0.1249999925f,
    -0.1406249851f,
    -0.1562499851f,
    -0.1718749851f,
    -0.1874999851f,
    -0.2031249851f,
    -0.2187499851f,
    -0.2343749851f,
    -0.25f,
    -0.28125f,
    -0.3125f,
    -0.34375f,
    -0.375f,
    -0.40625f,
    -0.4375f,
    -0.46875f,
    -0.5f,
    -0.5625f,
    -0.625f,
    -0.6875f,
    -0.75f,
    -0.8125f,
    -0.875f,
    -0.9375f,
    -1.0f,
    -1.125f,
    -1.25f,
    -1.375f,
    -1.5f,
    -1.625f,
    -1.75f,
    -1.875f,
    -2.0f,
    -2.25f,
    -2.5f,
    -2.75f,
    -3.0f,
    -3.25f,
    -3.5f,
    -3.75f,
    -4.0f,
    -4.5f,
    -5.0f,
    -5.5f,
    -6.0f,
    -6.5f,
    -7.0f,
    -7.5f,
    -8.0f,
    -9.0f,
    -10.0f,
    -11.0f,
    -12.0f,
    -13.0f,
    -14.0f,
    -15.0f,
    -16.0f,
    -18.0f,
    -20.0f,
    -22.0f,
    -24.0f,
    -26.0f,
    -28.0f,
    -30.0f,
    -32.0f,
    -36.0f,
    -40.0f,
    -44.0f,
    -48.0f,
    -52.0f,
    -56.0f,
    -60.0f,
    -64.0f,
    -72.0f,
    -80.0f,
    -88.0f,
    -96.0f,
    -104.0f,
    -112.0f,
    -120.0f,
    -128.0f,
    -144.0f,
    -160.0f,
    -176.0f,
    -192.0f,
    -208.0f,
    -224.0f,
    -240.0f,
    -256.0f,
    -288.0f,
    -320.0f,
    -352.0f,
    -384.0f,
    -416.0f,
    -448.0f,
    0.0f,
};

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ half ToHalf(float v) { return __float2half_rn(v); }

template <int Bits, typename ScaleT, typename BiasT>
__global__ void dequant_affine_int(const uint32_t *wq, const ScaleT *scales,
                                   const BiasT *biases, half *out, int64_t K,
                                   int64_t N, int64_t n_words,
                                   int64_t group_size, int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    int64_t bit_offset = n * Bits;
    int64_t word_idx = bit_offset >> 5;
    int32_t shift = static_cast<int32_t>(bit_offset & 31);
    const uint32_t *row = wq + k * n_words;
    uint32_t low_word = row[word_idx];
    int32_t low_bits = (shift + Bits > 32) ? (32 - shift) : Bits;
    int32_t high_bits = Bits - low_bits;
    uint32_t low_mask = (uint32_t(1) << low_bits) - 1u;
    uint32_t low = (low_word >> shift) & low_mask;
    uint32_t high = 0;
    if (high_bits > 0) {
      uint32_t high_mask = (uint32_t(1) << high_bits) - 1u;
      high = row[word_idx + 1] & high_mask;
    }
    uint32_t q = low | (high << low_bits);
    int64_t g = n / group_size;
    float scale = ToFloat(scales[k * n_groups + g]);
    float bias = ToFloat(biases[k * n_groups + g]);
    float val = static_cast<float>(q) * scale + bias;
    out[idx] = ToHalf(val);
  }
}

template <int Bits, int GroupSize, typename ScaleT, typename BiasT>
__global__ void dequant_affine_int_gs(const uint32_t *wq, const ScaleT *scales,
                                      const BiasT *biases, half *out,
                                      int64_t K, int64_t N, int64_t n_words,
                                      int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    int64_t bit_offset = n * Bits;
    int64_t word_idx = bit_offset >> 5;
    int32_t shift = static_cast<int32_t>(bit_offset & 31);
    const uint32_t *row = wq + k * n_words;
    uint32_t low_word = row[word_idx];
    int32_t low_bits = (shift + Bits > 32) ? (32 - shift) : Bits;
    int32_t high_bits = Bits - low_bits;
    uint32_t low_mask = (uint32_t(1) << low_bits) - 1u;
    uint32_t low = (low_word >> shift) & low_mask;
    uint32_t high = 0;
    if (high_bits > 0) {
      uint32_t high_mask = (uint32_t(1) << high_bits) - 1u;
      high = row[word_idx + 1] & high_mask;
    }
    uint32_t q = low | (high << low_bits);
    int64_t g = n / GroupSize;
    float scale = ToFloat(scales[k * n_groups + g]);
    float bias = ToFloat(biases[k * n_groups + g]);
    float val = static_cast<float>(q) * scale + bias;
    out[idx] = ToHalf(val);
  }
}

template <int Bits, typename ScaleT>
__global__ void dequant_nf4_int(const uint32_t *wq, const ScaleT *scales,
                                half *out, int64_t K, int64_t N,
                                int64_t n_words, int64_t group_size,
                                int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    constexpr int32_t values_per_word = 32 / Bits;
    int64_t word_idx = n / values_per_word;
    int32_t lane = static_cast<int32_t>(n % values_per_word);
    uint32_t word = wq[k * n_words + word_idx];
    constexpr uint32_t mask = (Bits == 8) ? 0xFFu : 0xFu;
    uint32_t q = (word >> (lane * Bits)) & mask;
    int64_t g = n / group_size;
    float scale = ToFloat(scales[k * n_groups + g]);
    float val = kNF4Table[q] * scale;
    out[idx] = ToHalf(val);
  }
}

template <int Bits, int GroupSize, typename ScaleT>
__global__ void dequant_nf4_int_gs(const uint32_t *wq, const ScaleT *scales,
                                   half *out, int64_t K, int64_t N,
                                   int64_t n_words, int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    constexpr int32_t values_per_word = 32 / Bits;
    int64_t word_idx = n / values_per_word;
    int32_t lane = static_cast<int32_t>(n % values_per_word);
    uint32_t word = wq[k * n_words + word_idx];
    constexpr uint32_t mask = (Bits == 8) ? 0xFFu : 0xFu;
    uint32_t q = (word >> (lane * Bits)) & mask;
    int64_t g = n / GroupSize;
    float scale = ToFloat(scales[k * n_groups + g]);
    float val = kNF4Table[q] * scale;
    out[idx] = ToHalf(val);
  }
}

template <int GroupSize>
__global__ void dequant_mxfp4(const uint32_t *wq, const uint8_t *scales,
                              half *out, int64_t K, int64_t N, int64_t n_words,
                              int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    int32_t values_per_word = 8;
    int64_t word_idx = n / values_per_word;
    int32_t lane = static_cast<int32_t>(n % values_per_word);
    uint32_t word = wq[k * n_words + word_idx];
    uint32_t q = (word >> (lane * 4)) & 0xFu;
    int64_t g = n / GroupSize;
    int8_t exp = static_cast<int8_t>(scales[k * n_groups + g]);
    float scale = exp2f(static_cast<float>(exp));
    float val = kE2M1Table[q] * scale;
    out[idx] = ToHalf(val);
  }
}

template <int GroupSize>
__global__ void dequant_mxfp8(const uint32_t *wq, const uint8_t *scales,
                              half *out, int64_t K, int64_t N, int64_t n_words,
                              int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    int32_t values_per_word = 4;
    int64_t word_idx = n / values_per_word;
    int32_t lane = static_cast<int32_t>(n % values_per_word);
    uint32_t word = wq[k * n_words + word_idx];
    uint32_t q = (word >> (lane * 8)) & 0xFFu;
    int64_t g = n / GroupSize;
    int8_t exp = static_cast<int8_t>(scales[k * n_groups + g]);
    float scale = exp2f(static_cast<float>(exp));
    float val = kE4M3Table[q] * scale;
    out[idx] = ToHalf(val);
  }
}

template <int GroupSize>
__global__ void dequant_nvfp4(const uint32_t *wq, const uint8_t *scales,
                              half *out, int64_t K, int64_t N, int64_t n_words,
                              int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    int32_t values_per_word = 8;
    int64_t word_idx = n / values_per_word;
    int32_t lane = static_cast<int32_t>(n % values_per_word);
    uint32_t word = wq[k * n_words + word_idx];
    uint32_t q = (word >> (lane * 4)) & 0xFu;
    int64_t g = n / GroupSize;
    uint8_t scale_code = scales[k * n_groups + g];
    float scale = kE4M3Table[scale_code];
    float val = kE2M1Table[q] * scale;
    out[idx] = ToHalf(val);
  }
}

template <int GroupSize>
__global__ void dequant_nvfp8(const uint32_t *wq, const uint8_t *scales,
                              half *out, int64_t K, int64_t N, int64_t n_words,
                              int64_t n_groups) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k = idx / N;
    int64_t n = idx - k * N;
    int32_t values_per_word = 4;
    int64_t word_idx = n / values_per_word;
    int32_t lane = static_cast<int32_t>(n % values_per_word);
    uint32_t word = wq[k * n_words + word_idx];
    uint32_t q = (word >> (lane * 8)) & 0xFFu;
    int64_t g = n / GroupSize;
    uint8_t scale_code = scales[k * n_groups + g];
    float scale = kE4M3Table[scale_code];
    float val = kE4M3Table[q] * scale;
    out[idx] = ToHalf(val);
  }
}
