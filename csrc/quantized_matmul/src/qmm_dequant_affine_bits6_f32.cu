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
//
// This file is auto-generated. See "code_gen.py".

#include "qmm_dequant_kernels.h"

void LaunchDequantAffineBits6F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<6, float, float><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits6F32Gs8(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 8, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs9(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 9, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs10(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 10, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs11(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 11, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs12(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 12, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs13(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 13, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs14(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 14, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs15(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 15, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 16, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs17(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 17, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs18(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 18, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs19(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 19, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs20(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 20, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs21(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 21, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs22(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 22, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs23(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 23, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs24(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 24, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs25(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 25, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs26(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 26, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs27(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 27, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs28(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 28, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs29(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 29, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs30(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 30, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs31(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 31, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 32, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs33(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 33, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs34(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 34, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs35(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 35, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs36(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 36, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs37(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 37, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs38(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 38, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs39(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 39, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs40(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 40, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs41(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 41, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs42(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 42, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs43(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 43, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs44(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 44, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs45(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 45, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs46(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 46, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs47(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 47, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs48(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 48, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs49(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 49, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs50(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 50, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs51(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 51, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs52(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 52, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs53(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 53, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs54(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 54, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs55(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 55, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs56(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 56, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs57(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 57, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs58(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 58, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs59(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 59, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs60(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 60, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs61(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 61, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs62(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 62, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs63(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 63, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 64, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs65(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 65, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs66(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 66, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs67(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 67, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs68(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 68, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs69(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 69, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs70(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 70, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs71(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 71, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs72(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 72, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs73(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 73, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs74(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 74, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs75(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 75, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs76(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 76, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs77(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 77, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs78(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 78, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs79(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 79, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs80(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 80, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs81(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 81, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs82(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 82, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs83(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 83, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs84(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 84, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs85(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 85, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs86(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 86, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs87(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 87, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs88(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 88, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs89(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 89, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs90(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 90, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs91(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 91, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs92(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 92, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs93(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 93, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs94(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 94, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs95(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 95, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs96(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 96, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs97(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 97, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs98(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 98, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs99(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 99, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs100(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 100, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs101(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 101, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs102(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 102, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs103(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 103, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs104(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 104, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs105(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 105, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs106(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 106, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs107(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 107, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs108(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 108, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs109(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 109, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs110(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 110, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs111(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 111, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs112(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 112, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs113(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 113, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs114(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 114, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs115(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 115, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs116(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 116, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs117(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 117, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs118(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 118, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs119(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 119, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs120(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 120, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs121(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 121, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs122(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 122, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs123(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 123, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs124(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 124, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs125(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 125, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs126(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 126, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs127(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 127, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 128, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs129(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 129, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs130(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 130, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs131(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 131, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs132(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 132, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs133(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 133, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs134(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 134, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs135(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 135, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs136(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 136, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs137(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 137, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs138(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 138, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs139(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 139, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs140(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 140, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs141(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 141, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs142(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 142, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs143(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 143, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs144(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 144, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs145(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 145, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs146(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 146, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs147(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 147, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs148(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 148, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs149(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 149, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs150(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 150, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs151(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 151, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs152(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 152, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs153(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 153, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs154(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 154, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs155(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 155, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs156(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 156, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs157(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 157, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs158(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 158, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs159(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 159, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs160(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 160, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs161(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 161, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs162(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 162, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs163(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 163, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs164(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 164, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs165(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 165, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs166(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 166, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs167(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 167, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs168(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 168, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs169(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 169, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs170(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 170, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs171(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 171, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs172(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 172, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs173(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 173, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs174(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 174, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs175(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 175, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs176(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 176, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs177(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 177, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs178(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 178, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs179(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 179, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs180(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 180, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs181(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 181, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs182(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 182, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs183(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 183, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs184(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 184, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs185(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 185, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs186(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 186, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs187(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 187, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs188(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 188, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs189(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 189, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs190(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 190, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs191(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 191, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs192(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 192, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs193(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 193, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs194(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 194, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs195(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 195, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs196(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 196, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs197(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 197, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs198(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 198, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs199(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 199, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs200(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 200, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs201(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 201, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs202(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 202, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs203(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 203, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs204(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 204, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs205(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 205, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs206(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 206, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs207(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 207, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs208(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 208, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs209(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 209, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs210(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 210, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs211(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 211, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs212(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 212, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs213(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 213, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs214(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 214, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs215(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 215, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs216(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 216, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs217(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 217, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs218(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 218, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs219(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 219, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs220(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 220, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs221(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 221, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs222(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 222, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs223(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 223, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs224(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 224, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs225(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 225, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs226(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 226, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs227(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 227, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs228(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 228, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs229(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 229, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs230(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 230, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs231(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 231, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs232(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 232, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs233(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 233, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs234(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 234, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs235(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 235, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs236(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 236, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs237(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 237, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs238(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 238, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs239(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 239, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs240(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 240, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs241(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 241, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs242(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 242, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs243(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 243, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs244(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 244, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs245(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 245, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs246(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 246, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs247(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 247, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs248(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 248, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs249(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 249, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs250(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 250, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs251(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 251, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs252(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 252, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs253(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 253, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs254(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 254, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs255(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 255, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 256, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs257(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 257, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs258(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 258, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs259(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 259, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs260(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 260, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs261(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 261, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs262(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 262, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs263(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 263, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs264(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 264, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs265(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 265, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs266(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 266, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs267(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 267, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs268(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 268, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs269(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 269, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs270(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 270, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs271(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 271, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs272(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 272, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs273(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 273, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs274(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 274, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs275(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 275, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs276(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 276, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs277(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 277, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs278(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 278, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs279(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 279, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs280(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 280, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs281(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 281, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs282(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 282, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs283(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 283, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs284(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 284, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs285(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 285, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs286(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 286, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs287(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 287, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs288(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 288, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs289(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 289, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs290(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 290, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs291(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 291, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs292(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 292, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs293(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 293, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs294(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 294, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs295(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 295, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs296(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 296, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs297(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 297, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs298(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 298, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs299(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 299, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs300(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 300, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs301(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 301, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs302(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 302, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs303(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 303, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs304(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 304, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs305(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 305, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs306(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 306, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs307(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 307, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs308(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 308, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs309(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 309, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs310(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 310, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs311(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 311, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs312(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 312, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs313(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 313, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs314(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 314, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs315(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 315, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs316(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 316, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs317(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 317, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs318(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 318, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs319(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 319, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs320(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 320, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs321(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 321, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs322(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 322, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs323(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 323, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs324(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 324, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs325(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 325, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs326(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 326, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs327(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 327, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs328(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 328, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs329(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 329, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs330(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 330, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs331(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 331, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs332(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 332, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs333(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 333, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs334(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 334, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs335(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 335, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs336(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 336, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs337(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 337, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs338(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 338, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs339(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 339, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs340(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 340, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs341(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 341, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs342(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 342, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs343(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 343, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs344(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 344, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs345(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 345, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs346(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 346, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs347(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 347, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs348(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 348, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs349(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 349, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs350(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 350, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs351(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 351, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs352(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 352, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs353(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 353, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs354(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 354, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs355(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 355, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs356(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 356, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs357(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 357, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs358(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 358, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs359(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 359, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs360(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 360, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs361(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 361, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs362(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 362, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs363(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 363, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs364(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 364, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs365(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 365, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs366(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 366, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs367(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 367, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs368(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 368, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs369(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 369, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs370(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 370, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs371(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 371, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs372(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 372, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs373(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 373, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs374(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 374, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs375(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 375, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs376(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 376, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs377(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 377, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs378(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 378, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs379(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 379, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs380(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 380, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs381(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 381, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs382(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 382, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs383(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 383, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs384(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 384, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs385(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 385, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs386(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 386, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs387(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 387, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs388(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 388, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs389(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 389, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs390(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 390, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs391(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 391, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs392(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 392, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs393(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 393, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs394(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 394, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs395(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 395, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs396(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 396, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs397(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 397, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs398(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 398, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs399(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 399, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs400(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 400, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs401(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 401, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs402(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 402, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs403(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 403, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs404(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 404, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs405(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 405, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs406(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 406, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs407(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 407, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs408(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 408, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs409(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 409, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs410(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 410, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs411(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 411, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs412(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 412, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs413(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 413, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs414(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 414, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs415(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 415, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs416(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 416, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs417(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 417, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs418(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 418, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs419(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 419, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs420(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 420, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs421(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 421, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs422(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 422, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs423(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 423, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs424(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 424, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs425(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 425, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs426(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 426, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs427(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 427, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs428(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 428, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs429(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 429, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs430(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 430, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs431(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 431, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs432(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 432, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs433(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 433, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs434(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 434, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs435(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 435, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs436(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 436, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs437(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 437, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs438(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 438, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs439(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 439, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs440(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 440, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs441(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 441, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs442(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 442, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs443(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 443, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs444(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 444, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs445(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 445, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs446(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 446, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs447(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 447, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs448(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 448, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs449(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 449, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs450(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 450, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs451(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 451, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs452(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 452, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs453(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 453, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs454(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 454, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs455(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 455, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs456(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 456, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs457(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 457, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs458(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 458, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs459(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 459, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs460(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 460, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs461(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 461, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs462(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 462, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs463(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 463, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs464(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 464, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs465(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 465, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs466(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 466, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs467(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 467, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs468(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 468, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs469(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 469, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs470(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 470, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs471(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 471, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs472(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 472, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs473(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 473, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs474(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 474, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs475(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 475, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs476(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 476, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs477(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 477, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs478(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 478, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs479(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 479, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs480(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 480, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs481(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 481, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs482(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 482, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs483(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 483, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs484(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 484, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs485(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 485, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs486(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 486, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs487(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 487, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs488(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 488, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs489(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 489, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs490(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 490, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs491(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 491, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs492(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 492, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs493(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 493, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs494(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 494, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs495(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 495, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs496(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 496, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs497(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 497, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs498(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 498, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs499(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 499, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs500(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 500, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs501(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 501, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs502(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 502, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs503(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 503, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs504(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 504, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs505(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 505, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs506(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 506, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs507(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 507, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs508(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 508, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs509(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 509, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs510(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 510, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs511(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 511, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 512, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs513(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 513, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs514(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 514, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs515(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 515, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs516(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 516, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs517(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 517, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs518(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 518, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs519(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 519, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs520(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 520, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs521(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 521, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs522(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 522, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs523(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 523, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs524(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 524, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs525(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 525, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs526(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 526, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs527(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 527, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs528(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 528, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs529(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 529, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs530(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 530, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs531(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 531, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs532(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 532, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs533(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 533, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs534(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 534, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs535(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 535, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs536(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 536, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs537(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 537, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs538(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 538, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs539(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 539, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs540(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 540, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs541(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 541, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs542(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 542, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs543(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 543, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs544(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 544, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs545(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 545, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs546(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 546, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs547(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 547, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs548(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 548, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs549(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 549, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs550(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 550, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs551(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 551, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs552(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 552, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs553(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 553, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs554(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 554, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs555(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 555, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs556(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 556, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs557(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 557, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs558(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 558, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs559(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 559, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs560(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 560, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs561(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 561, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs562(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 562, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs563(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 563, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs564(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 564, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs565(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 565, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs566(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 566, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs567(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 567, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs568(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 568, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs569(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 569, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs570(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 570, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs571(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 571, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs572(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 572, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs573(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 573, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs574(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 574, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs575(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 575, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs576(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 576, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs577(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 577, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs578(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 578, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs579(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 579, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs580(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 580, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs581(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 581, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs582(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 582, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs583(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 583, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs584(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 584, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs585(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 585, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs586(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 586, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs587(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 587, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs588(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 588, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs589(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 589, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs590(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 590, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs591(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 591, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs592(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 592, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs593(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 593, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs594(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 594, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs595(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 595, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs596(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 596, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs597(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 597, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs598(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 598, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs599(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 599, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs600(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 600, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs601(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 601, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs602(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 602, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs603(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 603, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs604(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 604, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs605(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 605, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs606(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 606, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs607(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 607, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs608(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 608, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs609(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 609, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs610(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 610, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs611(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 611, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs612(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 612, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs613(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 613, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs614(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 614, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs615(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 615, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs616(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 616, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs617(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 617, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs618(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 618, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs619(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 619, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs620(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 620, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs621(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 621, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs622(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 622, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs623(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 623, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs624(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 624, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs625(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 625, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs626(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 626, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs627(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 627, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs628(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 628, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs629(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 629, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs630(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 630, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs631(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 631, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs632(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 632, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs633(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 633, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs634(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 634, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs635(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 635, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs636(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 636, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs637(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 637, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs638(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 638, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs639(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 639, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs640(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 640, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs641(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 641, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs642(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 642, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs643(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 643, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs644(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 644, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs645(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 645, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs646(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 646, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs647(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 647, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs648(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 648, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs649(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 649, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs650(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 650, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs651(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 651, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs652(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 652, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs653(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 653, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs654(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 654, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs655(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 655, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs656(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 656, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs657(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 657, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs658(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 658, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs659(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 659, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs660(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 660, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs661(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 661, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs662(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 662, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs663(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 663, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs664(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 664, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs665(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 665, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs666(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 666, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs667(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 667, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs668(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 668, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs669(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 669, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs670(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 670, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs671(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 671, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs672(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 672, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs673(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 673, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs674(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 674, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs675(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 675, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs676(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 676, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs677(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 677, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs678(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 678, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs679(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 679, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs680(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 680, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs681(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 681, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs682(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 682, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs683(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 683, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs684(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 684, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs685(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 685, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs686(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 686, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs687(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 687, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs688(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 688, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs689(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 689, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs690(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 690, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs691(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 691, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs692(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 692, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs693(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 693, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs694(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 694, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs695(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 695, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs696(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 696, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs697(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 697, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs698(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 698, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs699(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 699, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs700(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 700, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs701(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 701, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs702(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 702, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs703(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 703, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs704(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 704, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs705(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 705, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs706(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 706, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs707(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 707, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs708(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 708, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs709(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 709, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs710(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 710, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs711(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 711, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs712(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 712, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs713(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 713, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs714(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 714, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs715(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 715, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs716(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 716, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs717(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 717, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs718(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 718, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs719(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 719, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs720(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 720, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs721(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 721, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs722(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 722, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs723(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 723, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs724(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 724, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs725(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 725, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs726(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 726, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs727(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 727, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs728(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 728, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs729(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 729, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs730(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 730, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs731(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 731, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs732(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 732, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs733(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 733, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs734(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 734, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs735(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 735, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs736(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 736, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs737(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 737, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs738(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 738, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs739(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 739, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs740(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 740, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs741(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 741, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs742(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 742, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs743(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 743, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs744(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 744, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs745(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 745, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs746(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 746, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs747(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 747, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs748(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 748, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs749(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 749, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs750(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 750, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs751(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 751, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs752(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 752, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs753(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 753, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs754(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 754, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs755(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 755, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs756(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 756, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs757(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 757, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs758(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 758, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs759(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 759, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs760(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 760, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs761(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 761, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs762(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 762, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs763(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 763, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs764(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 764, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs765(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 765, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs766(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 766, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs767(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 767, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs768(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 768, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs769(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 769, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs770(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 770, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs771(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 771, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs772(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 772, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs773(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 773, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs774(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 774, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs775(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 775, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs776(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 776, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs777(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 777, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs778(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 778, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs779(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 779, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs780(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 780, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs781(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 781, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs782(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 782, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs783(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 783, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs784(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 784, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs785(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 785, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs786(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 786, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs787(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 787, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs788(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 788, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs789(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 789, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs790(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 790, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs791(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 791, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs792(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 792, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs793(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 793, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs794(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 794, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs795(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 795, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs796(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 796, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs797(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 797, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs798(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 798, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs799(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 799, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs800(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 800, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs801(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 801, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs802(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 802, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs803(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 803, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs804(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 804, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs805(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 805, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs806(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 806, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs807(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 807, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs808(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 808, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs809(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 809, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs810(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 810, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs811(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 811, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs812(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 812, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs813(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 813, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs814(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 814, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs815(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 815, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs816(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 816, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs817(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 817, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs818(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 818, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs819(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 819, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs820(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 820, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs821(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 821, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs822(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 822, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs823(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 823, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs824(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 824, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs825(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 825, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs826(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 826, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs827(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 827, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs828(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 828, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs829(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 829, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs830(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 830, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs831(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 831, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs832(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 832, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs833(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 833, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs834(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 834, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs835(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 835, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs836(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 836, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs837(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 837, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs838(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 838, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs839(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 839, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs840(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 840, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs841(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 841, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs842(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 842, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs843(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 843, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs844(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 844, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs845(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 845, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs846(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 846, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs847(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 847, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs848(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 848, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs849(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 849, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs850(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 850, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs851(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 851, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs852(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 852, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs853(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 853, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs854(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 854, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs855(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 855, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs856(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 856, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs857(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 857, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs858(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 858, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs859(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 859, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs860(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 860, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs861(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 861, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs862(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 862, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs863(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 863, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs864(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 864, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs865(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 865, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs866(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 866, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs867(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 867, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs868(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 868, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs869(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 869, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs870(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 870, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs871(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 871, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs872(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 872, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs873(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 873, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs874(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 874, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs875(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 875, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs876(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 876, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs877(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 877, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs878(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 878, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs879(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 879, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs880(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 880, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs881(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 881, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs882(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 882, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs883(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 883, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs884(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 884, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs885(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 885, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs886(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 886, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs887(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 887, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs888(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 888, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs889(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 889, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs890(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 890, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs891(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 891, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs892(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 892, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs893(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 893, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs894(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 894, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs895(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 895, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs896(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 896, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs897(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 897, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs898(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 898, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs899(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 899, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs900(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 900, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs901(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 901, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs902(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 902, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs903(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 903, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs904(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 904, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs905(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 905, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs906(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 906, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs907(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 907, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs908(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 908, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs909(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 909, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs910(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 910, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs911(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 911, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs912(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 912, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs913(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 913, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs914(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 914, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs915(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 915, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs916(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 916, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs917(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 917, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs918(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 918, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs919(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 919, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs920(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 920, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs921(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 921, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs922(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 922, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs923(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 923, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs924(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 924, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs925(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 925, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs926(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 926, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs927(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 927, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs928(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 928, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs929(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 929, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs930(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 930, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs931(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 931, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs932(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 932, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs933(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 933, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs934(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 934, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs935(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 935, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs936(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 936, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs937(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 937, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs938(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 938, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs939(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 939, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs940(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 940, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs941(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 941, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs942(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 942, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs943(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 943, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs944(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 944, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs945(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 945, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs946(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 946, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs947(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 947, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs948(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 948, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs949(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 949, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs950(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 950, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs951(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 951, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs952(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 952, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs953(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 953, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs954(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 954, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs955(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 955, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs956(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 956, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs957(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 957, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs958(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 958, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs959(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 959, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs960(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 960, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs961(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 961, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs962(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 962, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs963(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 963, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs964(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 964, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs965(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 965, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs966(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 966, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs967(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 967, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs968(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 968, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs969(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 969, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs970(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 970, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs971(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 971, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs972(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 972, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs973(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 973, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs974(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 974, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs975(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 975, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs976(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 976, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs977(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 977, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs978(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 978, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs979(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 979, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs980(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 980, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs981(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 981, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs982(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 982, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs983(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 983, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs984(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 984, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs985(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 985, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs986(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 986, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs987(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 987, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs988(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 988, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs989(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 989, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs990(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 990, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs991(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 991, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs992(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 992, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs993(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 993, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs994(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 994, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs995(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 995, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs996(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 996, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs997(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 997, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs998(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 998, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs999(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 999, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1000(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1000, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1001(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1001, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1002(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1002, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1003(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1003, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1004(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1004, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1005(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1005, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1006(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1006, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1007(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1007, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1008(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1008, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1009(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1009, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1010(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1010, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1011(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1011, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1012(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1012, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1013(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1013, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1014(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1014, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1015(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1015, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1016(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1016, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1017(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1017, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1018(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1018, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1019(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1019, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1020(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1020, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1021(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1021, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1022(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1022, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1023(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1023, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1024, float, float><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

