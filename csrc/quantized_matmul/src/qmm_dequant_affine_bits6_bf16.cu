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

void LaunchDequantAffineBits6BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<6, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits6BF16Gs8(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 8, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs9(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 9, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs10(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 10, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs11(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 11, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs12(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 12, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs13(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 13, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs14(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 14, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs15(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 15, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 16, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs17(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 17, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs18(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 18, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs19(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 19, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs20(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 20, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs21(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 21, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs22(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 22, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs23(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 23, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs24(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 24, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs25(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 25, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs26(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 26, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs27(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 27, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs28(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 28, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs29(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 29, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs30(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 30, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs31(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 31, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 32, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs33(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 33, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs34(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 34, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs35(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 35, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs36(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 36, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs37(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 37, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs38(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 38, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs39(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 39, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs40(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 40, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs41(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 41, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs42(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 42, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs43(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 43, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs44(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 44, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs45(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 45, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs46(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 46, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs47(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 47, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs48(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 48, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs49(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 49, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs50(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 50, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs51(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 51, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs52(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 52, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs53(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 53, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs54(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 54, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs55(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 55, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs56(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 56, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs57(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 57, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs58(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 58, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs59(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 59, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs60(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 60, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs61(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 61, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs62(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 62, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs63(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 63, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 64, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs65(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 65, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs66(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 66, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs67(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 67, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs68(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 68, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs69(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 69, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs70(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 70, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs71(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 71, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs72(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 72, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs73(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 73, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs74(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 74, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs75(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 75, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs76(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 76, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs77(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 77, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs78(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 78, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs79(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 79, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs80(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 80, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs81(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 81, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs82(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 82, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs83(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 83, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs84(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 84, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs85(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 85, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs86(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 86, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs87(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 87, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs88(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 88, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs89(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 89, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs90(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 90, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs91(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 91, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs92(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 92, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs93(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 93, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs94(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 94, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs95(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 95, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs96(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 96, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs97(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 97, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs98(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 98, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs99(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 99, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs100(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 100, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs101(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 101, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs102(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 102, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs103(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 103, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs104(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 104, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs105(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 105, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs106(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 106, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs107(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 107, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs108(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 108, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs109(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 109, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs110(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 110, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs111(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 111, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs112(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 112, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs113(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 113, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs114(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 114, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs115(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 115, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs116(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 116, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs117(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 117, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs118(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 118, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs119(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 119, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs120(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 120, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs121(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 121, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs122(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 122, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs123(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 123, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs124(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 124, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs125(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 125, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs126(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 126, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs127(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 127, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 128, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs129(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 129, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs130(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 130, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs131(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 131, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs132(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 132, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs133(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 133, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs134(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 134, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs135(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 135, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs136(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 136, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs137(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 137, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs138(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 138, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs139(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 139, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs140(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 140, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs141(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 141, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs142(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 142, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs143(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 143, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs144(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 144, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs145(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 145, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs146(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 146, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs147(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 147, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs148(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 148, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs149(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 149, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs150(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 150, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs151(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 151, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs152(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 152, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs153(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 153, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs154(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 154, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs155(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 155, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs156(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 156, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs157(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 157, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs158(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 158, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs159(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 159, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs160(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 160, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs161(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 161, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs162(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 162, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs163(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 163, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs164(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 164, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs165(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 165, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs166(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 166, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs167(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 167, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs168(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 168, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs169(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 169, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs170(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 170, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs171(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 171, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs172(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 172, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs173(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 173, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs174(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 174, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs175(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 175, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs176(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 176, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs177(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 177, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs178(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 178, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs179(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 179, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs180(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 180, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs181(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 181, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs182(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 182, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs183(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 183, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs184(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 184, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs185(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 185, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs186(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 186, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs187(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 187, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs188(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 188, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs189(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 189, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs190(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 190, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs191(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 191, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs192(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 192, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs193(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 193, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs194(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 194, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs195(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 195, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs196(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 196, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs197(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 197, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs198(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 198, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs199(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 199, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs200(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 200, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs201(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 201, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs202(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 202, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs203(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 203, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs204(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 204, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs205(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 205, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs206(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 206, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs207(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 207, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs208(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 208, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs209(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 209, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs210(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 210, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs211(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 211, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs212(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 212, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs213(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 213, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs214(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 214, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs215(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 215, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs216(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 216, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs217(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 217, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs218(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 218, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs219(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 219, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs220(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 220, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs221(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 221, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs222(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 222, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs223(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 223, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs224(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 224, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs225(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 225, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs226(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 226, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs227(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 227, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs228(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 228, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs229(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 229, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs230(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 230, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs231(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 231, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs232(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 232, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs233(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 233, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs234(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 234, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs235(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 235, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs236(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 236, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs237(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 237, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs238(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 238, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs239(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 239, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs240(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 240, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs241(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 241, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs242(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 242, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs243(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 243, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs244(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 244, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs245(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 245, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs246(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 246, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs247(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 247, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs248(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 248, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs249(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 249, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs250(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 250, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs251(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 251, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs252(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 252, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs253(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 253, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs254(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 254, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs255(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 255, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 256, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs257(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 257, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs258(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 258, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs259(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 259, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs260(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 260, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs261(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 261, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs262(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 262, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs263(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 263, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs264(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 264, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs265(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 265, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs266(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 266, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs267(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 267, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs268(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 268, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs269(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 269, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs270(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 270, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs271(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 271, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs272(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 272, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs273(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 273, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs274(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 274, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs275(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 275, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs276(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 276, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs277(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 277, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs278(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 278, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs279(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 279, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs280(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 280, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs281(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 281, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs282(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 282, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs283(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 283, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs284(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 284, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs285(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 285, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs286(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 286, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs287(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 287, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs288(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 288, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs289(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 289, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs290(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 290, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs291(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 291, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs292(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 292, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs293(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 293, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs294(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 294, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs295(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 295, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs296(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 296, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs297(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 297, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs298(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 298, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs299(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 299, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs300(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 300, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs301(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 301, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs302(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 302, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs303(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 303, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs304(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 304, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs305(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 305, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs306(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 306, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs307(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 307, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs308(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 308, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs309(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 309, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs310(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 310, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs311(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 311, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs312(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 312, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs313(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 313, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs314(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 314, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs315(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 315, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs316(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 316, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs317(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 317, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs318(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 318, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs319(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 319, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs320(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 320, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs321(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 321, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs322(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 322, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs323(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 323, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs324(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 324, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs325(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 325, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs326(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 326, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs327(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 327, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs328(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 328, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs329(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 329, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs330(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 330, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs331(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 331, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs332(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 332, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs333(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 333, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs334(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 334, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs335(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 335, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs336(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 336, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs337(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 337, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs338(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 338, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs339(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 339, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs340(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 340, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs341(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 341, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs342(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 342, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs343(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 343, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs344(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 344, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs345(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 345, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs346(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 346, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs347(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 347, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs348(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 348, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs349(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 349, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs350(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 350, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs351(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 351, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs352(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 352, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs353(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 353, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs354(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 354, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs355(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 355, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs356(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 356, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs357(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 357, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs358(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 358, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs359(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 359, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs360(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 360, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs361(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 361, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs362(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 362, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs363(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 363, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs364(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 364, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs365(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 365, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs366(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 366, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs367(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 367, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs368(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 368, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs369(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 369, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs370(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 370, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs371(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 371, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs372(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 372, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs373(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 373, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs374(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 374, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs375(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 375, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs376(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 376, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs377(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 377, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs378(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 378, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs379(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 379, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs380(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 380, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs381(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 381, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs382(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 382, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs383(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 383, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs384(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 384, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs385(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 385, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs386(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 386, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs387(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 387, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs388(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 388, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs389(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 389, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs390(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 390, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs391(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 391, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs392(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 392, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs393(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 393, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs394(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 394, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs395(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 395, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs396(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 396, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs397(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 397, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs398(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 398, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs399(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 399, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs400(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 400, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs401(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 401, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs402(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 402, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs403(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 403, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs404(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 404, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs405(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 405, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs406(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 406, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs407(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 407, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs408(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 408, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs409(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 409, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs410(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 410, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs411(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 411, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs412(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 412, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs413(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 413, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs414(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 414, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs415(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 415, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs416(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 416, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs417(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 417, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs418(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 418, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs419(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 419, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs420(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 420, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs421(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 421, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs422(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 422, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs423(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 423, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs424(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 424, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs425(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 425, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs426(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 426, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs427(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 427, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs428(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 428, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs429(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 429, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs430(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 430, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs431(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 431, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs432(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 432, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs433(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 433, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs434(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 434, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs435(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 435, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs436(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 436, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs437(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 437, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs438(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 438, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs439(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 439, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs440(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 440, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs441(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 441, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs442(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 442, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs443(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 443, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs444(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 444, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs445(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 445, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs446(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 446, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs447(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 447, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs448(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 448, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs449(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 449, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs450(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 450, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs451(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 451, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs452(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 452, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs453(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 453, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs454(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 454, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs455(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 455, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs456(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 456, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs457(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 457, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs458(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 458, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs459(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 459, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs460(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 460, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs461(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 461, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs462(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 462, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs463(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 463, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs464(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 464, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs465(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 465, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs466(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 466, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs467(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 467, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs468(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 468, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs469(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 469, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs470(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 470, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs471(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 471, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs472(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 472, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs473(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 473, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs474(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 474, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs475(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 475, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs476(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 476, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs477(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 477, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs478(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 478, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs479(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 479, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs480(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 480, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs481(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 481, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs482(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 482, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs483(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 483, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs484(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 484, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs485(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 485, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs486(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 486, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs487(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 487, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs488(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 488, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs489(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 489, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs490(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 490, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs491(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 491, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs492(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 492, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs493(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 493, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs494(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 494, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs495(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 495, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs496(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 496, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs497(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 497, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs498(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 498, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs499(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 499, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs500(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 500, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs501(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 501, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs502(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 502, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs503(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 503, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs504(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 504, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs505(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 505, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs506(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 506, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs507(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 507, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs508(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 508, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs509(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 509, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs510(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 510, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs511(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 511, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 512, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs513(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 513, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs514(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 514, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs515(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 515, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs516(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 516, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs517(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 517, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs518(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 518, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs519(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 519, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs520(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 520, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs521(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 521, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs522(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 522, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs523(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 523, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs524(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 524, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs525(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 525, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs526(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 526, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs527(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 527, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs528(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 528, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs529(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 529, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs530(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 530, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs531(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 531, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs532(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 532, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs533(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 533, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs534(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 534, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs535(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 535, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs536(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 536, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs537(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 537, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs538(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 538, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs539(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 539, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs540(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 540, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs541(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 541, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs542(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 542, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs543(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 543, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs544(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 544, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs545(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 545, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs546(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 546, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs547(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 547, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs548(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 548, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs549(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 549, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs550(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 550, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs551(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 551, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs552(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 552, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs553(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 553, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs554(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 554, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs555(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 555, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs556(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 556, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs557(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 557, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs558(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 558, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs559(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 559, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs560(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 560, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs561(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 561, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs562(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 562, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs563(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 563, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs564(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 564, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs565(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 565, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs566(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 566, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs567(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 567, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs568(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 568, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs569(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 569, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs570(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 570, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs571(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 571, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs572(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 572, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs573(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 573, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs574(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 574, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs575(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 575, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs576(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 576, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs577(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 577, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs578(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 578, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs579(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 579, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs580(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 580, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs581(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 581, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs582(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 582, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs583(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 583, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs584(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 584, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs585(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 585, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs586(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 586, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs587(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 587, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs588(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 588, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs589(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 589, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs590(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 590, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs591(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 591, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs592(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 592, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs593(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 593, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs594(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 594, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs595(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 595, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs596(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 596, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs597(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 597, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs598(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 598, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs599(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 599, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs600(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 600, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs601(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 601, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs602(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 602, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs603(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 603, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs604(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 604, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs605(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 605, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs606(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 606, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs607(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 607, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs608(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 608, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs609(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 609, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs610(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 610, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs611(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 611, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs612(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 612, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs613(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 613, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs614(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 614, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs615(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 615, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs616(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 616, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs617(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 617, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs618(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 618, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs619(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 619, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs620(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 620, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs621(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 621, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs622(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 622, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs623(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 623, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs624(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 624, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs625(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 625, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs626(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 626, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs627(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 627, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs628(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 628, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs629(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 629, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs630(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 630, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs631(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 631, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs632(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 632, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs633(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 633, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs634(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 634, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs635(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 635, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs636(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 636, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs637(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 637, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs638(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 638, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs639(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 639, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs640(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 640, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs641(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 641, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs642(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 642, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs643(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 643, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs644(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 644, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs645(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 645, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs646(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 646, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs647(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 647, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs648(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 648, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs649(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 649, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs650(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 650, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs651(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 651, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs652(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 652, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs653(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 653, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs654(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 654, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs655(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 655, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs656(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 656, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs657(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 657, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs658(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 658, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs659(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 659, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs660(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 660, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs661(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 661, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs662(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 662, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs663(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 663, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs664(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 664, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs665(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 665, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs666(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 666, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs667(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 667, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs668(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 668, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs669(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 669, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs670(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 670, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs671(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 671, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs672(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 672, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs673(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 673, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs674(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 674, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs675(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 675, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs676(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 676, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs677(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 677, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs678(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 678, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs679(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 679, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs680(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 680, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs681(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 681, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs682(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 682, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs683(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 683, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs684(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 684, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs685(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 685, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs686(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 686, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs687(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 687, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs688(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 688, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs689(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 689, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs690(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 690, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs691(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 691, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs692(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 692, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs693(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 693, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs694(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 694, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs695(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 695, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs696(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 696, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs697(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 697, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs698(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 698, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs699(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 699, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs700(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 700, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs701(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 701, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs702(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 702, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs703(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 703, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs704(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 704, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs705(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 705, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs706(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 706, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs707(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 707, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs708(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 708, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs709(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 709, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs710(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 710, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs711(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 711, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs712(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 712, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs713(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 713, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs714(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 714, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs715(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 715, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs716(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 716, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs717(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 717, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs718(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 718, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs719(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 719, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs720(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 720, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs721(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 721, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs722(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 722, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs723(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 723, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs724(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 724, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs725(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 725, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs726(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 726, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs727(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 727, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs728(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 728, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs729(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 729, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs730(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 730, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs731(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 731, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs732(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 732, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs733(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 733, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs734(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 734, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs735(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 735, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs736(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 736, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs737(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 737, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs738(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 738, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs739(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 739, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs740(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 740, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs741(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 741, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs742(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 742, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs743(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 743, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs744(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 744, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs745(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 745, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs746(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 746, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs747(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 747, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs748(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 748, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs749(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 749, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs750(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 750, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs751(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 751, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs752(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 752, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs753(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 753, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs754(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 754, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs755(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 755, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs756(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 756, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs757(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 757, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs758(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 758, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs759(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 759, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs760(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 760, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs761(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 761, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs762(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 762, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs763(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 763, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs764(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 764, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs765(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 765, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs766(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 766, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs767(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 767, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs768(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 768, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs769(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 769, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs770(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 770, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs771(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 771, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs772(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 772, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs773(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 773, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs774(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 774, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs775(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 775, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs776(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 776, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs777(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 777, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs778(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 778, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs779(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 779, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs780(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 780, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs781(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 781, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs782(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 782, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs783(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 783, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs784(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 784, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs785(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 785, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs786(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 786, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs787(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 787, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs788(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 788, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs789(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 789, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs790(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 790, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs791(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 791, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs792(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 792, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs793(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 793, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs794(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 794, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs795(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 795, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs796(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 796, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs797(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 797, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs798(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 798, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs799(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 799, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs800(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 800, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs801(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 801, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs802(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 802, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs803(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 803, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs804(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 804, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs805(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 805, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs806(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 806, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs807(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 807, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs808(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 808, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs809(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 809, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs810(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 810, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs811(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 811, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs812(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 812, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs813(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 813, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs814(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 814, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs815(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 815, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs816(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 816, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs817(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 817, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs818(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 818, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs819(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 819, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs820(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 820, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs821(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 821, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs822(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 822, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs823(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 823, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs824(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 824, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs825(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 825, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs826(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 826, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs827(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 827, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs828(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 828, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs829(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 829, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs830(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 830, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs831(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 831, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs832(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 832, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs833(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 833, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs834(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 834, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs835(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 835, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs836(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 836, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs837(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 837, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs838(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 838, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs839(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 839, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs840(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 840, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs841(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 841, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs842(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 842, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs843(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 843, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs844(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 844, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs845(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 845, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs846(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 846, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs847(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 847, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs848(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 848, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs849(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 849, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs850(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 850, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs851(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 851, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs852(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 852, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs853(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 853, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs854(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 854, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs855(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 855, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs856(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 856, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs857(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 857, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs858(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 858, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs859(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 859, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs860(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 860, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs861(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 861, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs862(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 862, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs863(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 863, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs864(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 864, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs865(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 865, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs866(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 866, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs867(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 867, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs868(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 868, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs869(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 869, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs870(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 870, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs871(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 871, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs872(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 872, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs873(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 873, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs874(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 874, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs875(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 875, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs876(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 876, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs877(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 877, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs878(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 878, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs879(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 879, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs880(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 880, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs881(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 881, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs882(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 882, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs883(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 883, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs884(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 884, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs885(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 885, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs886(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 886, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs887(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 887, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs888(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 888, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs889(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 889, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs890(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 890, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs891(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 891, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs892(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 892, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs893(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 893, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs894(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 894, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs895(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 895, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs896(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 896, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs897(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 897, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs898(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 898, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs899(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 899, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs900(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 900, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs901(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 901, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs902(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 902, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs903(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 903, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs904(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 904, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs905(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 905, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs906(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 906, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs907(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 907, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs908(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 908, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs909(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 909, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs910(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 910, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs911(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 911, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs912(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 912, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs913(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 913, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs914(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 914, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs915(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 915, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs916(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 916, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs917(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 917, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs918(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 918, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs919(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 919, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs920(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 920, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs921(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 921, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs922(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 922, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs923(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 923, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs924(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 924, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs925(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 925, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs926(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 926, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs927(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 927, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs928(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 928, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs929(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 929, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs930(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 930, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs931(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 931, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs932(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 932, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs933(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 933, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs934(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 934, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs935(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 935, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs936(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 936, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs937(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 937, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs938(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 938, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs939(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 939, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs940(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 940, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs941(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 941, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs942(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 942, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs943(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 943, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs944(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 944, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs945(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 945, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs946(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 946, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs947(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 947, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs948(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 948, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs949(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 949, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs950(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 950, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs951(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 951, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs952(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 952, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs953(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 953, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs954(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 954, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs955(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 955, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs956(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 956, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs957(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 957, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs958(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 958, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs959(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 959, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs960(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 960, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs961(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 961, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs962(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 962, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs963(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 963, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs964(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 964, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs965(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 965, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs966(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 966, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs967(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 967, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs968(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 968, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs969(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 969, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs970(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 970, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs971(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 971, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs972(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 972, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs973(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 973, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs974(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 974, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs975(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 975, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs976(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 976, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs977(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 977, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs978(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 978, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs979(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 979, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs980(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 980, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs981(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 981, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs982(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 982, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs983(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 983, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs984(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 984, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs985(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 985, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs986(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 986, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs987(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 987, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs988(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 988, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs989(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 989, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs990(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 990, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs991(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 991, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs992(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 992, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs993(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 993, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs994(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 994, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs995(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 995, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs996(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 996, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs997(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 997, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs998(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 998, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs999(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 999, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1000(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1000, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1001(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1001, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1002(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1002, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1003(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1003, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1004(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1004, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1005(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1005, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1006(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1006, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1007(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1007, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1008(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1008, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1009(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1009, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1010(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1010, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1011(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1011, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1012(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1012, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1013(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1013, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1014(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1014, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1015(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1015, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1016(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1016, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1017(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1017, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1018(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1018, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1019(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1019, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1020(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1020, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1021(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1021, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1022(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1022, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1023(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1023, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits6BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<6, 1024, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

