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

void LaunchDequantAffineBits8BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<8, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits8BF16Gs8(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 8, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs9(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 9, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs10(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 10, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs11(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 11, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs12(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 12, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs13(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 13, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs14(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 14, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs15(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 15, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 16, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs17(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 17, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs18(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 18, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs19(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 19, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs20(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 20, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs21(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 21, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs22(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 22, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs23(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 23, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs24(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 24, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs25(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 25, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs26(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 26, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs27(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 27, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs28(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 28, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs29(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 29, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs30(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 30, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs31(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 31, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 32, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs33(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 33, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs34(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 34, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs35(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 35, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs36(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 36, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs37(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 37, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs38(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 38, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs39(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 39, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs40(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 40, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs41(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 41, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs42(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 42, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs43(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 43, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs44(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 44, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs45(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 45, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs46(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 46, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs47(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 47, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs48(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 48, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs49(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 49, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs50(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 50, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs51(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 51, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs52(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 52, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs53(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 53, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs54(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 54, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs55(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 55, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs56(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 56, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs57(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 57, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs58(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 58, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs59(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 59, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs60(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 60, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs61(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 61, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs62(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 62, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs63(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 63, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 64, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs65(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 65, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs66(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 66, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs67(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 67, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs68(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 68, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs69(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 69, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs70(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 70, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs71(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 71, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs72(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 72, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs73(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 73, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs74(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 74, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs75(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 75, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs76(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 76, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs77(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 77, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs78(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 78, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs79(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 79, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs80(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 80, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs81(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 81, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs82(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 82, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs83(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 83, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs84(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 84, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs85(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 85, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs86(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 86, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs87(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 87, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs88(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 88, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs89(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 89, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs90(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 90, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs91(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 91, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs92(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 92, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs93(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 93, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs94(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 94, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs95(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 95, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs96(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 96, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs97(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 97, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs98(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 98, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs99(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 99, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs100(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 100, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs101(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 101, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs102(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 102, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs103(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 103, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs104(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 104, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs105(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 105, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs106(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 106, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs107(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 107, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs108(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 108, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs109(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 109, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs110(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 110, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs111(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 111, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs112(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 112, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs113(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 113, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs114(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 114, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs115(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 115, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs116(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 116, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs117(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 117, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs118(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 118, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs119(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 119, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs120(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 120, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs121(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 121, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs122(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 122, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs123(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 123, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs124(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 124, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs125(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 125, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs126(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 126, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs127(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 127, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 128, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs129(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 129, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs130(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 130, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs131(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 131, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs132(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 132, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs133(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 133, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs134(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 134, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs135(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 135, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs136(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 136, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs137(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 137, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs138(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 138, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs139(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 139, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs140(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 140, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs141(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 141, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs142(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 142, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs143(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 143, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs144(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 144, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs145(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 145, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs146(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 146, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs147(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 147, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs148(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 148, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs149(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 149, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs150(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 150, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs151(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 151, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs152(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 152, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs153(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 153, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs154(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 154, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs155(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 155, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs156(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 156, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs157(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 157, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs158(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 158, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs159(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 159, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs160(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 160, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs161(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 161, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs162(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 162, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs163(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 163, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs164(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 164, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs165(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 165, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs166(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 166, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs167(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 167, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs168(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 168, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs169(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 169, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs170(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 170, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs171(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 171, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs172(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 172, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs173(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 173, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs174(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 174, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs175(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 175, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs176(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 176, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs177(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 177, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs178(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 178, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs179(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 179, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs180(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 180, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs181(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 181, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs182(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 182, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs183(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 183, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs184(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 184, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs185(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 185, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs186(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 186, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs187(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 187, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs188(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 188, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs189(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 189, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs190(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 190, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs191(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 191, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs192(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 192, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs193(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 193, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs194(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 194, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs195(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 195, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs196(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 196, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs197(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 197, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs198(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 198, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs199(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 199, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs200(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 200, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs201(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 201, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs202(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 202, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs203(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 203, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs204(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 204, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs205(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 205, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs206(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 206, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs207(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 207, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs208(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 208, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs209(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 209, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs210(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 210, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs211(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 211, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs212(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 212, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs213(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 213, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs214(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 214, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs215(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 215, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs216(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 216, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs217(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 217, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs218(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 218, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs219(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 219, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs220(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 220, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs221(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 221, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs222(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 222, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs223(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 223, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs224(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 224, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs225(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 225, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs226(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 226, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs227(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 227, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs228(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 228, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs229(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 229, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs230(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 230, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs231(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 231, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs232(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 232, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs233(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 233, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs234(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 234, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs235(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 235, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs236(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 236, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs237(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 237, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs238(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 238, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs239(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 239, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs240(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 240, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs241(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 241, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs242(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 242, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs243(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 243, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs244(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 244, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs245(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 245, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs246(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 246, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs247(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 247, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs248(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 248, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs249(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 249, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs250(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 250, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs251(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 251, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs252(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 252, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs253(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 253, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs254(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 254, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs255(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 255, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 256, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs257(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 257, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs258(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 258, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs259(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 259, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs260(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 260, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs261(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 261, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs262(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 262, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs263(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 263, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs264(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 264, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs265(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 265, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs266(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 266, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs267(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 267, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs268(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 268, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs269(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 269, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs270(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 270, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs271(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 271, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs272(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 272, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs273(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 273, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs274(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 274, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs275(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 275, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs276(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 276, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs277(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 277, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs278(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 278, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs279(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 279, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs280(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 280, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs281(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 281, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs282(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 282, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs283(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 283, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs284(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 284, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs285(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 285, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs286(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 286, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs287(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 287, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs288(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 288, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs289(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 289, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs290(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 290, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs291(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 291, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs292(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 292, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs293(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 293, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs294(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 294, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs295(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 295, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs296(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 296, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs297(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 297, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs298(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 298, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs299(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 299, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs300(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 300, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs301(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 301, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs302(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 302, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs303(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 303, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs304(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 304, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs305(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 305, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs306(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 306, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs307(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 307, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs308(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 308, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs309(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 309, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs310(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 310, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs311(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 311, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs312(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 312, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs313(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 313, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs314(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 314, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs315(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 315, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs316(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 316, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs317(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 317, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs318(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 318, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs319(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 319, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs320(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 320, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs321(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 321, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs322(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 322, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs323(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 323, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs324(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 324, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs325(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 325, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs326(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 326, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs327(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 327, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs328(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 328, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs329(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 329, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs330(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 330, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs331(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 331, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs332(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 332, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs333(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 333, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs334(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 334, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs335(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 335, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs336(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 336, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs337(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 337, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs338(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 338, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs339(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 339, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs340(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 340, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs341(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 341, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs342(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 342, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs343(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 343, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs344(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 344, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs345(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 345, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs346(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 346, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs347(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 347, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs348(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 348, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs349(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 349, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs350(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 350, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs351(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 351, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs352(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 352, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs353(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 353, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs354(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 354, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs355(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 355, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs356(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 356, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs357(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 357, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs358(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 358, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs359(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 359, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs360(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 360, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs361(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 361, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs362(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 362, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs363(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 363, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs364(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 364, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs365(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 365, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs366(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 366, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs367(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 367, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs368(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 368, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs369(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 369, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs370(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 370, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs371(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 371, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs372(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 372, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs373(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 373, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs374(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 374, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs375(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 375, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs376(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 376, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs377(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 377, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs378(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 378, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs379(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 379, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs380(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 380, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs381(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 381, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs382(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 382, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs383(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 383, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs384(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 384, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs385(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 385, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs386(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 386, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs387(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 387, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs388(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 388, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs389(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 389, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs390(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 390, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs391(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 391, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs392(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 392, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs393(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 393, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs394(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 394, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs395(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 395, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs396(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 396, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs397(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 397, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs398(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 398, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs399(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 399, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs400(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 400, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs401(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 401, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs402(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 402, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs403(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 403, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs404(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 404, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs405(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 405, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs406(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 406, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs407(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 407, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs408(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 408, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs409(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 409, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs410(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 410, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs411(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 411, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs412(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 412, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs413(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 413, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs414(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 414, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs415(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 415, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs416(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 416, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs417(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 417, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs418(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 418, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs419(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 419, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs420(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 420, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs421(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 421, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs422(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 422, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs423(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 423, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs424(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 424, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs425(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 425, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs426(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 426, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs427(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 427, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs428(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 428, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs429(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 429, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs430(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 430, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs431(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 431, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs432(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 432, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs433(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 433, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs434(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 434, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs435(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 435, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs436(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 436, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs437(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 437, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs438(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 438, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs439(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 439, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs440(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 440, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs441(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 441, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs442(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 442, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs443(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 443, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs444(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 444, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs445(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 445, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs446(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 446, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs447(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 447, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs448(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 448, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs449(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 449, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs450(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 450, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs451(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 451, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs452(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 452, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs453(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 453, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs454(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 454, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs455(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 455, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs456(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 456, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs457(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 457, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs458(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 458, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs459(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 459, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs460(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 460, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs461(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 461, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs462(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 462, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs463(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 463, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs464(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 464, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs465(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 465, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs466(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 466, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs467(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 467, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs468(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 468, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs469(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 469, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs470(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 470, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs471(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 471, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs472(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 472, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs473(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 473, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs474(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 474, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs475(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 475, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs476(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 476, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs477(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 477, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs478(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 478, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs479(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 479, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs480(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 480, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs481(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 481, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs482(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 482, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs483(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 483, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs484(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 484, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs485(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 485, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs486(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 486, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs487(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 487, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs488(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 488, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs489(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 489, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs490(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 490, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs491(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 491, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs492(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 492, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs493(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 493, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs494(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 494, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs495(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 495, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs496(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 496, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs497(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 497, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs498(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 498, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs499(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 499, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs500(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 500, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs501(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 501, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs502(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 502, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs503(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 503, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs504(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 504, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs505(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 505, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs506(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 506, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs507(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 507, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs508(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 508, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs509(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 509, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs510(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 510, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs511(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 511, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 512, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs513(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 513, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs514(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 514, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs515(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 515, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs516(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 516, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs517(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 517, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs518(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 518, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs519(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 519, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs520(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 520, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs521(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 521, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs522(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 522, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs523(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 523, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs524(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 524, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs525(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 525, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs526(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 526, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs527(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 527, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs528(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 528, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs529(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 529, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs530(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 530, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs531(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 531, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs532(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 532, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs533(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 533, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs534(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 534, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs535(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 535, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs536(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 536, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs537(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 537, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs538(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 538, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs539(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 539, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs540(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 540, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs541(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 541, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs542(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 542, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs543(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 543, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs544(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 544, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs545(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 545, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs546(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 546, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs547(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 547, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs548(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 548, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs549(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 549, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs550(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 550, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs551(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 551, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs552(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 552, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs553(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 553, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs554(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 554, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs555(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 555, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs556(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 556, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs557(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 557, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs558(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 558, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs559(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 559, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs560(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 560, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs561(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 561, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs562(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 562, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs563(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 563, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs564(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 564, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs565(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 565, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs566(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 566, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs567(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 567, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs568(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 568, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs569(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 569, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs570(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 570, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs571(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 571, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs572(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 572, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs573(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 573, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs574(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 574, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs575(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 575, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs576(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 576, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs577(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 577, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs578(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 578, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs579(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 579, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs580(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 580, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs581(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 581, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs582(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 582, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs583(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 583, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs584(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 584, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs585(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 585, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs586(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 586, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs587(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 587, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs588(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 588, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs589(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 589, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs590(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 590, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs591(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 591, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs592(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 592, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs593(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 593, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs594(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 594, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs595(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 595, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs596(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 596, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs597(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 597, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs598(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 598, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs599(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 599, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs600(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 600, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs601(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 601, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs602(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 602, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs603(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 603, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs604(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 604, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs605(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 605, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs606(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 606, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs607(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 607, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs608(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 608, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs609(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 609, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs610(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 610, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs611(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 611, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs612(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 612, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs613(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 613, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs614(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 614, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs615(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 615, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs616(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 616, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs617(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 617, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs618(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 618, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs619(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 619, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs620(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 620, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs621(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 621, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs622(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 622, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs623(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 623, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs624(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 624, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs625(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 625, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs626(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 626, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs627(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 627, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs628(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 628, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs629(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 629, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs630(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 630, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs631(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 631, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs632(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 632, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs633(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 633, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs634(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 634, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs635(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 635, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs636(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 636, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs637(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 637, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs638(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 638, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs639(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 639, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs640(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 640, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs641(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 641, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs642(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 642, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs643(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 643, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs644(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 644, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs645(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 645, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs646(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 646, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs647(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 647, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs648(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 648, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs649(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 649, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs650(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 650, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs651(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 651, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs652(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 652, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs653(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 653, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs654(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 654, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs655(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 655, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs656(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 656, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs657(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 657, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs658(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 658, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs659(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 659, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs660(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 660, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs661(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 661, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs662(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 662, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs663(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 663, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs664(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 664, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs665(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 665, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs666(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 666, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs667(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 667, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs668(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 668, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs669(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 669, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs670(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 670, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs671(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 671, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs672(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 672, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs673(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 673, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs674(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 674, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs675(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 675, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs676(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 676, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs677(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 677, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs678(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 678, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs679(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 679, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs680(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 680, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs681(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 681, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs682(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 682, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs683(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 683, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs684(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 684, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs685(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 685, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs686(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 686, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs687(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 687, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs688(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 688, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs689(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 689, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs690(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 690, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs691(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 691, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs692(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 692, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs693(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 693, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs694(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 694, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs695(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 695, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs696(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 696, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs697(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 697, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs698(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 698, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs699(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 699, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs700(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 700, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs701(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 701, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs702(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 702, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs703(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 703, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs704(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 704, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs705(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 705, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs706(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 706, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs707(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 707, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs708(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 708, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs709(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 709, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs710(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 710, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs711(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 711, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs712(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 712, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs713(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 713, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs714(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 714, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs715(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 715, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs716(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 716, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs717(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 717, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs718(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 718, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs719(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 719, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs720(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 720, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs721(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 721, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs722(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 722, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs723(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 723, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs724(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 724, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs725(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 725, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs726(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 726, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs727(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 727, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs728(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 728, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs729(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 729, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs730(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 730, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs731(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 731, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs732(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 732, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs733(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 733, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs734(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 734, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs735(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 735, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs736(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 736, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs737(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 737, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs738(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 738, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs739(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 739, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs740(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 740, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs741(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 741, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs742(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 742, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs743(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 743, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs744(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 744, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs745(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 745, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs746(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 746, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs747(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 747, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs748(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 748, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs749(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 749, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs750(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 750, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs751(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 751, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs752(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 752, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs753(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 753, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs754(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 754, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs755(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 755, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs756(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 756, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs757(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 757, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs758(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 758, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs759(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 759, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs760(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 760, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs761(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 761, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs762(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 762, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs763(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 763, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs764(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 764, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs765(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 765, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs766(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 766, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs767(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 767, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs768(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 768, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs769(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 769, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs770(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 770, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs771(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 771, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs772(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 772, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs773(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 773, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs774(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 774, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs775(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 775, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs776(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 776, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs777(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 777, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs778(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 778, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs779(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 779, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs780(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 780, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs781(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 781, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs782(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 782, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs783(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 783, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs784(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 784, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs785(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 785, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs786(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 786, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs787(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 787, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs788(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 788, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs789(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 789, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs790(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 790, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs791(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 791, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs792(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 792, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs793(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 793, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs794(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 794, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs795(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 795, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs796(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 796, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs797(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 797, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs798(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 798, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs799(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 799, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs800(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 800, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs801(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 801, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs802(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 802, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs803(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 803, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs804(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 804, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs805(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 805, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs806(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 806, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs807(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 807, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs808(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 808, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs809(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 809, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs810(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 810, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs811(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 811, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs812(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 812, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs813(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 813, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs814(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 814, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs815(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 815, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs816(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 816, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs817(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 817, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs818(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 818, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs819(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 819, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs820(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 820, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs821(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 821, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs822(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 822, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs823(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 823, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs824(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 824, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs825(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 825, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs826(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 826, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs827(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 827, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs828(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 828, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs829(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 829, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs830(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 830, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs831(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 831, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs832(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 832, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs833(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 833, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs834(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 834, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs835(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 835, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs836(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 836, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs837(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 837, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs838(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 838, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs839(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 839, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs840(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 840, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs841(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 841, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs842(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 842, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs843(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 843, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs844(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 844, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs845(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 845, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs846(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 846, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs847(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 847, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs848(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 848, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs849(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 849, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs850(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 850, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs851(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 851, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs852(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 852, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs853(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 853, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs854(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 854, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs855(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 855, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs856(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 856, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs857(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 857, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs858(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 858, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs859(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 859, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs860(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 860, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs861(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 861, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs862(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 862, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs863(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 863, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs864(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 864, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs865(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 865, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs866(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 866, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs867(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 867, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs868(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 868, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs869(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 869, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs870(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 870, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs871(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 871, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs872(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 872, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs873(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 873, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs874(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 874, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs875(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 875, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs876(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 876, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs877(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 877, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs878(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 878, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs879(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 879, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs880(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 880, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs881(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 881, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs882(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 882, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs883(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 883, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs884(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 884, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs885(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 885, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs886(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 886, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs887(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 887, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs888(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 888, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs889(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 889, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs890(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 890, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs891(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 891, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs892(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 892, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs893(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 893, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs894(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 894, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs895(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 895, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs896(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 896, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs897(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 897, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs898(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 898, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs899(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 899, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs900(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 900, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs901(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 901, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs902(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 902, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs903(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 903, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs904(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 904, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs905(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 905, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs906(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 906, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs907(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 907, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs908(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 908, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs909(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 909, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs910(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 910, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs911(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 911, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs912(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 912, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs913(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 913, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs914(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 914, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs915(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 915, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs916(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 916, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs917(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 917, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs918(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 918, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs919(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 919, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs920(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 920, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs921(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 921, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs922(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 922, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs923(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 923, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs924(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 924, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs925(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 925, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs926(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 926, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs927(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 927, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs928(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 928, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs929(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 929, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs930(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 930, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs931(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 931, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs932(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 932, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs933(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 933, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs934(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 934, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs935(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 935, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs936(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 936, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs937(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 937, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs938(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 938, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs939(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 939, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs940(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 940, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs941(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 941, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs942(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 942, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs943(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 943, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs944(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 944, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs945(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 945, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs946(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 946, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs947(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 947, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs948(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 948, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs949(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 949, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs950(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 950, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs951(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 951, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs952(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 952, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs953(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 953, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs954(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 954, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs955(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 955, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs956(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 956, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs957(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 957, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs958(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 958, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs959(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 959, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs960(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 960, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs961(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 961, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs962(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 962, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs963(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 963, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs964(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 964, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs965(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 965, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs966(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 966, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs967(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 967, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs968(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 968, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs969(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 969, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs970(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 970, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs971(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 971, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs972(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 972, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs973(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 973, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs974(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 974, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs975(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 975, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs976(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 976, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs977(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 977, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs978(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 978, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs979(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 979, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs980(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 980, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs981(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 981, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs982(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 982, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs983(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 983, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs984(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 984, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs985(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 985, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs986(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 986, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs987(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 987, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs988(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 988, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs989(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 989, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs990(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 990, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs991(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 991, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs992(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 992, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs993(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 993, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs994(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 994, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs995(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 995, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs996(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 996, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs997(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 997, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs998(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 998, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs999(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 999, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1000(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1000, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1001(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1001, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1002(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1002, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1003(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1003, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1004(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1004, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1005(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1005, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1006(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1006, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1007(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1007, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1008(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1008, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1009(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1009, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1010(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1010, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1011(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1011, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1012(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1012, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1013(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1013, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1014(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1014, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1015(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1015, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1016(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1016, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1017(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1017, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1018(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1018, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1019(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1019, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1020(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1020, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1021(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1021, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1022(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1022, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1023(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1023, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits8BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<8, 1024, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

