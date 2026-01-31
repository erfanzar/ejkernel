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

void LaunchDequantAffineBits7BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<7, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits7BF16Gs8(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 8, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs9(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 9, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs10(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 10, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs11(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 11, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs12(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 12, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs13(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 13, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs14(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 14, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs15(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 15, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 16, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs17(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 17, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs18(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 18, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs19(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 19, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs20(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 20, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs21(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 21, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs22(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 22, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs23(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 23, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs24(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 24, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs25(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 25, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs26(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 26, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs27(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 27, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs28(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 28, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs29(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 29, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs30(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 30, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs31(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 31, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 32, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs33(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 33, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs34(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 34, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs35(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 35, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs36(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 36, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs37(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 37, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs38(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 38, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs39(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 39, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs40(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 40, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs41(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 41, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs42(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 42, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs43(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 43, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs44(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 44, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs45(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 45, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs46(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 46, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs47(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 47, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs48(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 48, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs49(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 49, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs50(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 50, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs51(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 51, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs52(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 52, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs53(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 53, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs54(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 54, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs55(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 55, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs56(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 56, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs57(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 57, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs58(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 58, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs59(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 59, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs60(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 60, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs61(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 61, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs62(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 62, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs63(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 63, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 64, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs65(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 65, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs66(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 66, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs67(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 67, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs68(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 68, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs69(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 69, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs70(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 70, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs71(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 71, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs72(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 72, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs73(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 73, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs74(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 74, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs75(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 75, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs76(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 76, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs77(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 77, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs78(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 78, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs79(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 79, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs80(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 80, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs81(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 81, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs82(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 82, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs83(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 83, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs84(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 84, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs85(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 85, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs86(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 86, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs87(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 87, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs88(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 88, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs89(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 89, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs90(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 90, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs91(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 91, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs92(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 92, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs93(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 93, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs94(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 94, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs95(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 95, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs96(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 96, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs97(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 97, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs98(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 98, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs99(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 99, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs100(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 100, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs101(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 101, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs102(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 102, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs103(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 103, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs104(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 104, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs105(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 105, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs106(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 106, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs107(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 107, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs108(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 108, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs109(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 109, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs110(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 110, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs111(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 111, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs112(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 112, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs113(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 113, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs114(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 114, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs115(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 115, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs116(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 116, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs117(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 117, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs118(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 118, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs119(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 119, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs120(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 120, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs121(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 121, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs122(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 122, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs123(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 123, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs124(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 124, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs125(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 125, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs126(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 126, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs127(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 127, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 128, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs129(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 129, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs130(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 130, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs131(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 131, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs132(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 132, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs133(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 133, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs134(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 134, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs135(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 135, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs136(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 136, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs137(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 137, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs138(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 138, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs139(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 139, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs140(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 140, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs141(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 141, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs142(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 142, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs143(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 143, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs144(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 144, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs145(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 145, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs146(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 146, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs147(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 147, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs148(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 148, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs149(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 149, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs150(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 150, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs151(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 151, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs152(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 152, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs153(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 153, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs154(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 154, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs155(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 155, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs156(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 156, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs157(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 157, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs158(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 158, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs159(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 159, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs160(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 160, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs161(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 161, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs162(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 162, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs163(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 163, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs164(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 164, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs165(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 165, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs166(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 166, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs167(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 167, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs168(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 168, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs169(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 169, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs170(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 170, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs171(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 171, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs172(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 172, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs173(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 173, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs174(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 174, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs175(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 175, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs176(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 176, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs177(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 177, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs178(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 178, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs179(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 179, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs180(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 180, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs181(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 181, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs182(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 182, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs183(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 183, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs184(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 184, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs185(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 185, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs186(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 186, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs187(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 187, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs188(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 188, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs189(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 189, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs190(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 190, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs191(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 191, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs192(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 192, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs193(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 193, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs194(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 194, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs195(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 195, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs196(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 196, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs197(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 197, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs198(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 198, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs199(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 199, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs200(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 200, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs201(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 201, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs202(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 202, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs203(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 203, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs204(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 204, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs205(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 205, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs206(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 206, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs207(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 207, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs208(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 208, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs209(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 209, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs210(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 210, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs211(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 211, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs212(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 212, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs213(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 213, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs214(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 214, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs215(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 215, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs216(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 216, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs217(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 217, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs218(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 218, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs219(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 219, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs220(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 220, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs221(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 221, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs222(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 222, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs223(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 223, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs224(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 224, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs225(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 225, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs226(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 226, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs227(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 227, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs228(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 228, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs229(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 229, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs230(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 230, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs231(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 231, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs232(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 232, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs233(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 233, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs234(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 234, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs235(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 235, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs236(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 236, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs237(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 237, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs238(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 238, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs239(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 239, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs240(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 240, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs241(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 241, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs242(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 242, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs243(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 243, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs244(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 244, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs245(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 245, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs246(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 246, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs247(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 247, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs248(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 248, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs249(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 249, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs250(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 250, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs251(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 251, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs252(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 252, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs253(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 253, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs254(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 254, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs255(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 255, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 256, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs257(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 257, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs258(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 258, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs259(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 259, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs260(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 260, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs261(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 261, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs262(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 262, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs263(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 263, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs264(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 264, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs265(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 265, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs266(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 266, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs267(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 267, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs268(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 268, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs269(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 269, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs270(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 270, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs271(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 271, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs272(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 272, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs273(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 273, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs274(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 274, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs275(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 275, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs276(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 276, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs277(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 277, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs278(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 278, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs279(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 279, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs280(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 280, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs281(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 281, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs282(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 282, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs283(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 283, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs284(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 284, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs285(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 285, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs286(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 286, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs287(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 287, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs288(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 288, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs289(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 289, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs290(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 290, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs291(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 291, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs292(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 292, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs293(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 293, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs294(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 294, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs295(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 295, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs296(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 296, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs297(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 297, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs298(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 298, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs299(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 299, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs300(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 300, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs301(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 301, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs302(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 302, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs303(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 303, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs304(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 304, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs305(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 305, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs306(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 306, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs307(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 307, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs308(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 308, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs309(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 309, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs310(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 310, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs311(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 311, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs312(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 312, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs313(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 313, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs314(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 314, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs315(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 315, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs316(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 316, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs317(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 317, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs318(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 318, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs319(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 319, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs320(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 320, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs321(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 321, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs322(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 322, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs323(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 323, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs324(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 324, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs325(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 325, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs326(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 326, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs327(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 327, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs328(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 328, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs329(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 329, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs330(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 330, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs331(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 331, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs332(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 332, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs333(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 333, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs334(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 334, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs335(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 335, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs336(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 336, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs337(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 337, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs338(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 338, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs339(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 339, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs340(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 340, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs341(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 341, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs342(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 342, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs343(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 343, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs344(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 344, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs345(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 345, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs346(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 346, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs347(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 347, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs348(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 348, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs349(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 349, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs350(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 350, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs351(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 351, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs352(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 352, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs353(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 353, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs354(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 354, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs355(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 355, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs356(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 356, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs357(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 357, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs358(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 358, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs359(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 359, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs360(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 360, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs361(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 361, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs362(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 362, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs363(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 363, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs364(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 364, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs365(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 365, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs366(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 366, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs367(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 367, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs368(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 368, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs369(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 369, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs370(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 370, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs371(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 371, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs372(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 372, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs373(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 373, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs374(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 374, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs375(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 375, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs376(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 376, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs377(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 377, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs378(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 378, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs379(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 379, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs380(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 380, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs381(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 381, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs382(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 382, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs383(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 383, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs384(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 384, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs385(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 385, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs386(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 386, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs387(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 387, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs388(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 388, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs389(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 389, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs390(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 390, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs391(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 391, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs392(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 392, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs393(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 393, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs394(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 394, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs395(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 395, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs396(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 396, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs397(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 397, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs398(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 398, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs399(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 399, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs400(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 400, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs401(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 401, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs402(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 402, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs403(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 403, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs404(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 404, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs405(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 405, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs406(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 406, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs407(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 407, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs408(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 408, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs409(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 409, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs410(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 410, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs411(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 411, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs412(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 412, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs413(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 413, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs414(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 414, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs415(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 415, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs416(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 416, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs417(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 417, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs418(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 418, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs419(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 419, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs420(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 420, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs421(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 421, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs422(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 422, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs423(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 423, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs424(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 424, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs425(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 425, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs426(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 426, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs427(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 427, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs428(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 428, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs429(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 429, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs430(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 430, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs431(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 431, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs432(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 432, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs433(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 433, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs434(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 434, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs435(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 435, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs436(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 436, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs437(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 437, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs438(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 438, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs439(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 439, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs440(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 440, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs441(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 441, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs442(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 442, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs443(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 443, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs444(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 444, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs445(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 445, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs446(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 446, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs447(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 447, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs448(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 448, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs449(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 449, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs450(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 450, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs451(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 451, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs452(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 452, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs453(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 453, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs454(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 454, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs455(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 455, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs456(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 456, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs457(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 457, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs458(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 458, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs459(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 459, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs460(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 460, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs461(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 461, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs462(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 462, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs463(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 463, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs464(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 464, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs465(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 465, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs466(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 466, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs467(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 467, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs468(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 468, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs469(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 469, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs470(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 470, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs471(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 471, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs472(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 472, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs473(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 473, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs474(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 474, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs475(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 475, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs476(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 476, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs477(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 477, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs478(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 478, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs479(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 479, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs480(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 480, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs481(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 481, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs482(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 482, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs483(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 483, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs484(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 484, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs485(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 485, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs486(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 486, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs487(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 487, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs488(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 488, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs489(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 489, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs490(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 490, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs491(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 491, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs492(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 492, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs493(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 493, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs494(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 494, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs495(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 495, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs496(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 496, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs497(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 497, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs498(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 498, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs499(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 499, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs500(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 500, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs501(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 501, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs502(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 502, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs503(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 503, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs504(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 504, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs505(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 505, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs506(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 506, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs507(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 507, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs508(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 508, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs509(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 509, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs510(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 510, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs511(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 511, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 512, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs513(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 513, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs514(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 514, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs515(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 515, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs516(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 516, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs517(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 517, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs518(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 518, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs519(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 519, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs520(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 520, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs521(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 521, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs522(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 522, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs523(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 523, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs524(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 524, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs525(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 525, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs526(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 526, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs527(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 527, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs528(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 528, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs529(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 529, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs530(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 530, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs531(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 531, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs532(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 532, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs533(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 533, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs534(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 534, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs535(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 535, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs536(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 536, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs537(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 537, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs538(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 538, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs539(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 539, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs540(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 540, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs541(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 541, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs542(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 542, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs543(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 543, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs544(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 544, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs545(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 545, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs546(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 546, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs547(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 547, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs548(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 548, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs549(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 549, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs550(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 550, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs551(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 551, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs552(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 552, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs553(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 553, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs554(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 554, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs555(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 555, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs556(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 556, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs557(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 557, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs558(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 558, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs559(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 559, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs560(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 560, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs561(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 561, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs562(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 562, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs563(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 563, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs564(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 564, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs565(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 565, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs566(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 566, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs567(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 567, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs568(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 568, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs569(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 569, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs570(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 570, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs571(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 571, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs572(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 572, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs573(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 573, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs574(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 574, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs575(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 575, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs576(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 576, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs577(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 577, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs578(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 578, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs579(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 579, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs580(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 580, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs581(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 581, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs582(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 582, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs583(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 583, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs584(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 584, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs585(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 585, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs586(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 586, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs587(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 587, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs588(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 588, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs589(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 589, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs590(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 590, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs591(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 591, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs592(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 592, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs593(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 593, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs594(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 594, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs595(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 595, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs596(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 596, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs597(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 597, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs598(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 598, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs599(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 599, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs600(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 600, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs601(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 601, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs602(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 602, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs603(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 603, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs604(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 604, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs605(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 605, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs606(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 606, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs607(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 607, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs608(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 608, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs609(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 609, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs610(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 610, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs611(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 611, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs612(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 612, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs613(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 613, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs614(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 614, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs615(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 615, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs616(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 616, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs617(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 617, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs618(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 618, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs619(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 619, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs620(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 620, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs621(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 621, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs622(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 622, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs623(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 623, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs624(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 624, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs625(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 625, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs626(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 626, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs627(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 627, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs628(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 628, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs629(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 629, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs630(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 630, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs631(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 631, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs632(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 632, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs633(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 633, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs634(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 634, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs635(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 635, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs636(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 636, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs637(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 637, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs638(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 638, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs639(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 639, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs640(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 640, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs641(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 641, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs642(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 642, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs643(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 643, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs644(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 644, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs645(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 645, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs646(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 646, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs647(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 647, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs648(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 648, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs649(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 649, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs650(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 650, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs651(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 651, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs652(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 652, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs653(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 653, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs654(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 654, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs655(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 655, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs656(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 656, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs657(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 657, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs658(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 658, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs659(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 659, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs660(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 660, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs661(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 661, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs662(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 662, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs663(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 663, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs664(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 664, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs665(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 665, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs666(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 666, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs667(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 667, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs668(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 668, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs669(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 669, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs670(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 670, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs671(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 671, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs672(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 672, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs673(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 673, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs674(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 674, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs675(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 675, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs676(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 676, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs677(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 677, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs678(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 678, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs679(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 679, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs680(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 680, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs681(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 681, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs682(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 682, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs683(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 683, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs684(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 684, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs685(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 685, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs686(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 686, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs687(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 687, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs688(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 688, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs689(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 689, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs690(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 690, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs691(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 691, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs692(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 692, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs693(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 693, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs694(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 694, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs695(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 695, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs696(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 696, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs697(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 697, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs698(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 698, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs699(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 699, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs700(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 700, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs701(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 701, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs702(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 702, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs703(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 703, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs704(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 704, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs705(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 705, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs706(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 706, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs707(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 707, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs708(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 708, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs709(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 709, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs710(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 710, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs711(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 711, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs712(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 712, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs713(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 713, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs714(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 714, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs715(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 715, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs716(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 716, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs717(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 717, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs718(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 718, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs719(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 719, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs720(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 720, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs721(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 721, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs722(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 722, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs723(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 723, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs724(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 724, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs725(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 725, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs726(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 726, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs727(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 727, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs728(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 728, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs729(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 729, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs730(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 730, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs731(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 731, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs732(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 732, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs733(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 733, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs734(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 734, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs735(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 735, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs736(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 736, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs737(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 737, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs738(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 738, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs739(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 739, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs740(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 740, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs741(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 741, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs742(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 742, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs743(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 743, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs744(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 744, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs745(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 745, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs746(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 746, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs747(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 747, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs748(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 748, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs749(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 749, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs750(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 750, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs751(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 751, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs752(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 752, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs753(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 753, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs754(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 754, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs755(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 755, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs756(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 756, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs757(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 757, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs758(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 758, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs759(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 759, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs760(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 760, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs761(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 761, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs762(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 762, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs763(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 763, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs764(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 764, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs765(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 765, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs766(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 766, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs767(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 767, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs768(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 768, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs769(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 769, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs770(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 770, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs771(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 771, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs772(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 772, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs773(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 773, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs774(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 774, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs775(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 775, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs776(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 776, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs777(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 777, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs778(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 778, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs779(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 779, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs780(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 780, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs781(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 781, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs782(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 782, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs783(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 783, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs784(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 784, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs785(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 785, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs786(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 786, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs787(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 787, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs788(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 788, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs789(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 789, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs790(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 790, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs791(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 791, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs792(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 792, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs793(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 793, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs794(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 794, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs795(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 795, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs796(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 796, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs797(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 797, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs798(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 798, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs799(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 799, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs800(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 800, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs801(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 801, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs802(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 802, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs803(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 803, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs804(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 804, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs805(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 805, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs806(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 806, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs807(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 807, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs808(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 808, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs809(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 809, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs810(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 810, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs811(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 811, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs812(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 812, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs813(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 813, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs814(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 814, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs815(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 815, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs816(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 816, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs817(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 817, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs818(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 818, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs819(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 819, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs820(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 820, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs821(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 821, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs822(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 822, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs823(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 823, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs824(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 824, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs825(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 825, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs826(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 826, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs827(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 827, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs828(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 828, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs829(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 829, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs830(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 830, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs831(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 831, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs832(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 832, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs833(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 833, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs834(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 834, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs835(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 835, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs836(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 836, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs837(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 837, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs838(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 838, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs839(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 839, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs840(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 840, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs841(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 841, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs842(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 842, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs843(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 843, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs844(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 844, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs845(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 845, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs846(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 846, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs847(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 847, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs848(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 848, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs849(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 849, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs850(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 850, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs851(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 851, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs852(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 852, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs853(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 853, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs854(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 854, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs855(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 855, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs856(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 856, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs857(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 857, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs858(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 858, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs859(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 859, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs860(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 860, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs861(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 861, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs862(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 862, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs863(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 863, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs864(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 864, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs865(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 865, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs866(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 866, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs867(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 867, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs868(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 868, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs869(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 869, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs870(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 870, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs871(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 871, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs872(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 872, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs873(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 873, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs874(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 874, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs875(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 875, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs876(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 876, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs877(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 877, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs878(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 878, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs879(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 879, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs880(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 880, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs881(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 881, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs882(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 882, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs883(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 883, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs884(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 884, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs885(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 885, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs886(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 886, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs887(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 887, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs888(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 888, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs889(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 889, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs890(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 890, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs891(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 891, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs892(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 892, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs893(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 893, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs894(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 894, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs895(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 895, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs896(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 896, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs897(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 897, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs898(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 898, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs899(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 899, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs900(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 900, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs901(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 901, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs902(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 902, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs903(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 903, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs904(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 904, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs905(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 905, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs906(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 906, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs907(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 907, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs908(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 908, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs909(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 909, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs910(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 910, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs911(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 911, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs912(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 912, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs913(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 913, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs914(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 914, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs915(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 915, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs916(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 916, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs917(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 917, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs918(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 918, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs919(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 919, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs920(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 920, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs921(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 921, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs922(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 922, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs923(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 923, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs924(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 924, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs925(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 925, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs926(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 926, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs927(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 927, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs928(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 928, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs929(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 929, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs930(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 930, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs931(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 931, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs932(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 932, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs933(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 933, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs934(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 934, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs935(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 935, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs936(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 936, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs937(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 937, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs938(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 938, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs939(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 939, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs940(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 940, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs941(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 941, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs942(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 942, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs943(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 943, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs944(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 944, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs945(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 945, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs946(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 946, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs947(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 947, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs948(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 948, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs949(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 949, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs950(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 950, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs951(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 951, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs952(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 952, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs953(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 953, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs954(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 954, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs955(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 955, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs956(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 956, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs957(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 957, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs958(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 958, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs959(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 959, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs960(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 960, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs961(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 961, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs962(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 962, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs963(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 963, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs964(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 964, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs965(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 965, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs966(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 966, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs967(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 967, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs968(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 968, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs969(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 969, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs970(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 970, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs971(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 971, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs972(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 972, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs973(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 973, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs974(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 974, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs975(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 975, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs976(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 976, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs977(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 977, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs978(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 978, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs979(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 979, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs980(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 980, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs981(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 981, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs982(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 982, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs983(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 983, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs984(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 984, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs985(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 985, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs986(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 986, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs987(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 987, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs988(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 988, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs989(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 989, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs990(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 990, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs991(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 991, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs992(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 992, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs993(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 993, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs994(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 994, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs995(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 995, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs996(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 996, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs997(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 997, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs998(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 998, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs999(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 999, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1000(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1000, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1001(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1001, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1002(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1002, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1003(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1003, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1004(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1004, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1005(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1005, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1006(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1006, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1007(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1007, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1008(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1008, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1009(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1009, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1010(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1010, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1011(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1011, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1012(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1012, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1013(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1013, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1014(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1014, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1015(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1015, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1016(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1016, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1017(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1017, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1018(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1018, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1019(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1019, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1020(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1020, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1021(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1021, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1022(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1022, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1023(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1023, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1024, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

