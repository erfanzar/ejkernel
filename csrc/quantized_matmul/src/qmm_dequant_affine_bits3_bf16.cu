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

void LaunchDequantAffineBits3BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<3, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits3BF16Gs8(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 8, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs9(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 9, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs10(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 10, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs11(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 11, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs12(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 12, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs13(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 13, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs14(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 14, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs15(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 15, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 16, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs17(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 17, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs18(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 18, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs19(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 19, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs20(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 20, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs21(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 21, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs22(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 22, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs23(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 23, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs24(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 24, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs25(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 25, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs26(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 26, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs27(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 27, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs28(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 28, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs29(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 29, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs30(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 30, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs31(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 31, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 32, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs33(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 33, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs34(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 34, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs35(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 35, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs36(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 36, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs37(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 37, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs38(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 38, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs39(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 39, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs40(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 40, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs41(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 41, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs42(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 42, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs43(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 43, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs44(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 44, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs45(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 45, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs46(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 46, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs47(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 47, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs48(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 48, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs49(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 49, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs50(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 50, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs51(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 51, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs52(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 52, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs53(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 53, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs54(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 54, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs55(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 55, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs56(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 56, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs57(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 57, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs58(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 58, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs59(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 59, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs60(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 60, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs61(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 61, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs62(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 62, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs63(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 63, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 64, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs65(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 65, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs66(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 66, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs67(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 67, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs68(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 68, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs69(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 69, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs70(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 70, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs71(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 71, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs72(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 72, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs73(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 73, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs74(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 74, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs75(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 75, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs76(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 76, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs77(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 77, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs78(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 78, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs79(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 79, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs80(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 80, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs81(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 81, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs82(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 82, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs83(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 83, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs84(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 84, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs85(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 85, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs86(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 86, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs87(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 87, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs88(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 88, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs89(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 89, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs90(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 90, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs91(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 91, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs92(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 92, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs93(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 93, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs94(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 94, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs95(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 95, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs96(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 96, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs97(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 97, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs98(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 98, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs99(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 99, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs100(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 100, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs101(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 101, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs102(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 102, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs103(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 103, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs104(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 104, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs105(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 105, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs106(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 106, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs107(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 107, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs108(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 108, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs109(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 109, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs110(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 110, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs111(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 111, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs112(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 112, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs113(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 113, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs114(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 114, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs115(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 115, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs116(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 116, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs117(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 117, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs118(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 118, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs119(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 119, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs120(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 120, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs121(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 121, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs122(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 122, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs123(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 123, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs124(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 124, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs125(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 125, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs126(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 126, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs127(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 127, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 128, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs129(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 129, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs130(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 130, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs131(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 131, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs132(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 132, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs133(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 133, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs134(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 134, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs135(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 135, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs136(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 136, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs137(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 137, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs138(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 138, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs139(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 139, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs140(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 140, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs141(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 141, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs142(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 142, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs143(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 143, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs144(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 144, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs145(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 145, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs146(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 146, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs147(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 147, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs148(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 148, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs149(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 149, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs150(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 150, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs151(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 151, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs152(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 152, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs153(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 153, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs154(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 154, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs155(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 155, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs156(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 156, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs157(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 157, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs158(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 158, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs159(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 159, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs160(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 160, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs161(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 161, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs162(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 162, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs163(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 163, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs164(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 164, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs165(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 165, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs166(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 166, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs167(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 167, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs168(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 168, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs169(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 169, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs170(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 170, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs171(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 171, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs172(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 172, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs173(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 173, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs174(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 174, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs175(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 175, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs176(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 176, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs177(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 177, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs178(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 178, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs179(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 179, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs180(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 180, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs181(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 181, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs182(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 182, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs183(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 183, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs184(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 184, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs185(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 185, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs186(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 186, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs187(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 187, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs188(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 188, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs189(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 189, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs190(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 190, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs191(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 191, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs192(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 192, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs193(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 193, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs194(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 194, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs195(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 195, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs196(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 196, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs197(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 197, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs198(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 198, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs199(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 199, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs200(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 200, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs201(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 201, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs202(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 202, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs203(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 203, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs204(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 204, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs205(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 205, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs206(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 206, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs207(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 207, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs208(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 208, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs209(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 209, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs210(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 210, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs211(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 211, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs212(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 212, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs213(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 213, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs214(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 214, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs215(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 215, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs216(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 216, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs217(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 217, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs218(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 218, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs219(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 219, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs220(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 220, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs221(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 221, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs222(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 222, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs223(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 223, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs224(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 224, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs225(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 225, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs226(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 226, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs227(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 227, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs228(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 228, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs229(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 229, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs230(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 230, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs231(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 231, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs232(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 232, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs233(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 233, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs234(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 234, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs235(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 235, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs236(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 236, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs237(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 237, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs238(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 238, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs239(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 239, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs240(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 240, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs241(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 241, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs242(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 242, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs243(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 243, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs244(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 244, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs245(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 245, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs246(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 246, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs247(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 247, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs248(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 248, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs249(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 249, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs250(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 250, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs251(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 251, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs252(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 252, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs253(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 253, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs254(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 254, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs255(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 255, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 256, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs257(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 257, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs258(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 258, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs259(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 259, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs260(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 260, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs261(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 261, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs262(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 262, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs263(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 263, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs264(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 264, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs265(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 265, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs266(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 266, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs267(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 267, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs268(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 268, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs269(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 269, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs270(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 270, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs271(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 271, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs272(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 272, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs273(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 273, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs274(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 274, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs275(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 275, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs276(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 276, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs277(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 277, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs278(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 278, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs279(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 279, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs280(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 280, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs281(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 281, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs282(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 282, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs283(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 283, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs284(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 284, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs285(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 285, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs286(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 286, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs287(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 287, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs288(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 288, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs289(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 289, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs290(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 290, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs291(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 291, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs292(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 292, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs293(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 293, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs294(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 294, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs295(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 295, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs296(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 296, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs297(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 297, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs298(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 298, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs299(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 299, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs300(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 300, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs301(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 301, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs302(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 302, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs303(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 303, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs304(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 304, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs305(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 305, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs306(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 306, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs307(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 307, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs308(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 308, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs309(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 309, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs310(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 310, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs311(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 311, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs312(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 312, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs313(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 313, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs314(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 314, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs315(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 315, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs316(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 316, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs317(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 317, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs318(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 318, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs319(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 319, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs320(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 320, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs321(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 321, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs322(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 322, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs323(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 323, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs324(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 324, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs325(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 325, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs326(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 326, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs327(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 327, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs328(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 328, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs329(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 329, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs330(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 330, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs331(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 331, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs332(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 332, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs333(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 333, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs334(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 334, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs335(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 335, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs336(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 336, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs337(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 337, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs338(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 338, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs339(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 339, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs340(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 340, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs341(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 341, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs342(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 342, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs343(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 343, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs344(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 344, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs345(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 345, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs346(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 346, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs347(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 347, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs348(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 348, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs349(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 349, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs350(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 350, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs351(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 351, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs352(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 352, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs353(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 353, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs354(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 354, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs355(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 355, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs356(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 356, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs357(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 357, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs358(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 358, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs359(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 359, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs360(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 360, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs361(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 361, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs362(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 362, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs363(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 363, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs364(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 364, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs365(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 365, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs366(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 366, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs367(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 367, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs368(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 368, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs369(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 369, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs370(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 370, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs371(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 371, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs372(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 372, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs373(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 373, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs374(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 374, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs375(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 375, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs376(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 376, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs377(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 377, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs378(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 378, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs379(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 379, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs380(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 380, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs381(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 381, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs382(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 382, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs383(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 383, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs384(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 384, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs385(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 385, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs386(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 386, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs387(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 387, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs388(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 388, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs389(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 389, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs390(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 390, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs391(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 391, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs392(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 392, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs393(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 393, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs394(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 394, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs395(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 395, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs396(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 396, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs397(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 397, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs398(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 398, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs399(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 399, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs400(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 400, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs401(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 401, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs402(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 402, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs403(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 403, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs404(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 404, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs405(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 405, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs406(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 406, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs407(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 407, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs408(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 408, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs409(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 409, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs410(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 410, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs411(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 411, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs412(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 412, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs413(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 413, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs414(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 414, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs415(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 415, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs416(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 416, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs417(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 417, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs418(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 418, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs419(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 419, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs420(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 420, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs421(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 421, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs422(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 422, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs423(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 423, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs424(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 424, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs425(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 425, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs426(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 426, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs427(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 427, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs428(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 428, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs429(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 429, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs430(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 430, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs431(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 431, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs432(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 432, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs433(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 433, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs434(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 434, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs435(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 435, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs436(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 436, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs437(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 437, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs438(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 438, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs439(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 439, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs440(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 440, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs441(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 441, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs442(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 442, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs443(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 443, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs444(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 444, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs445(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 445, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs446(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 446, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs447(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 447, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs448(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 448, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs449(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 449, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs450(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 450, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs451(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 451, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs452(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 452, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs453(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 453, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs454(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 454, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs455(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 455, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs456(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 456, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs457(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 457, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs458(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 458, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs459(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 459, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs460(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 460, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs461(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 461, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs462(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 462, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs463(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 463, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs464(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 464, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs465(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 465, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs466(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 466, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs467(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 467, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs468(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 468, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs469(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 469, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs470(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 470, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs471(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 471, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs472(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 472, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs473(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 473, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs474(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 474, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs475(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 475, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs476(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 476, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs477(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 477, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs478(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 478, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs479(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 479, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs480(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 480, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs481(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 481, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs482(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 482, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs483(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 483, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs484(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 484, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs485(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 485, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs486(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 486, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs487(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 487, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs488(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 488, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs489(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 489, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs490(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 490, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs491(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 491, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs492(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 492, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs493(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 493, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs494(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 494, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs495(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 495, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs496(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 496, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs497(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 497, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs498(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 498, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs499(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 499, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs500(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 500, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs501(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 501, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs502(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 502, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs503(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 503, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs504(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 504, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs505(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 505, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs506(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 506, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs507(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 507, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs508(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 508, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs509(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 509, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs510(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 510, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs511(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 511, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 512, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs513(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 513, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs514(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 514, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs515(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 515, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs516(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 516, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs517(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 517, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs518(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 518, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs519(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 519, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs520(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 520, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs521(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 521, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs522(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 522, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs523(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 523, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs524(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 524, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs525(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 525, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs526(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 526, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs527(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 527, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs528(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 528, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs529(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 529, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs530(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 530, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs531(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 531, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs532(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 532, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs533(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 533, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs534(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 534, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs535(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 535, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs536(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 536, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs537(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 537, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs538(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 538, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs539(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 539, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs540(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 540, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs541(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 541, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs542(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 542, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs543(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 543, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs544(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 544, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs545(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 545, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs546(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 546, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs547(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 547, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs548(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 548, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs549(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 549, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs550(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 550, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs551(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 551, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs552(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 552, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs553(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 553, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs554(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 554, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs555(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 555, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs556(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 556, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs557(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 557, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs558(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 558, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs559(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 559, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs560(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 560, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs561(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 561, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs562(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 562, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs563(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 563, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs564(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 564, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs565(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 565, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs566(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 566, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs567(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 567, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs568(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 568, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs569(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 569, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs570(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 570, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs571(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 571, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs572(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 572, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs573(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 573, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs574(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 574, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs575(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 575, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs576(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 576, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs577(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 577, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs578(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 578, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs579(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 579, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs580(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 580, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs581(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 581, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs582(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 582, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs583(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 583, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs584(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 584, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs585(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 585, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs586(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 586, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs587(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 587, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs588(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 588, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs589(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 589, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs590(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 590, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs591(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 591, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs592(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 592, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs593(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 593, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs594(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 594, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs595(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 595, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs596(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 596, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs597(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 597, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs598(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 598, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs599(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 599, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs600(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 600, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs601(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 601, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs602(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 602, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs603(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 603, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs604(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 604, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs605(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 605, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs606(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 606, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs607(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 607, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs608(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 608, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs609(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 609, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs610(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 610, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs611(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 611, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs612(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 612, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs613(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 613, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs614(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 614, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs615(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 615, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs616(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 616, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs617(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 617, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs618(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 618, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs619(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 619, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs620(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 620, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs621(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 621, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs622(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 622, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs623(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 623, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs624(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 624, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs625(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 625, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs626(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 626, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs627(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 627, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs628(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 628, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs629(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 629, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs630(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 630, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs631(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 631, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs632(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 632, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs633(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 633, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs634(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 634, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs635(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 635, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs636(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 636, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs637(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 637, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs638(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 638, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs639(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 639, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs640(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 640, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs641(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 641, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs642(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 642, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs643(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 643, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs644(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 644, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs645(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 645, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs646(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 646, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs647(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 647, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs648(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 648, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs649(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 649, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs650(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 650, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs651(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 651, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs652(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 652, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs653(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 653, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs654(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 654, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs655(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 655, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs656(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 656, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs657(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 657, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs658(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 658, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs659(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 659, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs660(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 660, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs661(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 661, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs662(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 662, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs663(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 663, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs664(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 664, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs665(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 665, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs666(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 666, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs667(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 667, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs668(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 668, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs669(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 669, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs670(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 670, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs671(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 671, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs672(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 672, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs673(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 673, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs674(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 674, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs675(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 675, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs676(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 676, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs677(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 677, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs678(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 678, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs679(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 679, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs680(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 680, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs681(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 681, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs682(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 682, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs683(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 683, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs684(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 684, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs685(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 685, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs686(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 686, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs687(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 687, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs688(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 688, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs689(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 689, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs690(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 690, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs691(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 691, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs692(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 692, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs693(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 693, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs694(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 694, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs695(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 695, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs696(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 696, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs697(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 697, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs698(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 698, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs699(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 699, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs700(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 700, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs701(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 701, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs702(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 702, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs703(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 703, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs704(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 704, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs705(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 705, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs706(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 706, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs707(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 707, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs708(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 708, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs709(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 709, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs710(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 710, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs711(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 711, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs712(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 712, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs713(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 713, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs714(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 714, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs715(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 715, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs716(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 716, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs717(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 717, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs718(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 718, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs719(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 719, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs720(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 720, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs721(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 721, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs722(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 722, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs723(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 723, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs724(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 724, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs725(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 725, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs726(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 726, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs727(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 727, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs728(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 728, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs729(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 729, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs730(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 730, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs731(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 731, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs732(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 732, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs733(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 733, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs734(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 734, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs735(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 735, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs736(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 736, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs737(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 737, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs738(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 738, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs739(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 739, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs740(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 740, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs741(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 741, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs742(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 742, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs743(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 743, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs744(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 744, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs745(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 745, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs746(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 746, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs747(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 747, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs748(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 748, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs749(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 749, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs750(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 750, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs751(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 751, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs752(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 752, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs753(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 753, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs754(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 754, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs755(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 755, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs756(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 756, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs757(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 757, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs758(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 758, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs759(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 759, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs760(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 760, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs761(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 761, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs762(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 762, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs763(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 763, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs764(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 764, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs765(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 765, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs766(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 766, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs767(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 767, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs768(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 768, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs769(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 769, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs770(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 770, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs771(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 771, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs772(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 772, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs773(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 773, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs774(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 774, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs775(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 775, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs776(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 776, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs777(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 777, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs778(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 778, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs779(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 779, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs780(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 780, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs781(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 781, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs782(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 782, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs783(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 783, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs784(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 784, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs785(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 785, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs786(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 786, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs787(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 787, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs788(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 788, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs789(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 789, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs790(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 790, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs791(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 791, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs792(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 792, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs793(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 793, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs794(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 794, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs795(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 795, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs796(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 796, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs797(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 797, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs798(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 798, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs799(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 799, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs800(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 800, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs801(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 801, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs802(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 802, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs803(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 803, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs804(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 804, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs805(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 805, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs806(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 806, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs807(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 807, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs808(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 808, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs809(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 809, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs810(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 810, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs811(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 811, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs812(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 812, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs813(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 813, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs814(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 814, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs815(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 815, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs816(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 816, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs817(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 817, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs818(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 818, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs819(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 819, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs820(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 820, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs821(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 821, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs822(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 822, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs823(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 823, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs824(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 824, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs825(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 825, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs826(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 826, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs827(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 827, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs828(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 828, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs829(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 829, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs830(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 830, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs831(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 831, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs832(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 832, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs833(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 833, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs834(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 834, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs835(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 835, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs836(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 836, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs837(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 837, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs838(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 838, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs839(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 839, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs840(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 840, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs841(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 841, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs842(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 842, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs843(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 843, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs844(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 844, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs845(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 845, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs846(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 846, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs847(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 847, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs848(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 848, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs849(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 849, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs850(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 850, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs851(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 851, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs852(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 852, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs853(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 853, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs854(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 854, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs855(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 855, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs856(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 856, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs857(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 857, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs858(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 858, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs859(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 859, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs860(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 860, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs861(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 861, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs862(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 862, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs863(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 863, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs864(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 864, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs865(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 865, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs866(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 866, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs867(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 867, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs868(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 868, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs869(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 869, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs870(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 870, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs871(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 871, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs872(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 872, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs873(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 873, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs874(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 874, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs875(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 875, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs876(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 876, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs877(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 877, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs878(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 878, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs879(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 879, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs880(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 880, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs881(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 881, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs882(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 882, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs883(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 883, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs884(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 884, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs885(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 885, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs886(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 886, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs887(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 887, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs888(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 888, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs889(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 889, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs890(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 890, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs891(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 891, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs892(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 892, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs893(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 893, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs894(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 894, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs895(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 895, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs896(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 896, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs897(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 897, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs898(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 898, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs899(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 899, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs900(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 900, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs901(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 901, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs902(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 902, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs903(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 903, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs904(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 904, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs905(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 905, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs906(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 906, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs907(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 907, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs908(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 908, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs909(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 909, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs910(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 910, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs911(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 911, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs912(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 912, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs913(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 913, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs914(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 914, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs915(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 915, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs916(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 916, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs917(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 917, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs918(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 918, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs919(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 919, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs920(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 920, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs921(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 921, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs922(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 922, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs923(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 923, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs924(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 924, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs925(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 925, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs926(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 926, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs927(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 927, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs928(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 928, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs929(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 929, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs930(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 930, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs931(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 931, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs932(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 932, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs933(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 933, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs934(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 934, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs935(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 935, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs936(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 936, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs937(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 937, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs938(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 938, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs939(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 939, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs940(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 940, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs941(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 941, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs942(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 942, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs943(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 943, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs944(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 944, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs945(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 945, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs946(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 946, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs947(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 947, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs948(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 948, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs949(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 949, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs950(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 950, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs951(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 951, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs952(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 952, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs953(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 953, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs954(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 954, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs955(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 955, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs956(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 956, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs957(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 957, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs958(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 958, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs959(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 959, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs960(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 960, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs961(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 961, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs962(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 962, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs963(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 963, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs964(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 964, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs965(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 965, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs966(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 966, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs967(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 967, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs968(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 968, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs969(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 969, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs970(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 970, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs971(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 971, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs972(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 972, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs973(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 973, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs974(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 974, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs975(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 975, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs976(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 976, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs977(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 977, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs978(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 978, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs979(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 979, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs980(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 980, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs981(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 981, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs982(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 982, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs983(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 983, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs984(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 984, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs985(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 985, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs986(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 986, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs987(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 987, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs988(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 988, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs989(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 989, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs990(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 990, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs991(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 991, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs992(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 992, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs993(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 993, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs994(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 994, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs995(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 995, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs996(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 996, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs997(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 997, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs998(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 998, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs999(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 999, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1000(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1000, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1001(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1001, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1002(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1002, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1003(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1003, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1004(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1004, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1005(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1005, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1006(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1006, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1007(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1007, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1008(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1008, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1009(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1009, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1010(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1010, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1011(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1011, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1012(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1012, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1013(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1013, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1014(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1014, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1015(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1015, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1016(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1016, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1017(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1017, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1018(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1018, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1019(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1019, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1020(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1020, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1021(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1021, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1022(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1022, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1023(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1023, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits3BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<3, 1024, __nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

