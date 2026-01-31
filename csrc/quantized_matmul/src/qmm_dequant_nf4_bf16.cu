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

void LaunchDequantNf4BF16(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_nf4_int<4, __nv_bfloat16><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantNf4BF16Gs8(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 8, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs9(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 9, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs10(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 10, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs11(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 11, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs12(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 12, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs13(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 13, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs14(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 14, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs15(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 15, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 16, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs17(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 17, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs18(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 18, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs19(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 19, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs20(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 20, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs21(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 21, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs22(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 22, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs23(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 23, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs24(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 24, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs25(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 25, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs26(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 26, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs27(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 27, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs28(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 28, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs29(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 29, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs30(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 30, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs31(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 31, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 32, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs33(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 33, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs34(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 34, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs35(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 35, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs36(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 36, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs37(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 37, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs38(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 38, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs39(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 39, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs40(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 40, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs41(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 41, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs42(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 42, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs43(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 43, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs44(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 44, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs45(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 45, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs46(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 46, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs47(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 47, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs48(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 48, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs49(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 49, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs50(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 50, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs51(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 51, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs52(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 52, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs53(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 53, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs54(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 54, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs55(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 55, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs56(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 56, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs57(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 57, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs58(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 58, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs59(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 59, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs60(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 60, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs61(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 61, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs62(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 62, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs63(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 63, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 64, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs65(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 65, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs66(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 66, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs67(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 67, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs68(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 68, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs69(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 69, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs70(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 70, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs71(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 71, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs72(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 72, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs73(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 73, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs74(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 74, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs75(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 75, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs76(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 76, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs77(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 77, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs78(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 78, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs79(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 79, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs80(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 80, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs81(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 81, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs82(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 82, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs83(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 83, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs84(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 84, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs85(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 85, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs86(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 86, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs87(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 87, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs88(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 88, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs89(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 89, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs90(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 90, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs91(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 91, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs92(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 92, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs93(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 93, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs94(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 94, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs95(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 95, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs96(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 96, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs97(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 97, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs98(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 98, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs99(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 99, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs100(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 100, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs101(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 101, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs102(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 102, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs103(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 103, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs104(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 104, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs105(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 105, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs106(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 106, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs107(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 107, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs108(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 108, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs109(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 109, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs110(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 110, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs111(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 111, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs112(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 112, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs113(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 113, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs114(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 114, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs115(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 115, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs116(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 116, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs117(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 117, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs118(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 118, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs119(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 119, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs120(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 120, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs121(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 121, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs122(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 122, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs123(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 123, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs124(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 124, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs125(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 125, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs126(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 126, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs127(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 127, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 128, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs129(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 129, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs130(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 130, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs131(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 131, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs132(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 132, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs133(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 133, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs134(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 134, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs135(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 135, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs136(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 136, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs137(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 137, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs138(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 138, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs139(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 139, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs140(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 140, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs141(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 141, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs142(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 142, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs143(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 143, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs144(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 144, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs145(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 145, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs146(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 146, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs147(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 147, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs148(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 148, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs149(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 149, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs150(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 150, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs151(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 151, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs152(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 152, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs153(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 153, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs154(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 154, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs155(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 155, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs156(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 156, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs157(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 157, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs158(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 158, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs159(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 159, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs160(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 160, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs161(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 161, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs162(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 162, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs163(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 163, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs164(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 164, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs165(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 165, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs166(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 166, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs167(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 167, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs168(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 168, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs169(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 169, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs170(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 170, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs171(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 171, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs172(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 172, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs173(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 173, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs174(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 174, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs175(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 175, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs176(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 176, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs177(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 177, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs178(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 178, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs179(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 179, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs180(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 180, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs181(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 181, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs182(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 182, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs183(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 183, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs184(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 184, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs185(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 185, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs186(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 186, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs187(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 187, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs188(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 188, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs189(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 189, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs190(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 190, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs191(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 191, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs192(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 192, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs193(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 193, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs194(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 194, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs195(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 195, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs196(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 196, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs197(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 197, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs198(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 198, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs199(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 199, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs200(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 200, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs201(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 201, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs202(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 202, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs203(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 203, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs204(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 204, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs205(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 205, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs206(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 206, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs207(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 207, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs208(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 208, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs209(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 209, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs210(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 210, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs211(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 211, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs212(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 212, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs213(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 213, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs214(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 214, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs215(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 215, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs216(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 216, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs217(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 217, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs218(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 218, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs219(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 219, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs220(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 220, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs221(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 221, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs222(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 222, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs223(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 223, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs224(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 224, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs225(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 225, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs226(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 226, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs227(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 227, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs228(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 228, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs229(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 229, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs230(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 230, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs231(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 231, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs232(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 232, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs233(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 233, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs234(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 234, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs235(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 235, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs236(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 236, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs237(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 237, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs238(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 238, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs239(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 239, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs240(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 240, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs241(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 241, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs242(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 242, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs243(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 243, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs244(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 244, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs245(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 245, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs246(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 246, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs247(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 247, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs248(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 248, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs249(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 249, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs250(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 250, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs251(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 251, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs252(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 252, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs253(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 253, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs254(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 254, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs255(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 255, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 256, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs257(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 257, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs258(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 258, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs259(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 259, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs260(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 260, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs261(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 261, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs262(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 262, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs263(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 263, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs264(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 264, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs265(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 265, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs266(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 266, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs267(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 267, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs268(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 268, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs269(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 269, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs270(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 270, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs271(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 271, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs272(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 272, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs273(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 273, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs274(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 274, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs275(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 275, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs276(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 276, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs277(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 277, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs278(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 278, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs279(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 279, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs280(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 280, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs281(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 281, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs282(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 282, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs283(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 283, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs284(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 284, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs285(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 285, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs286(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 286, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs287(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 287, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs288(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 288, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs289(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 289, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs290(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 290, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs291(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 291, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs292(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 292, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs293(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 293, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs294(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 294, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs295(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 295, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs296(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 296, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs297(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 297, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs298(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 298, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs299(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 299, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs300(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 300, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs301(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 301, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs302(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 302, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs303(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 303, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs304(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 304, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs305(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 305, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs306(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 306, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs307(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 307, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs308(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 308, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs309(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 309, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs310(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 310, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs311(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 311, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs312(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 312, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs313(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 313, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs314(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 314, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs315(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 315, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs316(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 316, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs317(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 317, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs318(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 318, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs319(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 319, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs320(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 320, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs321(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 321, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs322(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 322, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs323(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 323, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs324(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 324, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs325(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 325, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs326(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 326, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs327(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 327, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs328(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 328, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs329(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 329, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs330(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 330, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs331(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 331, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs332(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 332, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs333(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 333, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs334(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 334, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs335(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 335, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs336(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 336, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs337(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 337, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs338(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 338, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs339(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 339, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs340(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 340, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs341(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 341, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs342(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 342, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs343(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 343, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs344(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 344, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs345(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 345, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs346(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 346, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs347(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 347, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs348(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 348, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs349(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 349, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs350(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 350, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs351(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 351, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs352(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 352, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs353(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 353, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs354(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 354, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs355(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 355, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs356(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 356, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs357(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 357, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs358(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 358, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs359(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 359, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs360(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 360, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs361(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 361, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs362(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 362, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs363(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 363, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs364(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 364, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs365(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 365, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs366(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 366, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs367(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 367, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs368(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 368, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs369(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 369, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs370(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 370, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs371(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 371, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs372(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 372, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs373(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 373, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs374(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 374, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs375(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 375, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs376(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 376, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs377(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 377, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs378(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 378, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs379(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 379, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs380(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 380, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs381(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 381, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs382(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 382, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs383(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 383, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs384(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 384, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs385(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 385, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs386(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 386, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs387(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 387, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs388(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 388, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs389(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 389, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs390(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 390, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs391(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 391, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs392(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 392, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs393(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 393, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs394(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 394, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs395(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 395, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs396(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 396, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs397(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 397, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs398(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 398, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs399(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 399, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs400(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 400, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs401(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 401, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs402(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 402, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs403(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 403, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs404(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 404, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs405(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 405, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs406(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 406, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs407(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 407, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs408(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 408, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs409(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 409, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs410(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 410, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs411(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 411, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs412(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 412, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs413(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 413, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs414(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 414, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs415(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 415, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs416(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 416, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs417(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 417, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs418(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 418, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs419(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 419, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs420(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 420, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs421(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 421, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs422(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 422, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs423(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 423, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs424(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 424, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs425(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 425, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs426(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 426, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs427(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 427, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs428(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 428, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs429(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 429, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs430(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 430, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs431(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 431, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs432(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 432, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs433(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 433, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs434(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 434, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs435(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 435, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs436(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 436, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs437(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 437, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs438(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 438, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs439(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 439, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs440(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 440, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs441(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 441, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs442(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 442, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs443(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 443, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs444(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 444, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs445(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 445, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs446(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 446, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs447(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 447, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs448(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 448, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs449(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 449, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs450(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 450, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs451(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 451, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs452(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 452, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs453(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 453, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs454(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 454, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs455(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 455, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs456(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 456, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs457(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 457, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs458(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 458, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs459(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 459, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs460(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 460, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs461(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 461, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs462(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 462, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs463(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 463, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs464(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 464, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs465(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 465, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs466(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 466, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs467(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 467, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs468(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 468, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs469(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 469, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs470(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 470, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs471(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 471, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs472(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 472, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs473(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 473, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs474(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 474, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs475(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 475, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs476(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 476, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs477(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 477, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs478(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 478, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs479(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 479, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs480(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 480, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs481(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 481, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs482(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 482, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs483(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 483, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs484(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 484, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs485(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 485, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs486(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 486, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs487(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 487, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs488(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 488, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs489(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 489, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs490(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 490, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs491(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 491, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs492(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 492, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs493(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 493, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs494(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 494, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs495(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 495, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs496(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 496, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs497(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 497, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs498(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 498, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs499(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 499, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs500(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 500, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs501(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 501, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs502(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 502, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs503(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 503, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs504(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 504, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs505(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 505, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs506(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 506, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs507(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 507, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs508(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 508, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs509(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 509, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs510(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 510, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs511(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 511, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 512, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs513(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 513, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs514(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 514, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs515(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 515, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs516(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 516, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs517(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 517, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs518(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 518, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs519(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 519, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs520(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 520, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs521(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 521, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs522(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 522, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs523(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 523, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs524(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 524, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs525(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 525, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs526(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 526, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs527(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 527, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs528(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 528, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs529(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 529, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs530(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 530, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs531(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 531, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs532(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 532, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs533(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 533, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs534(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 534, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs535(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 535, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs536(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 536, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs537(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 537, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs538(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 538, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs539(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 539, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs540(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 540, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs541(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 541, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs542(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 542, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs543(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 543, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs544(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 544, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs545(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 545, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs546(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 546, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs547(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 547, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs548(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 548, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs549(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 549, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs550(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 550, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs551(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 551, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs552(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 552, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs553(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 553, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs554(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 554, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs555(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 555, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs556(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 556, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs557(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 557, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs558(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 558, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs559(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 559, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs560(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 560, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs561(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 561, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs562(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 562, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs563(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 563, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs564(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 564, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs565(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 565, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs566(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 566, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs567(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 567, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs568(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 568, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs569(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 569, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs570(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 570, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs571(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 571, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs572(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 572, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs573(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 573, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs574(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 574, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs575(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 575, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs576(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 576, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs577(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 577, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs578(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 578, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs579(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 579, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs580(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 580, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs581(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 581, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs582(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 582, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs583(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 583, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs584(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 584, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs585(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 585, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs586(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 586, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs587(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 587, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs588(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 588, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs589(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 589, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs590(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 590, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs591(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 591, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs592(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 592, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs593(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 593, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs594(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 594, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs595(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 595, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs596(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 596, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs597(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 597, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs598(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 598, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs599(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 599, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs600(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 600, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs601(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 601, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs602(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 602, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs603(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 603, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs604(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 604, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs605(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 605, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs606(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 606, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs607(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 607, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs608(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 608, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs609(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 609, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs610(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 610, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs611(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 611, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs612(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 612, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs613(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 613, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs614(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 614, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs615(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 615, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs616(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 616, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs617(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 617, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs618(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 618, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs619(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 619, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs620(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 620, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs621(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 621, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs622(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 622, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs623(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 623, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs624(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 624, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs625(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 625, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs626(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 626, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs627(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 627, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs628(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 628, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs629(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 629, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs630(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 630, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs631(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 631, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs632(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 632, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs633(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 633, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs634(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 634, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs635(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 635, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs636(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 636, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs637(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 637, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs638(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 638, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs639(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 639, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs640(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 640, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs641(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 641, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs642(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 642, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs643(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 643, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs644(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 644, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs645(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 645, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs646(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 646, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs647(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 647, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs648(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 648, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs649(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 649, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs650(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 650, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs651(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 651, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs652(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 652, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs653(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 653, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs654(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 654, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs655(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 655, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs656(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 656, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs657(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 657, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs658(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 658, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs659(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 659, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs660(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 660, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs661(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 661, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs662(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 662, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs663(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 663, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs664(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 664, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs665(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 665, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs666(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 666, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs667(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 667, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs668(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 668, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs669(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 669, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs670(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 670, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs671(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 671, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs672(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 672, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs673(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 673, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs674(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 674, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs675(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 675, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs676(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 676, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs677(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 677, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs678(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 678, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs679(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 679, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs680(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 680, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs681(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 681, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs682(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 682, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs683(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 683, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs684(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 684, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs685(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 685, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs686(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 686, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs687(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 687, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs688(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 688, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs689(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 689, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs690(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 690, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs691(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 691, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs692(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 692, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs693(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 693, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs694(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 694, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs695(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 695, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs696(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 696, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs697(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 697, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs698(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 698, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs699(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 699, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs700(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 700, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs701(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 701, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs702(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 702, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs703(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 703, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs704(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 704, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs705(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 705, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs706(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 706, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs707(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 707, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs708(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 708, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs709(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 709, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs710(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 710, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs711(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 711, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs712(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 712, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs713(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 713, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs714(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 714, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs715(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 715, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs716(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 716, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs717(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 717, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs718(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 718, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs719(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 719, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs720(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 720, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs721(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 721, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs722(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 722, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs723(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 723, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs724(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 724, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs725(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 725, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs726(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 726, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs727(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 727, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs728(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 728, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs729(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 729, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs730(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 730, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs731(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 731, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs732(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 732, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs733(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 733, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs734(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 734, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs735(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 735, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs736(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 736, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs737(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 737, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs738(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 738, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs739(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 739, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs740(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 740, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs741(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 741, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs742(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 742, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs743(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 743, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs744(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 744, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs745(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 745, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs746(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 746, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs747(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 747, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs748(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 748, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs749(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 749, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs750(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 750, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs751(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 751, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs752(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 752, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs753(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 753, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs754(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 754, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs755(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 755, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs756(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 756, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs757(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 757, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs758(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 758, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs759(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 759, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs760(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 760, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs761(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 761, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs762(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 762, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs763(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 763, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs764(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 764, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs765(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 765, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs766(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 766, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs767(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 767, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs768(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 768, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs769(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 769, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs770(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 770, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs771(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 771, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs772(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 772, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs773(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 773, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs774(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 774, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs775(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 775, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs776(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 776, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs777(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 777, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs778(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 778, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs779(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 779, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs780(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 780, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs781(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 781, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs782(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 782, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs783(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 783, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs784(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 784, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs785(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 785, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs786(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 786, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs787(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 787, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs788(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 788, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs789(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 789, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs790(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 790, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs791(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 791, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs792(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 792, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs793(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 793, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs794(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 794, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs795(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 795, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs796(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 796, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs797(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 797, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs798(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 798, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs799(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 799, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs800(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 800, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs801(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 801, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs802(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 802, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs803(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 803, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs804(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 804, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs805(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 805, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs806(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 806, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs807(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 807, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs808(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 808, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs809(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 809, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs810(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 810, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs811(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 811, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs812(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 812, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs813(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 813, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs814(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 814, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs815(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 815, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs816(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 816, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs817(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 817, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs818(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 818, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs819(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 819, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs820(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 820, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs821(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 821, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs822(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 822, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs823(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 823, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs824(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 824, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs825(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 825, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs826(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 826, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs827(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 827, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs828(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 828, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs829(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 829, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs830(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 830, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs831(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 831, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs832(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 832, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs833(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 833, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs834(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 834, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs835(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 835, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs836(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 836, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs837(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 837, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs838(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 838, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs839(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 839, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs840(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 840, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs841(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 841, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs842(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 842, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs843(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 843, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs844(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 844, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs845(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 845, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs846(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 846, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs847(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 847, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs848(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 848, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs849(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 849, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs850(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 850, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs851(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 851, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs852(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 852, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs853(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 853, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs854(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 854, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs855(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 855, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs856(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 856, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs857(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 857, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs858(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 858, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs859(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 859, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs860(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 860, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs861(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 861, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs862(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 862, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs863(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 863, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs864(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 864, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs865(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 865, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs866(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 866, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs867(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 867, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs868(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 868, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs869(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 869, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs870(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 870, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs871(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 871, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs872(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 872, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs873(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 873, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs874(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 874, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs875(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 875, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs876(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 876, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs877(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 877, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs878(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 878, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs879(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 879, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs880(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 880, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs881(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 881, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs882(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 882, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs883(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 883, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs884(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 884, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs885(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 885, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs886(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 886, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs887(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 887, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs888(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 888, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs889(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 889, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs890(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 890, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs891(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 891, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs892(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 892, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs893(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 893, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs894(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 894, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs895(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 895, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs896(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 896, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs897(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 897, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs898(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 898, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs899(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 899, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs900(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 900, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs901(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 901, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs902(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 902, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs903(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 903, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs904(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 904, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs905(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 905, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs906(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 906, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs907(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 907, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs908(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 908, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs909(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 909, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs910(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 910, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs911(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 911, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs912(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 912, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs913(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 913, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs914(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 914, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs915(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 915, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs916(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 916, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs917(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 917, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs918(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 918, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs919(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 919, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs920(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 920, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs921(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 921, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs922(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 922, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs923(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 923, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs924(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 924, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs925(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 925, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs926(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 926, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs927(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 927, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs928(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 928, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs929(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 929, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs930(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 930, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs931(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 931, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs932(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 932, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs933(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 933, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs934(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 934, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs935(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 935, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs936(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 936, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs937(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 937, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs938(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 938, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs939(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 939, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs940(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 940, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs941(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 941, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs942(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 942, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs943(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 943, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs944(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 944, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs945(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 945, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs946(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 946, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs947(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 947, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs948(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 948, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs949(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 949, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs950(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 950, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs951(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 951, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs952(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 952, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs953(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 953, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs954(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 954, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs955(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 955, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs956(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 956, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs957(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 957, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs958(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 958, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs959(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 959, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs960(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 960, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs961(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 961, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs962(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 962, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs963(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 963, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs964(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 964, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs965(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 965, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs966(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 966, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs967(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 967, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs968(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 968, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs969(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 969, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs970(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 970, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs971(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 971, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs972(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 972, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs973(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 973, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs974(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 974, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs975(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 975, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs976(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 976, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs977(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 977, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs978(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 978, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs979(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 979, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs980(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 980, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs981(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 981, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs982(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 982, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs983(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 983, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs984(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 984, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs985(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 985, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs986(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 986, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs987(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 987, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs988(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 988, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs989(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 989, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs990(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 990, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs991(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 991, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs992(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 992, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs993(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 993, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs994(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 994, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs995(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 995, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs996(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 996, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs997(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 997, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs998(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 998, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs999(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 999, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1000(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1000, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1001(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1001, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1002(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1002, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1003(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1003, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1004(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1004, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1005(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1005, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1006(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1006, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1007(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1007, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1008(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1008, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1009(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1009, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1010(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1010, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1011(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1011, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1012(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1012, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1013(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1013, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1014(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1014, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1015(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1015, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1016(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1016, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1017(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1017, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1018(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1018, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1019(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1019, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1020(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1020, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1021(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1021, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1022(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1022, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1023(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1023, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1024, __nv_bfloat16><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}

