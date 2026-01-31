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

void LaunchDequantNf4F32(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_nf4_int<4, float><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantNf4F32Gs8(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 8, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs9(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 9, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs10(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 10, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs11(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 11, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs12(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 12, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs13(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 13, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs14(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 14, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs15(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 15, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs16(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 16, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs17(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 17, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs18(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 18, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs19(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 19, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs20(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 20, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs21(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 21, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs22(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 22, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs23(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 23, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs24(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 24, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs25(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 25, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs26(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 26, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs27(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 27, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs28(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 28, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs29(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 29, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs30(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 30, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs31(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 31, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs32(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 32, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs33(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 33, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs34(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 34, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs35(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 35, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs36(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 36, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs37(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 37, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs38(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 38, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs39(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 39, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs40(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 40, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs41(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 41, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs42(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 42, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs43(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 43, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs44(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 44, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs45(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 45, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs46(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 46, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs47(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 47, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs48(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 48, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs49(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 49, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs50(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 50, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs51(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 51, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs52(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 52, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs53(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 53, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs54(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 54, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs55(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 55, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs56(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 56, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs57(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 57, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs58(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 58, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs59(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 59, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs60(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 60, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs61(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 61, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs62(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 62, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs63(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 63, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs64(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 64, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs65(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 65, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs66(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 66, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs67(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 67, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs68(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 68, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs69(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 69, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs70(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 70, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs71(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 71, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs72(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 72, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs73(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 73, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs74(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 74, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs75(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 75, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs76(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 76, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs77(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 77, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs78(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 78, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs79(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 79, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs80(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 80, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs81(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 81, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs82(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 82, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs83(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 83, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs84(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 84, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs85(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 85, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs86(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 86, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs87(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 87, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs88(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 88, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs89(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 89, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs90(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 90, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs91(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 91, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs92(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 92, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs93(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 93, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs94(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 94, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs95(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 95, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs96(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 96, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs97(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 97, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs98(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 98, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs99(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 99, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs100(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 100, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs101(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 101, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs102(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 102, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs103(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 103, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs104(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 104, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs105(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 105, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs106(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 106, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs107(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 107, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs108(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 108, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs109(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 109, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs110(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 110, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs111(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 111, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs112(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 112, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs113(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 113, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs114(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 114, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs115(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 115, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs116(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 116, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs117(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 117, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs118(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 118, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs119(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 119, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs120(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 120, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs121(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 121, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs122(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 122, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs123(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 123, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs124(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 124, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs125(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 125, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs126(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 126, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs127(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 127, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs128(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 128, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs129(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 129, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs130(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 130, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs131(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 131, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs132(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 132, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs133(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 133, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs134(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 134, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs135(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 135, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs136(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 136, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs137(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 137, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs138(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 138, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs139(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 139, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs140(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 140, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs141(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 141, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs142(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 142, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs143(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 143, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs144(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 144, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs145(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 145, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs146(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 146, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs147(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 147, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs148(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 148, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs149(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 149, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs150(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 150, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs151(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 151, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs152(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 152, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs153(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 153, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs154(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 154, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs155(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 155, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs156(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 156, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs157(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 157, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs158(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 158, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs159(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 159, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs160(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 160, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs161(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 161, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs162(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 162, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs163(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 163, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs164(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 164, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs165(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 165, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs166(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 166, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs167(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 167, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs168(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 168, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs169(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 169, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs170(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 170, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs171(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 171, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs172(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 172, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs173(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 173, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs174(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 174, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs175(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 175, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs176(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 176, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs177(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 177, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs178(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 178, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs179(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 179, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs180(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 180, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs181(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 181, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs182(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 182, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs183(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 183, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs184(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 184, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs185(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 185, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs186(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 186, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs187(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 187, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs188(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 188, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs189(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 189, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs190(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 190, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs191(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 191, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs192(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 192, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs193(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 193, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs194(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 194, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs195(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 195, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs196(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 196, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs197(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 197, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs198(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 198, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs199(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 199, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs200(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 200, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs201(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 201, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs202(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 202, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs203(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 203, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs204(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 204, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs205(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 205, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs206(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 206, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs207(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 207, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs208(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 208, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs209(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 209, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs210(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 210, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs211(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 211, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs212(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 212, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs213(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 213, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs214(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 214, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs215(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 215, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs216(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 216, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs217(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 217, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs218(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 218, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs219(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 219, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs220(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 220, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs221(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 221, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs222(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 222, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs223(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 223, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs224(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 224, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs225(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 225, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs226(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 226, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs227(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 227, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs228(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 228, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs229(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 229, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs230(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 230, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs231(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 231, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs232(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 232, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs233(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 233, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs234(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 234, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs235(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 235, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs236(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 236, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs237(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 237, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs238(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 238, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs239(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 239, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs240(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 240, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs241(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 241, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs242(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 242, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs243(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 243, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs244(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 244, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs245(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 245, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs246(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 246, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs247(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 247, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs248(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 248, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs249(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 249, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs250(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 250, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs251(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 251, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs252(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 252, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs253(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 253, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs254(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 254, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs255(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 255, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs256(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 256, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs257(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 257, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs258(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 258, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs259(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 259, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs260(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 260, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs261(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 261, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs262(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 262, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs263(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 263, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs264(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 264, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs265(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 265, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs266(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 266, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs267(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 267, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs268(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 268, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs269(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 269, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs270(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 270, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs271(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 271, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs272(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 272, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs273(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 273, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs274(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 274, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs275(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 275, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs276(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 276, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs277(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 277, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs278(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 278, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs279(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 279, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs280(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 280, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs281(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 281, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs282(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 282, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs283(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 283, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs284(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 284, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs285(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 285, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs286(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 286, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs287(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 287, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs288(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 288, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs289(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 289, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs290(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 290, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs291(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 291, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs292(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 292, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs293(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 293, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs294(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 294, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs295(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 295, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs296(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 296, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs297(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 297, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs298(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 298, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs299(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 299, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs300(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 300, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs301(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 301, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs302(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 302, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs303(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 303, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs304(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 304, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs305(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 305, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs306(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 306, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs307(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 307, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs308(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 308, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs309(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 309, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs310(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 310, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs311(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 311, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs312(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 312, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs313(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 313, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs314(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 314, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs315(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 315, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs316(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 316, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs317(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 317, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs318(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 318, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs319(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 319, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs320(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 320, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs321(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 321, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs322(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 322, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs323(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 323, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs324(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 324, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs325(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 325, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs326(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 326, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs327(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 327, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs328(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 328, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs329(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 329, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs330(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 330, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs331(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 331, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs332(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 332, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs333(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 333, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs334(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 334, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs335(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 335, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs336(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 336, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs337(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 337, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs338(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 338, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs339(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 339, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs340(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 340, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs341(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 341, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs342(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 342, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs343(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 343, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs344(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 344, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs345(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 345, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs346(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 346, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs347(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 347, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs348(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 348, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs349(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 349, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs350(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 350, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs351(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 351, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs352(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 352, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs353(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 353, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs354(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 354, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs355(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 355, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs356(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 356, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs357(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 357, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs358(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 358, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs359(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 359, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs360(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 360, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs361(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 361, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs362(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 362, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs363(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 363, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs364(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 364, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs365(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 365, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs366(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 366, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs367(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 367, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs368(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 368, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs369(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 369, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs370(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 370, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs371(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 371, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs372(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 372, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs373(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 373, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs374(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 374, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs375(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 375, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs376(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 376, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs377(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 377, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs378(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 378, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs379(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 379, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs380(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 380, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs381(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 381, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs382(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 382, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs383(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 383, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs384(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 384, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs385(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 385, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs386(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 386, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs387(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 387, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs388(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 388, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs389(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 389, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs390(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 390, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs391(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 391, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs392(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 392, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs393(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 393, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs394(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 394, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs395(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 395, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs396(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 396, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs397(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 397, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs398(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 398, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs399(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 399, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs400(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 400, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs401(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 401, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs402(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 402, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs403(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 403, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs404(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 404, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs405(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 405, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs406(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 406, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs407(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 407, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs408(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 408, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs409(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 409, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs410(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 410, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs411(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 411, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs412(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 412, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs413(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 413, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs414(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 414, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs415(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 415, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs416(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 416, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs417(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 417, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs418(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 418, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs419(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 419, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs420(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 420, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs421(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 421, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs422(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 422, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs423(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 423, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs424(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 424, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs425(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 425, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs426(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 426, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs427(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 427, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs428(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 428, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs429(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 429, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs430(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 430, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs431(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 431, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs432(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 432, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs433(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 433, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs434(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 434, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs435(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 435, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs436(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 436, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs437(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 437, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs438(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 438, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs439(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 439, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs440(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 440, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs441(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 441, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs442(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 442, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs443(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 443, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs444(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 444, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs445(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 445, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs446(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 446, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs447(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 447, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs448(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 448, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs449(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 449, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs450(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 450, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs451(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 451, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs452(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 452, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs453(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 453, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs454(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 454, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs455(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 455, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs456(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 456, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs457(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 457, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs458(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 458, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs459(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 459, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs460(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 460, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs461(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 461, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs462(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 462, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs463(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 463, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs464(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 464, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs465(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 465, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs466(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 466, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs467(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 467, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs468(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 468, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs469(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 469, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs470(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 470, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs471(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 471, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs472(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 472, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs473(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 473, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs474(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 474, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs475(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 475, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs476(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 476, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs477(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 477, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs478(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 478, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs479(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 479, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs480(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 480, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs481(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 481, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs482(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 482, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs483(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 483, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs484(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 484, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs485(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 485, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs486(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 486, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs487(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 487, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs488(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 488, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs489(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 489, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs490(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 490, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs491(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 491, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs492(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 492, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs493(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 493, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs494(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 494, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs495(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 495, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs496(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 496, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs497(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 497, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs498(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 498, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs499(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 499, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs500(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 500, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs501(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 501, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs502(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 502, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs503(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 503, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs504(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 504, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs505(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 505, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs506(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 506, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs507(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 507, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs508(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 508, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs509(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 509, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs510(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 510, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs511(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 511, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs512(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 512, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs513(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 513, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs514(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 514, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs515(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 515, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs516(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 516, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs517(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 517, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs518(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 518, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs519(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 519, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs520(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 520, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs521(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 521, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs522(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 522, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs523(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 523, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs524(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 524, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs525(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 525, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs526(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 526, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs527(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 527, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs528(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 528, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs529(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 529, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs530(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 530, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs531(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 531, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs532(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 532, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs533(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 533, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs534(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 534, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs535(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 535, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs536(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 536, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs537(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 537, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs538(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 538, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs539(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 539, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs540(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 540, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs541(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 541, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs542(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 542, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs543(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 543, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs544(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 544, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs545(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 545, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs546(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 546, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs547(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 547, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs548(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 548, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs549(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 549, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs550(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 550, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs551(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 551, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs552(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 552, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs553(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 553, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs554(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 554, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs555(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 555, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs556(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 556, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs557(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 557, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs558(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 558, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs559(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 559, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs560(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 560, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs561(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 561, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs562(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 562, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs563(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 563, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs564(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 564, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs565(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 565, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs566(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 566, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs567(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 567, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs568(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 568, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs569(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 569, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs570(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 570, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs571(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 571, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs572(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 572, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs573(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 573, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs574(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 574, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs575(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 575, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs576(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 576, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs577(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 577, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs578(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 578, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs579(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 579, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs580(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 580, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs581(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 581, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs582(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 582, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs583(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 583, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs584(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 584, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs585(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 585, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs586(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 586, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs587(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 587, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs588(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 588, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs589(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 589, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs590(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 590, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs591(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 591, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs592(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 592, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs593(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 593, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs594(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 594, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs595(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 595, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs596(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 596, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs597(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 597, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs598(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 598, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs599(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 599, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs600(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 600, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs601(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 601, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs602(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 602, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs603(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 603, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs604(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 604, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs605(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 605, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs606(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 606, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs607(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 607, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs608(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 608, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs609(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 609, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs610(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 610, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs611(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 611, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs612(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 612, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs613(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 613, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs614(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 614, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs615(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 615, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs616(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 616, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs617(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 617, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs618(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 618, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs619(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 619, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs620(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 620, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs621(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 621, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs622(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 622, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs623(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 623, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs624(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 624, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs625(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 625, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs626(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 626, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs627(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 627, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs628(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 628, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs629(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 629, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs630(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 630, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs631(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 631, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs632(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 632, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs633(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 633, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs634(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 634, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs635(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 635, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs636(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 636, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs637(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 637, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs638(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 638, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs639(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 639, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs640(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 640, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs641(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 641, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs642(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 642, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs643(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 643, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs644(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 644, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs645(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 645, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs646(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 646, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs647(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 647, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs648(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 648, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs649(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 649, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs650(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 650, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs651(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 651, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs652(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 652, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs653(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 653, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs654(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 654, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs655(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 655, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs656(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 656, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs657(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 657, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs658(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 658, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs659(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 659, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs660(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 660, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs661(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 661, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs662(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 662, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs663(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 663, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs664(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 664, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs665(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 665, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs666(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 666, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs667(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 667, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs668(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 668, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs669(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 669, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs670(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 670, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs671(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 671, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs672(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 672, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs673(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 673, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs674(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 674, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs675(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 675, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs676(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 676, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs677(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 677, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs678(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 678, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs679(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 679, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs680(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 680, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs681(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 681, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs682(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 682, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs683(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 683, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs684(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 684, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs685(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 685, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs686(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 686, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs687(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 687, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs688(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 688, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs689(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 689, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs690(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 690, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs691(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 691, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs692(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 692, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs693(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 693, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs694(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 694, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs695(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 695, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs696(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 696, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs697(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 697, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs698(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 698, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs699(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 699, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs700(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 700, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs701(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 701, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs702(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 702, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs703(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 703, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs704(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 704, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs705(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 705, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs706(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 706, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs707(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 707, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs708(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 708, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs709(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 709, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs710(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 710, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs711(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 711, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs712(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 712, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs713(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 713, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs714(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 714, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs715(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 715, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs716(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 716, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs717(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 717, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs718(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 718, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs719(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 719, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs720(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 720, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs721(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 721, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs722(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 722, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs723(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 723, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs724(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 724, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs725(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 725, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs726(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 726, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs727(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 727, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs728(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 728, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs729(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 729, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs730(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 730, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs731(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 731, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs732(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 732, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs733(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 733, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs734(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 734, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs735(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 735, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs736(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 736, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs737(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 737, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs738(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 738, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs739(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 739, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs740(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 740, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs741(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 741, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs742(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 742, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs743(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 743, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs744(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 744, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs745(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 745, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs746(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 746, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs747(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 747, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs748(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 748, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs749(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 749, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs750(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 750, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs751(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 751, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs752(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 752, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs753(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 753, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs754(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 754, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs755(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 755, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs756(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 756, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs757(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 757, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs758(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 758, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs759(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 759, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs760(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 760, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs761(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 761, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs762(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 762, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs763(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 763, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs764(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 764, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs765(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 765, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs766(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 766, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs767(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 767, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs768(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 768, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs769(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 769, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs770(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 770, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs771(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 771, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs772(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 772, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs773(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 773, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs774(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 774, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs775(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 775, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs776(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 776, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs777(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 777, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs778(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 778, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs779(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 779, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs780(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 780, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs781(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 781, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs782(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 782, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs783(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 783, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs784(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 784, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs785(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 785, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs786(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 786, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs787(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 787, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs788(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 788, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs789(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 789, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs790(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 790, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs791(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 791, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs792(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 792, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs793(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 793, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs794(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 794, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs795(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 795, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs796(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 796, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs797(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 797, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs798(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 798, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs799(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 799, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs800(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 800, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs801(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 801, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs802(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 802, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs803(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 803, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs804(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 804, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs805(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 805, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs806(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 806, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs807(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 807, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs808(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 808, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs809(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 809, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs810(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 810, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs811(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 811, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs812(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 812, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs813(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 813, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs814(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 814, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs815(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 815, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs816(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 816, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs817(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 817, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs818(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 818, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs819(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 819, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs820(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 820, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs821(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 821, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs822(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 822, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs823(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 823, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs824(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 824, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs825(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 825, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs826(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 826, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs827(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 827, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs828(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 828, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs829(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 829, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs830(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 830, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs831(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 831, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs832(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 832, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs833(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 833, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs834(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 834, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs835(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 835, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs836(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 836, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs837(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 837, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs838(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 838, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs839(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 839, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs840(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 840, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs841(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 841, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs842(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 842, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs843(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 843, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs844(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 844, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs845(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 845, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs846(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 846, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs847(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 847, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs848(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 848, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs849(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 849, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs850(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 850, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs851(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 851, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs852(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 852, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs853(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 853, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs854(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 854, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs855(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 855, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs856(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 856, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs857(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 857, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs858(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 858, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs859(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 859, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs860(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 860, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs861(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 861, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs862(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 862, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs863(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 863, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs864(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 864, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs865(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 865, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs866(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 866, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs867(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 867, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs868(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 868, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs869(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 869, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs870(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 870, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs871(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 871, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs872(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 872, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs873(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 873, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs874(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 874, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs875(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 875, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs876(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 876, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs877(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 877, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs878(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 878, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs879(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 879, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs880(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 880, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs881(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 881, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs882(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 882, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs883(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 883, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs884(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 884, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs885(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 885, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs886(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 886, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs887(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 887, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs888(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 888, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs889(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 889, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs890(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 890, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs891(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 891, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs892(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 892, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs893(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 893, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs894(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 894, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs895(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 895, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs896(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 896, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs897(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 897, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs898(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 898, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs899(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 899, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs900(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 900, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs901(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 901, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs902(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 902, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs903(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 903, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs904(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 904, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs905(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 905, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs906(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 906, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs907(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 907, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs908(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 908, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs909(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 909, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs910(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 910, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs911(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 911, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs912(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 912, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs913(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 913, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs914(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 914, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs915(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 915, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs916(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 916, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs917(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 917, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs918(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 918, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs919(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 919, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs920(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 920, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs921(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 921, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs922(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 922, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs923(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 923, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs924(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 924, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs925(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 925, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs926(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 926, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs927(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 927, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs928(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 928, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs929(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 929, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs930(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 930, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs931(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 931, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs932(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 932, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs933(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 933, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs934(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 934, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs935(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 935, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs936(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 936, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs937(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 937, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs938(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 938, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs939(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 939, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs940(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 940, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs941(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 941, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs942(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 942, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs943(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 943, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs944(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 944, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs945(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 945, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs946(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 946, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs947(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 947, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs948(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 948, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs949(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 949, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs950(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 950, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs951(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 951, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs952(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 952, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs953(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 953, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs954(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 954, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs955(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 955, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs956(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 956, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs957(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 957, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs958(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 958, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs959(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 959, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs960(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 960, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs961(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 961, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs962(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 962, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs963(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 963, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs964(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 964, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs965(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 965, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs966(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 966, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs967(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 967, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs968(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 968, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs969(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 969, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs970(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 970, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs971(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 971, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs972(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 972, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs973(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 973, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs974(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 974, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs975(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 975, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs976(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 976, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs977(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 977, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs978(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 978, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs979(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 979, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs980(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 980, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs981(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 981, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs982(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 982, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs983(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 983, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs984(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 984, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs985(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 985, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs986(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 986, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs987(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 987, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs988(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 988, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs989(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 989, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs990(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 990, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs991(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 991, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs992(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 992, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs993(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 993, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs994(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 994, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs995(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 995, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs996(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 996, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs997(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 997, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs998(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 998, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs999(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 999, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1000(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1000, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1001(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1001, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1002(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1002, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1003(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1003, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1004(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1004, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1005(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1005, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1006(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1006, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1007(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1007, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1008(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1008, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1009(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1009, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1010(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1010, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1011(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1011, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1012(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1012, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1013(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1013, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1014(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1014, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1015(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1015, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1016(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1016, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1017(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1017, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1018(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1018, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1019(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1019, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1020(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1020, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1021(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1021, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1022(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1022, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1023(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1023, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F32Gs1024(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1024, float><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}

