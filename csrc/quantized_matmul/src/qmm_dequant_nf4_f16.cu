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

void LaunchDequantNf4F16(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_nf4_int<4, half><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantNf4F16Gs8(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 8, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs9(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 9, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs10(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 10, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs11(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 11, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs12(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 12, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs13(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 13, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs14(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 14, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs15(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 15, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs16(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 16, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs17(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 17, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs18(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 18, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs19(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 19, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs20(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 20, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs21(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 21, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs22(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 22, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs23(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 23, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs24(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 24, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs25(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 25, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs26(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 26, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs27(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 27, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs28(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 28, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs29(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 29, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs30(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 30, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs31(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 31, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs32(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 32, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs33(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 33, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs34(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 34, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs35(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 35, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs36(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 36, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs37(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 37, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs38(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 38, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs39(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 39, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs40(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 40, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs41(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 41, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs42(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 42, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs43(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 43, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs44(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 44, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs45(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 45, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs46(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 46, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs47(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 47, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs48(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 48, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs49(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 49, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs50(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 50, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs51(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 51, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs52(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 52, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs53(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 53, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs54(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 54, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs55(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 55, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs56(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 56, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs57(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 57, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs58(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 58, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs59(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 59, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs60(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 60, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs61(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 61, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs62(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 62, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs63(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 63, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs64(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 64, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs65(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 65, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs66(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 66, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs67(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 67, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs68(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 68, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs69(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 69, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs70(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 70, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs71(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 71, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs72(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 72, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs73(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 73, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs74(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 74, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs75(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 75, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs76(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 76, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs77(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 77, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs78(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 78, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs79(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 79, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs80(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 80, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs81(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 81, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs82(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 82, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs83(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 83, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs84(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 84, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs85(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 85, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs86(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 86, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs87(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 87, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs88(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 88, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs89(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 89, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs90(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 90, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs91(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 91, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs92(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 92, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs93(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 93, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs94(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 94, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs95(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 95, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs96(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 96, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs97(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 97, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs98(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 98, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs99(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 99, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs100(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 100, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs101(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 101, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs102(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 102, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs103(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 103, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs104(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 104, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs105(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 105, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs106(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 106, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs107(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 107, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs108(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 108, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs109(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 109, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs110(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 110, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs111(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 111, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs112(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 112, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs113(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 113, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs114(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 114, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs115(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 115, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs116(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 116, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs117(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 117, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs118(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 118, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs119(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 119, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs120(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 120, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs121(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 121, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs122(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 122, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs123(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 123, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs124(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 124, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs125(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 125, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs126(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 126, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs127(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 127, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs128(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 128, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs129(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 129, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs130(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 130, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs131(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 131, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs132(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 132, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs133(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 133, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs134(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 134, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs135(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 135, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs136(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 136, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs137(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 137, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs138(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 138, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs139(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 139, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs140(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 140, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs141(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 141, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs142(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 142, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs143(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 143, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs144(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 144, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs145(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 145, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs146(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 146, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs147(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 147, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs148(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 148, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs149(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 149, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs150(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 150, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs151(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 151, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs152(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 152, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs153(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 153, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs154(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 154, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs155(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 155, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs156(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 156, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs157(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 157, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs158(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 158, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs159(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 159, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs160(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 160, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs161(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 161, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs162(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 162, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs163(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 163, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs164(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 164, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs165(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 165, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs166(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 166, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs167(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 167, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs168(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 168, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs169(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 169, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs170(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 170, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs171(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 171, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs172(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 172, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs173(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 173, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs174(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 174, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs175(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 175, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs176(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 176, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs177(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 177, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs178(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 178, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs179(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 179, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs180(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 180, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs181(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 181, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs182(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 182, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs183(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 183, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs184(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 184, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs185(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 185, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs186(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 186, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs187(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 187, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs188(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 188, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs189(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 189, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs190(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 190, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs191(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 191, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs192(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 192, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs193(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 193, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs194(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 194, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs195(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 195, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs196(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 196, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs197(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 197, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs198(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 198, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs199(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 199, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs200(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 200, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs201(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 201, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs202(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 202, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs203(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 203, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs204(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 204, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs205(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 205, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs206(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 206, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs207(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 207, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs208(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 208, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs209(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 209, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs210(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 210, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs211(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 211, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs212(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 212, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs213(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 213, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs214(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 214, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs215(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 215, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs216(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 216, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs217(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 217, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs218(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 218, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs219(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 219, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs220(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 220, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs221(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 221, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs222(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 222, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs223(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 223, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs224(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 224, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs225(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 225, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs226(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 226, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs227(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 227, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs228(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 228, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs229(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 229, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs230(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 230, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs231(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 231, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs232(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 232, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs233(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 233, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs234(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 234, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs235(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 235, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs236(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 236, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs237(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 237, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs238(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 238, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs239(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 239, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs240(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 240, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs241(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 241, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs242(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 242, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs243(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 243, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs244(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 244, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs245(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 245, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs246(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 246, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs247(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 247, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs248(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 248, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs249(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 249, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs250(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 250, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs251(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 251, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs252(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 252, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs253(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 253, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs254(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 254, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs255(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 255, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs256(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 256, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs257(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 257, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs258(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 258, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs259(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 259, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs260(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 260, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs261(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 261, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs262(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 262, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs263(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 263, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs264(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 264, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs265(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 265, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs266(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 266, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs267(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 267, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs268(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 268, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs269(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 269, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs270(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 270, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs271(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 271, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs272(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 272, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs273(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 273, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs274(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 274, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs275(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 275, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs276(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 276, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs277(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 277, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs278(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 278, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs279(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 279, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs280(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 280, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs281(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 281, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs282(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 282, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs283(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 283, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs284(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 284, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs285(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 285, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs286(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 286, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs287(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 287, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs288(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 288, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs289(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 289, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs290(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 290, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs291(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 291, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs292(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 292, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs293(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 293, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs294(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 294, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs295(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 295, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs296(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 296, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs297(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 297, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs298(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 298, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs299(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 299, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs300(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 300, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs301(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 301, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs302(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 302, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs303(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 303, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs304(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 304, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs305(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 305, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs306(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 306, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs307(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 307, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs308(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 308, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs309(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 309, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs310(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 310, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs311(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 311, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs312(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 312, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs313(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 313, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs314(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 314, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs315(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 315, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs316(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 316, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs317(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 317, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs318(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 318, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs319(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 319, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs320(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 320, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs321(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 321, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs322(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 322, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs323(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 323, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs324(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 324, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs325(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 325, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs326(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 326, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs327(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 327, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs328(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 328, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs329(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 329, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs330(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 330, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs331(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 331, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs332(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 332, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs333(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 333, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs334(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 334, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs335(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 335, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs336(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 336, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs337(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 337, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs338(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 338, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs339(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 339, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs340(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 340, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs341(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 341, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs342(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 342, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs343(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 343, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs344(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 344, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs345(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 345, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs346(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 346, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs347(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 347, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs348(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 348, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs349(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 349, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs350(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 350, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs351(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 351, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs352(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 352, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs353(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 353, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs354(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 354, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs355(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 355, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs356(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 356, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs357(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 357, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs358(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 358, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs359(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 359, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs360(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 360, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs361(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 361, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs362(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 362, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs363(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 363, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs364(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 364, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs365(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 365, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs366(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 366, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs367(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 367, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs368(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 368, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs369(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 369, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs370(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 370, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs371(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 371, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs372(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 372, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs373(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 373, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs374(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 374, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs375(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 375, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs376(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 376, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs377(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 377, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs378(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 378, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs379(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 379, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs380(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 380, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs381(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 381, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs382(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 382, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs383(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 383, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs384(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 384, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs385(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 385, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs386(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 386, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs387(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 387, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs388(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 388, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs389(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 389, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs390(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 390, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs391(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 391, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs392(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 392, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs393(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 393, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs394(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 394, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs395(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 395, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs396(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 396, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs397(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 397, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs398(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 398, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs399(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 399, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs400(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 400, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs401(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 401, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs402(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 402, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs403(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 403, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs404(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 404, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs405(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 405, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs406(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 406, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs407(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 407, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs408(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 408, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs409(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 409, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs410(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 410, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs411(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 411, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs412(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 412, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs413(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 413, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs414(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 414, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs415(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 415, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs416(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 416, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs417(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 417, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs418(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 418, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs419(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 419, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs420(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 420, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs421(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 421, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs422(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 422, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs423(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 423, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs424(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 424, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs425(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 425, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs426(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 426, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs427(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 427, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs428(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 428, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs429(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 429, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs430(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 430, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs431(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 431, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs432(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 432, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs433(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 433, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs434(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 434, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs435(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 435, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs436(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 436, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs437(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 437, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs438(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 438, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs439(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 439, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs440(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 440, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs441(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 441, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs442(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 442, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs443(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 443, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs444(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 444, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs445(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 445, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs446(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 446, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs447(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 447, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs448(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 448, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs449(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 449, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs450(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 450, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs451(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 451, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs452(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 452, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs453(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 453, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs454(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 454, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs455(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 455, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs456(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 456, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs457(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 457, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs458(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 458, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs459(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 459, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs460(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 460, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs461(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 461, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs462(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 462, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs463(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 463, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs464(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 464, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs465(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 465, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs466(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 466, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs467(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 467, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs468(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 468, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs469(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 469, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs470(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 470, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs471(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 471, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs472(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 472, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs473(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 473, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs474(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 474, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs475(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 475, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs476(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 476, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs477(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 477, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs478(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 478, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs479(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 479, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs480(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 480, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs481(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 481, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs482(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 482, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs483(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 483, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs484(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 484, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs485(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 485, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs486(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 486, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs487(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 487, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs488(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 488, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs489(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 489, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs490(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 490, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs491(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 491, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs492(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 492, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs493(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 493, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs494(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 494, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs495(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 495, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs496(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 496, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs497(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 497, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs498(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 498, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs499(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 499, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs500(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 500, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs501(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 501, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs502(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 502, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs503(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 503, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs504(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 504, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs505(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 505, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs506(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 506, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs507(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 507, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs508(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 508, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs509(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 509, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs510(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 510, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs511(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 511, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs512(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 512, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs513(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 513, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs514(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 514, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs515(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 515, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs516(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 516, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs517(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 517, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs518(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 518, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs519(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 519, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs520(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 520, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs521(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 521, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs522(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 522, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs523(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 523, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs524(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 524, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs525(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 525, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs526(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 526, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs527(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 527, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs528(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 528, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs529(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 529, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs530(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 530, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs531(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 531, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs532(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 532, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs533(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 533, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs534(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 534, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs535(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 535, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs536(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 536, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs537(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 537, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs538(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 538, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs539(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 539, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs540(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 540, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs541(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 541, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs542(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 542, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs543(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 543, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs544(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 544, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs545(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 545, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs546(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 546, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs547(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 547, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs548(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 548, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs549(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 549, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs550(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 550, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs551(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 551, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs552(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 552, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs553(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 553, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs554(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 554, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs555(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 555, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs556(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 556, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs557(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 557, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs558(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 558, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs559(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 559, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs560(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 560, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs561(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 561, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs562(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 562, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs563(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 563, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs564(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 564, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs565(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 565, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs566(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 566, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs567(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 567, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs568(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 568, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs569(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 569, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs570(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 570, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs571(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 571, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs572(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 572, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs573(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 573, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs574(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 574, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs575(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 575, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs576(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 576, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs577(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 577, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs578(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 578, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs579(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 579, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs580(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 580, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs581(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 581, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs582(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 582, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs583(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 583, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs584(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 584, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs585(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 585, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs586(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 586, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs587(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 587, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs588(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 588, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs589(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 589, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs590(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 590, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs591(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 591, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs592(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 592, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs593(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 593, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs594(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 594, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs595(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 595, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs596(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 596, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs597(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 597, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs598(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 598, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs599(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 599, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs600(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 600, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs601(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 601, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs602(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 602, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs603(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 603, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs604(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 604, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs605(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 605, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs606(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 606, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs607(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 607, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs608(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 608, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs609(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 609, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs610(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 610, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs611(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 611, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs612(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 612, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs613(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 613, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs614(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 614, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs615(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 615, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs616(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 616, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs617(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 617, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs618(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 618, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs619(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 619, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs620(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 620, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs621(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 621, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs622(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 622, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs623(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 623, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs624(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 624, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs625(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 625, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs626(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 626, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs627(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 627, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs628(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 628, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs629(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 629, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs630(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 630, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs631(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 631, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs632(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 632, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs633(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 633, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs634(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 634, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs635(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 635, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs636(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 636, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs637(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 637, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs638(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 638, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs639(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 639, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs640(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 640, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs641(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 641, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs642(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 642, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs643(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 643, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs644(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 644, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs645(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 645, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs646(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 646, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs647(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 647, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs648(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 648, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs649(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 649, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs650(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 650, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs651(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 651, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs652(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 652, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs653(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 653, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs654(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 654, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs655(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 655, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs656(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 656, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs657(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 657, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs658(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 658, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs659(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 659, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs660(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 660, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs661(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 661, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs662(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 662, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs663(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 663, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs664(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 664, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs665(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 665, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs666(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 666, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs667(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 667, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs668(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 668, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs669(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 669, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs670(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 670, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs671(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 671, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs672(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 672, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs673(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 673, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs674(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 674, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs675(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 675, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs676(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 676, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs677(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 677, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs678(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 678, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs679(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 679, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs680(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 680, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs681(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 681, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs682(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 682, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs683(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 683, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs684(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 684, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs685(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 685, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs686(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 686, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs687(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 687, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs688(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 688, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs689(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 689, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs690(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 690, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs691(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 691, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs692(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 692, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs693(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 693, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs694(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 694, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs695(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 695, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs696(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 696, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs697(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 697, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs698(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 698, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs699(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 699, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs700(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 700, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs701(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 701, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs702(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 702, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs703(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 703, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs704(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 704, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs705(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 705, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs706(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 706, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs707(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 707, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs708(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 708, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs709(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 709, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs710(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 710, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs711(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 711, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs712(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 712, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs713(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 713, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs714(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 714, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs715(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 715, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs716(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 716, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs717(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 717, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs718(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 718, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs719(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 719, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs720(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 720, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs721(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 721, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs722(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 722, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs723(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 723, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs724(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 724, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs725(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 725, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs726(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 726, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs727(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 727, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs728(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 728, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs729(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 729, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs730(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 730, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs731(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 731, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs732(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 732, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs733(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 733, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs734(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 734, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs735(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 735, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs736(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 736, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs737(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 737, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs738(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 738, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs739(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 739, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs740(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 740, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs741(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 741, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs742(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 742, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs743(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 743, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs744(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 744, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs745(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 745, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs746(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 746, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs747(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 747, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs748(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 748, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs749(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 749, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs750(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 750, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs751(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 751, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs752(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 752, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs753(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 753, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs754(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 754, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs755(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 755, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs756(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 756, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs757(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 757, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs758(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 758, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs759(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 759, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs760(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 760, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs761(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 761, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs762(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 762, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs763(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 763, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs764(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 764, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs765(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 765, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs766(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 766, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs767(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 767, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs768(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 768, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs769(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 769, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs770(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 770, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs771(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 771, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs772(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 772, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs773(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 773, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs774(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 774, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs775(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 775, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs776(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 776, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs777(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 777, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs778(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 778, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs779(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 779, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs780(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 780, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs781(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 781, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs782(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 782, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs783(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 783, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs784(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 784, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs785(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 785, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs786(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 786, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs787(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 787, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs788(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 788, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs789(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 789, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs790(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 790, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs791(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 791, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs792(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 792, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs793(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 793, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs794(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 794, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs795(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 795, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs796(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 796, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs797(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 797, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs798(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 798, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs799(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 799, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs800(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 800, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs801(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 801, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs802(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 802, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs803(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 803, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs804(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 804, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs805(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 805, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs806(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 806, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs807(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 807, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs808(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 808, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs809(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 809, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs810(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 810, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs811(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 811, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs812(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 812, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs813(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 813, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs814(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 814, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs815(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 815, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs816(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 816, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs817(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 817, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs818(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 818, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs819(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 819, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs820(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 820, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs821(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 821, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs822(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 822, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs823(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 823, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs824(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 824, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs825(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 825, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs826(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 826, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs827(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 827, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs828(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 828, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs829(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 829, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs830(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 830, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs831(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 831, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs832(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 832, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs833(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 833, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs834(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 834, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs835(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 835, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs836(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 836, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs837(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 837, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs838(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 838, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs839(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 839, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs840(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 840, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs841(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 841, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs842(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 842, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs843(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 843, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs844(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 844, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs845(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 845, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs846(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 846, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs847(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 847, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs848(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 848, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs849(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 849, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs850(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 850, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs851(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 851, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs852(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 852, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs853(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 853, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs854(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 854, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs855(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 855, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs856(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 856, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs857(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 857, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs858(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 858, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs859(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 859, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs860(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 860, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs861(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 861, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs862(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 862, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs863(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 863, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs864(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 864, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs865(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 865, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs866(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 866, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs867(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 867, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs868(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 868, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs869(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 869, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs870(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 870, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs871(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 871, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs872(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 872, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs873(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 873, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs874(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 874, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs875(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 875, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs876(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 876, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs877(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 877, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs878(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 878, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs879(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 879, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs880(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 880, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs881(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 881, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs882(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 882, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs883(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 883, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs884(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 884, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs885(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 885, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs886(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 886, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs887(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 887, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs888(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 888, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs889(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 889, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs890(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 890, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs891(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 891, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs892(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 892, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs893(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 893, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs894(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 894, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs895(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 895, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs896(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 896, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs897(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 897, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs898(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 898, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs899(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 899, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs900(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 900, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs901(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 901, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs902(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 902, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs903(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 903, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs904(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 904, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs905(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 905, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs906(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 906, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs907(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 907, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs908(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 908, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs909(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 909, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs910(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 910, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs911(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 911, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs912(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 912, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs913(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 913, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs914(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 914, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs915(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 915, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs916(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 916, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs917(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 917, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs918(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 918, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs919(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 919, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs920(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 920, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs921(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 921, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs922(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 922, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs923(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 923, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs924(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 924, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs925(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 925, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs926(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 926, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs927(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 927, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs928(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 928, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs929(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 929, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs930(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 930, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs931(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 931, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs932(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 932, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs933(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 933, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs934(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 934, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs935(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 935, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs936(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 936, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs937(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 937, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs938(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 938, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs939(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 939, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs940(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 940, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs941(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 941, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs942(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 942, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs943(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 943, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs944(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 944, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs945(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 945, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs946(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 946, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs947(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 947, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs948(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 948, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs949(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 949, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs950(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 950, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs951(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 951, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs952(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 952, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs953(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 953, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs954(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 954, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs955(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 955, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs956(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 956, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs957(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 957, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs958(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 958, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs959(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 959, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs960(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 960, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs961(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 961, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs962(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 962, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs963(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 963, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs964(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 964, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs965(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 965, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs966(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 966, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs967(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 967, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs968(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 968, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs969(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 969, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs970(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 970, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs971(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 971, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs972(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 972, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs973(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 973, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs974(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 974, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs975(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 975, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs976(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 976, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs977(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 977, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs978(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 978, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs979(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 979, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs980(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 980, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs981(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 981, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs982(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 982, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs983(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 983, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs984(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 984, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs985(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 985, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs986(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 986, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs987(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 987, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs988(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 988, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs989(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 989, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs990(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 990, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs991(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 991, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs992(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 992, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs993(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 993, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs994(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 994, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs995(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 995, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs996(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 996, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs997(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 997, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs998(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 998, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs999(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 999, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1000(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1000, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1001(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1001, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1002(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1002, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1003(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1003, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1004(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1004, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1005(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1005, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1006(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1006, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1007(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1007, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1008(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1008, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1009(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1009, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1010(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1010, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1011(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1011, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1012(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1012, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1013(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1013, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1014(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1014, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1015(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1015, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1016(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1016, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1017(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1017, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1018(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1018, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1019(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1019, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1020(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1020, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1021(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1021, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1022(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1022, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1023(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1023, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1024(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1024, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}

