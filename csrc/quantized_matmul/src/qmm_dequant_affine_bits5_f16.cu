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

void LaunchDequantAffineBits5F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<5, half, half><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits5F16Gs8(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 8, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs9(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 9, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs10(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 10, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs11(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 11, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs12(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 12, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs13(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 13, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs14(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 14, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs15(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 15, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 16, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs17(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 17, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs18(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 18, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs19(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 19, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs20(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 20, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs21(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 21, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs22(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 22, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs23(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 23, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs24(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 24, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs25(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 25, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs26(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 26, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs27(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 27, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs28(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 28, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs29(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 29, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs30(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 30, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs31(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 31, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 32, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs33(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 33, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs34(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 34, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs35(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 35, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs36(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 36, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs37(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 37, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs38(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 38, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs39(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 39, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs40(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 40, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs41(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 41, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs42(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 42, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs43(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 43, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs44(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 44, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs45(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 45, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs46(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 46, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs47(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 47, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs48(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 48, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs49(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 49, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs50(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 50, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs51(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 51, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs52(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 52, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs53(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 53, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs54(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 54, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs55(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 55, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs56(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 56, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs57(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 57, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs58(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 58, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs59(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 59, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs60(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 60, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs61(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 61, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs62(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 62, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs63(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 63, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 64, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs65(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 65, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs66(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 66, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs67(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 67, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs68(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 68, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs69(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 69, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs70(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 70, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs71(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 71, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs72(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 72, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs73(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 73, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs74(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 74, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs75(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 75, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs76(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 76, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs77(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 77, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs78(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 78, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs79(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 79, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs80(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 80, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs81(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 81, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs82(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 82, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs83(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 83, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs84(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 84, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs85(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 85, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs86(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 86, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs87(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 87, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs88(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 88, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs89(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 89, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs90(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 90, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs91(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 91, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs92(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 92, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs93(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 93, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs94(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 94, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs95(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 95, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs96(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 96, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs97(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 97, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs98(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 98, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs99(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 99, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs100(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 100, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs101(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 101, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs102(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 102, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs103(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 103, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs104(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 104, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs105(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 105, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs106(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 106, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs107(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 107, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs108(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 108, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs109(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 109, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs110(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 110, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs111(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 111, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs112(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 112, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs113(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 113, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs114(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 114, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs115(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 115, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs116(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 116, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs117(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 117, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs118(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 118, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs119(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 119, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs120(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 120, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs121(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 121, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs122(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 122, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs123(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 123, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs124(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 124, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs125(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 125, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs126(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 126, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs127(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 127, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 128, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs129(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 129, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs130(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 130, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs131(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 131, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs132(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 132, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs133(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 133, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs134(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 134, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs135(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 135, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs136(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 136, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs137(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 137, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs138(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 138, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs139(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 139, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs140(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 140, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs141(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 141, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs142(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 142, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs143(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 143, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs144(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 144, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs145(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 145, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs146(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 146, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs147(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 147, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs148(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 148, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs149(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 149, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs150(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 150, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs151(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 151, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs152(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 152, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs153(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 153, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs154(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 154, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs155(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 155, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs156(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 156, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs157(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 157, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs158(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 158, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs159(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 159, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs160(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 160, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs161(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 161, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs162(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 162, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs163(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 163, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs164(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 164, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs165(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 165, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs166(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 166, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs167(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 167, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs168(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 168, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs169(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 169, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs170(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 170, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs171(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 171, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs172(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 172, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs173(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 173, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs174(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 174, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs175(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 175, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs176(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 176, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs177(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 177, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs178(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 178, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs179(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 179, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs180(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 180, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs181(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 181, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs182(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 182, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs183(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 183, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs184(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 184, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs185(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 185, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs186(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 186, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs187(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 187, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs188(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 188, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs189(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 189, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs190(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 190, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs191(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 191, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs192(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 192, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs193(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 193, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs194(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 194, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs195(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 195, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs196(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 196, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs197(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 197, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs198(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 198, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs199(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 199, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs200(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 200, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs201(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 201, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs202(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 202, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs203(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 203, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs204(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 204, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs205(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 205, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs206(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 206, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs207(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 207, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs208(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 208, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs209(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 209, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs210(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 210, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs211(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 211, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs212(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 212, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs213(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 213, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs214(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 214, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs215(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 215, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs216(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 216, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs217(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 217, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs218(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 218, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs219(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 219, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs220(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 220, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs221(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 221, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs222(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 222, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs223(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 223, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs224(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 224, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs225(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 225, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs226(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 226, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs227(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 227, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs228(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 228, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs229(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 229, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs230(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 230, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs231(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 231, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs232(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 232, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs233(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 233, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs234(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 234, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs235(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 235, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs236(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 236, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs237(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 237, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs238(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 238, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs239(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 239, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs240(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 240, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs241(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 241, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs242(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 242, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs243(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 243, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs244(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 244, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs245(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 245, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs246(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 246, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs247(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 247, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs248(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 248, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs249(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 249, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs250(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 250, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs251(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 251, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs252(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 252, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs253(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 253, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs254(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 254, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs255(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 255, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 256, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs257(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 257, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs258(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 258, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs259(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 259, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs260(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 260, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs261(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 261, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs262(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 262, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs263(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 263, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs264(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 264, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs265(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 265, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs266(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 266, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs267(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 267, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs268(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 268, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs269(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 269, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs270(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 270, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs271(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 271, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs272(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 272, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs273(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 273, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs274(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 274, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs275(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 275, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs276(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 276, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs277(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 277, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs278(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 278, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs279(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 279, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs280(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 280, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs281(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 281, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs282(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 282, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs283(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 283, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs284(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 284, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs285(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 285, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs286(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 286, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs287(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 287, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs288(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 288, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs289(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 289, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs290(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 290, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs291(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 291, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs292(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 292, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs293(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 293, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs294(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 294, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs295(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 295, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs296(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 296, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs297(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 297, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs298(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 298, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs299(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 299, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs300(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 300, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs301(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 301, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs302(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 302, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs303(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 303, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs304(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 304, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs305(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 305, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs306(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 306, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs307(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 307, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs308(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 308, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs309(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 309, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs310(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 310, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs311(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 311, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs312(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 312, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs313(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 313, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs314(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 314, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs315(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 315, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs316(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 316, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs317(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 317, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs318(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 318, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs319(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 319, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs320(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 320, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs321(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 321, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs322(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 322, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs323(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 323, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs324(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 324, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs325(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 325, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs326(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 326, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs327(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 327, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs328(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 328, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs329(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 329, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs330(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 330, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs331(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 331, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs332(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 332, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs333(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 333, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs334(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 334, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs335(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 335, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs336(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 336, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs337(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 337, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs338(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 338, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs339(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 339, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs340(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 340, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs341(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 341, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs342(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 342, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs343(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 343, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs344(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 344, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs345(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 345, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs346(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 346, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs347(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 347, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs348(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 348, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs349(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 349, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs350(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 350, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs351(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 351, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs352(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 352, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs353(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 353, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs354(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 354, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs355(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 355, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs356(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 356, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs357(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 357, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs358(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 358, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs359(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 359, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs360(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 360, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs361(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 361, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs362(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 362, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs363(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 363, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs364(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 364, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs365(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 365, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs366(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 366, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs367(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 367, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs368(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 368, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs369(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 369, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs370(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 370, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs371(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 371, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs372(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 372, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs373(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 373, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs374(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 374, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs375(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 375, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs376(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 376, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs377(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 377, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs378(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 378, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs379(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 379, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs380(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 380, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs381(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 381, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs382(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 382, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs383(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 383, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs384(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 384, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs385(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 385, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs386(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 386, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs387(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 387, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs388(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 388, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs389(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 389, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs390(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 390, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs391(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 391, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs392(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 392, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs393(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 393, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs394(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 394, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs395(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 395, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs396(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 396, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs397(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 397, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs398(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 398, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs399(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 399, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs400(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 400, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs401(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 401, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs402(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 402, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs403(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 403, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs404(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 404, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs405(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 405, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs406(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 406, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs407(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 407, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs408(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 408, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs409(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 409, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs410(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 410, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs411(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 411, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs412(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 412, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs413(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 413, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs414(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 414, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs415(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 415, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs416(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 416, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs417(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 417, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs418(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 418, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs419(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 419, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs420(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 420, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs421(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 421, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs422(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 422, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs423(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 423, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs424(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 424, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs425(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 425, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs426(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 426, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs427(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 427, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs428(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 428, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs429(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 429, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs430(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 430, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs431(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 431, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs432(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 432, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs433(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 433, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs434(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 434, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs435(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 435, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs436(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 436, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs437(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 437, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs438(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 438, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs439(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 439, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs440(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 440, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs441(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 441, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs442(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 442, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs443(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 443, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs444(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 444, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs445(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 445, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs446(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 446, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs447(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 447, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs448(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 448, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs449(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 449, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs450(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 450, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs451(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 451, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs452(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 452, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs453(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 453, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs454(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 454, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs455(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 455, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs456(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 456, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs457(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 457, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs458(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 458, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs459(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 459, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs460(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 460, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs461(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 461, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs462(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 462, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs463(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 463, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs464(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 464, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs465(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 465, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs466(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 466, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs467(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 467, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs468(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 468, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs469(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 469, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs470(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 470, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs471(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 471, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs472(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 472, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs473(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 473, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs474(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 474, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs475(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 475, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs476(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 476, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs477(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 477, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs478(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 478, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs479(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 479, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs480(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 480, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs481(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 481, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs482(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 482, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs483(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 483, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs484(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 484, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs485(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 485, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs486(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 486, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs487(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 487, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs488(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 488, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs489(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 489, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs490(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 490, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs491(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 491, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs492(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 492, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs493(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 493, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs494(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 494, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs495(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 495, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs496(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 496, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs497(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 497, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs498(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 498, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs499(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 499, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs500(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 500, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs501(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 501, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs502(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 502, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs503(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 503, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs504(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 504, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs505(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 505, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs506(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 506, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs507(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 507, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs508(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 508, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs509(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 509, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs510(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 510, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs511(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 511, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 512, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs513(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 513, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs514(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 514, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs515(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 515, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs516(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 516, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs517(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 517, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs518(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 518, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs519(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 519, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs520(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 520, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs521(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 521, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs522(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 522, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs523(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 523, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs524(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 524, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs525(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 525, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs526(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 526, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs527(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 527, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs528(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 528, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs529(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 529, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs530(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 530, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs531(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 531, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs532(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 532, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs533(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 533, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs534(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 534, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs535(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 535, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs536(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 536, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs537(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 537, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs538(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 538, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs539(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 539, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs540(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 540, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs541(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 541, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs542(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 542, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs543(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 543, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs544(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 544, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs545(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 545, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs546(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 546, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs547(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 547, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs548(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 548, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs549(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 549, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs550(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 550, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs551(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 551, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs552(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 552, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs553(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 553, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs554(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 554, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs555(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 555, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs556(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 556, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs557(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 557, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs558(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 558, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs559(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 559, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs560(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 560, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs561(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 561, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs562(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 562, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs563(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 563, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs564(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 564, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs565(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 565, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs566(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 566, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs567(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 567, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs568(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 568, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs569(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 569, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs570(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 570, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs571(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 571, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs572(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 572, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs573(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 573, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs574(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 574, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs575(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 575, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs576(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 576, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs577(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 577, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs578(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 578, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs579(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 579, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs580(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 580, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs581(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 581, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs582(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 582, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs583(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 583, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs584(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 584, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs585(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 585, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs586(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 586, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs587(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 587, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs588(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 588, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs589(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 589, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs590(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 590, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs591(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 591, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs592(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 592, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs593(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 593, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs594(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 594, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs595(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 595, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs596(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 596, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs597(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 597, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs598(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 598, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs599(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 599, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs600(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 600, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs601(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 601, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs602(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 602, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs603(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 603, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs604(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 604, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs605(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 605, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs606(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 606, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs607(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 607, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs608(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 608, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs609(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 609, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs610(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 610, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs611(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 611, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs612(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 612, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs613(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 613, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs614(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 614, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs615(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 615, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs616(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 616, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs617(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 617, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs618(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 618, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs619(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 619, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs620(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 620, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs621(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 621, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs622(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 622, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs623(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 623, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs624(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 624, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs625(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 625, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs626(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 626, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs627(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 627, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs628(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 628, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs629(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 629, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs630(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 630, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs631(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 631, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs632(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 632, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs633(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 633, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs634(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 634, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs635(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 635, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs636(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 636, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs637(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 637, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs638(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 638, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs639(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 639, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs640(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 640, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs641(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 641, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs642(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 642, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs643(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 643, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs644(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 644, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs645(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 645, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs646(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 646, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs647(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 647, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs648(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 648, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs649(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 649, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs650(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 650, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs651(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 651, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs652(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 652, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs653(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 653, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs654(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 654, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs655(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 655, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs656(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 656, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs657(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 657, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs658(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 658, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs659(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 659, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs660(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 660, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs661(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 661, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs662(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 662, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs663(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 663, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs664(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 664, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs665(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 665, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs666(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 666, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs667(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 667, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs668(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 668, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs669(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 669, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs670(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 670, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs671(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 671, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs672(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 672, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs673(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 673, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs674(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 674, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs675(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 675, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs676(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 676, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs677(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 677, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs678(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 678, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs679(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 679, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs680(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 680, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs681(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 681, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs682(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 682, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs683(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 683, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs684(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 684, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs685(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 685, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs686(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 686, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs687(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 687, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs688(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 688, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs689(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 689, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs690(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 690, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs691(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 691, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs692(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 692, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs693(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 693, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs694(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 694, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs695(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 695, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs696(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 696, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs697(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 697, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs698(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 698, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs699(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 699, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs700(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 700, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs701(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 701, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs702(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 702, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs703(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 703, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs704(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 704, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs705(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 705, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs706(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 706, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs707(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 707, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs708(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 708, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs709(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 709, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs710(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 710, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs711(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 711, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs712(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 712, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs713(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 713, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs714(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 714, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs715(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 715, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs716(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 716, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs717(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 717, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs718(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 718, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs719(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 719, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs720(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 720, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs721(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 721, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs722(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 722, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs723(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 723, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs724(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 724, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs725(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 725, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs726(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 726, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs727(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 727, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs728(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 728, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs729(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 729, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs730(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 730, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs731(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 731, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs732(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 732, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs733(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 733, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs734(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 734, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs735(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 735, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs736(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 736, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs737(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 737, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs738(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 738, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs739(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 739, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs740(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 740, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs741(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 741, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs742(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 742, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs743(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 743, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs744(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 744, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs745(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 745, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs746(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 746, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs747(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 747, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs748(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 748, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs749(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 749, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs750(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 750, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs751(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 751, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs752(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 752, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs753(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 753, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs754(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 754, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs755(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 755, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs756(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 756, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs757(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 757, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs758(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 758, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs759(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 759, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs760(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 760, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs761(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 761, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs762(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 762, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs763(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 763, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs764(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 764, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs765(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 765, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs766(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 766, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs767(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 767, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs768(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 768, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs769(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 769, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs770(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 770, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs771(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 771, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs772(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 772, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs773(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 773, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs774(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 774, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs775(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 775, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs776(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 776, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs777(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 777, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs778(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 778, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs779(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 779, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs780(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 780, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs781(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 781, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs782(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 782, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs783(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 783, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs784(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 784, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs785(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 785, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs786(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 786, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs787(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 787, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs788(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 788, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs789(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 789, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs790(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 790, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs791(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 791, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs792(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 792, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs793(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 793, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs794(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 794, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs795(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 795, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs796(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 796, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs797(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 797, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs798(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 798, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs799(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 799, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs800(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 800, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs801(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 801, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs802(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 802, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs803(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 803, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs804(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 804, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs805(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 805, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs806(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 806, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs807(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 807, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs808(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 808, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs809(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 809, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs810(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 810, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs811(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 811, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs812(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 812, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs813(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 813, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs814(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 814, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs815(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 815, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs816(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 816, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs817(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 817, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs818(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 818, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs819(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 819, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs820(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 820, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs821(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 821, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs822(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 822, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs823(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 823, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs824(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 824, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs825(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 825, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs826(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 826, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs827(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 827, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs828(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 828, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs829(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 829, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs830(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 830, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs831(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 831, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs832(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 832, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs833(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 833, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs834(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 834, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs835(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 835, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs836(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 836, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs837(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 837, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs838(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 838, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs839(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 839, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs840(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 840, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs841(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 841, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs842(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 842, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs843(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 843, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs844(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 844, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs845(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 845, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs846(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 846, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs847(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 847, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs848(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 848, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs849(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 849, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs850(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 850, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs851(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 851, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs852(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 852, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs853(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 853, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs854(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 854, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs855(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 855, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs856(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 856, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs857(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 857, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs858(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 858, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs859(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 859, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs860(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 860, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs861(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 861, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs862(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 862, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs863(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 863, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs864(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 864, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs865(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 865, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs866(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 866, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs867(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 867, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs868(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 868, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs869(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 869, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs870(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 870, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs871(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 871, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs872(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 872, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs873(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 873, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs874(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 874, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs875(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 875, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs876(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 876, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs877(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 877, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs878(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 878, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs879(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 879, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs880(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 880, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs881(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 881, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs882(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 882, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs883(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 883, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs884(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 884, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs885(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 885, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs886(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 886, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs887(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 887, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs888(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 888, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs889(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 889, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs890(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 890, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs891(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 891, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs892(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 892, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs893(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 893, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs894(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 894, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs895(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 895, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs896(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 896, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs897(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 897, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs898(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 898, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs899(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 899, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs900(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 900, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs901(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 901, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs902(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 902, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs903(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 903, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs904(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 904, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs905(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 905, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs906(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 906, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs907(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 907, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs908(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 908, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs909(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 909, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs910(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 910, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs911(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 911, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs912(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 912, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs913(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 913, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs914(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 914, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs915(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 915, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs916(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 916, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs917(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 917, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs918(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 918, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs919(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 919, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs920(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 920, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs921(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 921, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs922(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 922, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs923(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 923, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs924(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 924, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs925(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 925, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs926(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 926, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs927(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 927, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs928(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 928, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs929(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 929, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs930(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 930, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs931(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 931, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs932(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 932, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs933(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 933, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs934(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 934, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs935(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 935, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs936(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 936, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs937(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 937, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs938(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 938, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs939(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 939, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs940(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 940, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs941(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 941, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs942(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 942, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs943(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 943, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs944(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 944, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs945(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 945, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs946(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 946, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs947(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 947, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs948(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 948, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs949(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 949, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs950(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 950, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs951(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 951, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs952(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 952, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs953(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 953, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs954(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 954, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs955(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 955, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs956(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 956, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs957(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 957, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs958(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 958, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs959(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 959, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs960(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 960, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs961(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 961, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs962(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 962, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs963(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 963, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs964(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 964, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs965(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 965, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs966(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 966, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs967(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 967, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs968(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 968, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs969(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 969, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs970(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 970, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs971(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 971, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs972(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 972, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs973(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 973, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs974(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 974, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs975(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 975, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs976(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 976, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs977(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 977, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs978(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 978, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs979(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 979, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs980(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 980, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs981(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 981, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs982(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 982, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs983(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 983, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs984(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 984, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs985(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 985, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs986(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 986, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs987(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 987, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs988(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 988, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs989(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 989, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs990(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 990, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs991(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 991, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs992(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 992, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs993(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 993, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs994(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 994, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs995(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 995, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs996(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 996, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs997(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 997, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs998(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 998, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs999(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 999, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1000(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1000, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1001(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1001, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1002(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1002, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1003(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1003, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1004(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1004, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1005(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1005, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1006(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1006, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1007(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1007, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1008(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1008, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1009(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1009, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1010(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1010, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1011(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1011, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1012(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1012, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1013(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1013, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1014(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1014, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1015(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1015, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1016(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1016, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1017(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1017, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1018(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1018, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1019(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1019, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1020(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1020, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1021(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1021, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1022(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1022, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1023(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1023, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits5F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<5, 1024, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

