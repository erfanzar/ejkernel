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

void LaunchDequantAffineBits7F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_affine_int<7, half, half><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantAffineBits7F16Gs8(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 8, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs9(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 9, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs10(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 10, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs11(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 11, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs12(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 12, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs13(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 13, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs14(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 14, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs15(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 15, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 16, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs17(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 17, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs18(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 18, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs19(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 19, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs20(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 20, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs21(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 21, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs22(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 22, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs23(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 23, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs24(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 24, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs25(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 25, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs26(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 26, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs27(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 27, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs28(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 28, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs29(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 29, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs30(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 30, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs31(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 31, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 32, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs33(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 33, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs34(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 34, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs35(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 35, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs36(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 36, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs37(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 37, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs38(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 38, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs39(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 39, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs40(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 40, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs41(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 41, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs42(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 42, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs43(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 43, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs44(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 44, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs45(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 45, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs46(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 46, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs47(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 47, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs48(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 48, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs49(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 49, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs50(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 50, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs51(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 51, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs52(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 52, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs53(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 53, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs54(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 54, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs55(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 55, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs56(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 56, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs57(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 57, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs58(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 58, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs59(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 59, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs60(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 60, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs61(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 61, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs62(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 62, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs63(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 63, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 64, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs65(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 65, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs66(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 66, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs67(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 67, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs68(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 68, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs69(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 69, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs70(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 70, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs71(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 71, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs72(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 72, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs73(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 73, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs74(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 74, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs75(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 75, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs76(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 76, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs77(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 77, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs78(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 78, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs79(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 79, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs80(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 80, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs81(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 81, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs82(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 82, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs83(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 83, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs84(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 84, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs85(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 85, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs86(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 86, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs87(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 87, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs88(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 88, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs89(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 89, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs90(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 90, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs91(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 91, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs92(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 92, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs93(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 93, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs94(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 94, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs95(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 95, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs96(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 96, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs97(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 97, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs98(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 98, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs99(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 99, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs100(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 100, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs101(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 101, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs102(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 102, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs103(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 103, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs104(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 104, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs105(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 105, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs106(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 106, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs107(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 107, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs108(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 108, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs109(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 109, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs110(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 110, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs111(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 111, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs112(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 112, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs113(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 113, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs114(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 114, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs115(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 115, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs116(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 116, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs117(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 117, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs118(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 118, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs119(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 119, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs120(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 120, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs121(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 121, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs122(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 122, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs123(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 123, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs124(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 124, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs125(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 125, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs126(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 126, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs127(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 127, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 128, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs129(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 129, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs130(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 130, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs131(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 131, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs132(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 132, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs133(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 133, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs134(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 134, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs135(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 135, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs136(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 136, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs137(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 137, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs138(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 138, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs139(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 139, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs140(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 140, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs141(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 141, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs142(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 142, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs143(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 143, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs144(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 144, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs145(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 145, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs146(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 146, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs147(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 147, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs148(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 148, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs149(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 149, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs150(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 150, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs151(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 151, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs152(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 152, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs153(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 153, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs154(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 154, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs155(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 155, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs156(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 156, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs157(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 157, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs158(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 158, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs159(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 159, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs160(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 160, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs161(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 161, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs162(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 162, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs163(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 163, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs164(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 164, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs165(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 165, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs166(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 166, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs167(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 167, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs168(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 168, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs169(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 169, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs170(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 170, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs171(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 171, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs172(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 172, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs173(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 173, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs174(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 174, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs175(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 175, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs176(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 176, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs177(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 177, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs178(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 178, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs179(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 179, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs180(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 180, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs181(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 181, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs182(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 182, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs183(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 183, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs184(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 184, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs185(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 185, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs186(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 186, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs187(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 187, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs188(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 188, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs189(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 189, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs190(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 190, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs191(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 191, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs192(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 192, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs193(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 193, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs194(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 194, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs195(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 195, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs196(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 196, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs197(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 197, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs198(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 198, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs199(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 199, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs200(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 200, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs201(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 201, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs202(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 202, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs203(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 203, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs204(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 204, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs205(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 205, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs206(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 206, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs207(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 207, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs208(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 208, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs209(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 209, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs210(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 210, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs211(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 211, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs212(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 212, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs213(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 213, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs214(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 214, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs215(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 215, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs216(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 216, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs217(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 217, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs218(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 218, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs219(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 219, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs220(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 220, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs221(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 221, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs222(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 222, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs223(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 223, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs224(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 224, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs225(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 225, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs226(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 226, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs227(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 227, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs228(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 228, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs229(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 229, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs230(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 230, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs231(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 231, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs232(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 232, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs233(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 233, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs234(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 234, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs235(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 235, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs236(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 236, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs237(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 237, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs238(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 238, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs239(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 239, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs240(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 240, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs241(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 241, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs242(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 242, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs243(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 243, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs244(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 244, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs245(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 245, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs246(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 246, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs247(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 247, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs248(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 248, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs249(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 249, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs250(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 250, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs251(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 251, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs252(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 252, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs253(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 253, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs254(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 254, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs255(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 255, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 256, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs257(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 257, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs258(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 258, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs259(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 259, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs260(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 260, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs261(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 261, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs262(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 262, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs263(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 263, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs264(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 264, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs265(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 265, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs266(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 266, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs267(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 267, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs268(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 268, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs269(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 269, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs270(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 270, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs271(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 271, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs272(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 272, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs273(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 273, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs274(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 274, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs275(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 275, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs276(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 276, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs277(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 277, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs278(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 278, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs279(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 279, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs280(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 280, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs281(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 281, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs282(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 282, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs283(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 283, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs284(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 284, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs285(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 285, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs286(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 286, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs287(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 287, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs288(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 288, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs289(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 289, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs290(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 290, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs291(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 291, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs292(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 292, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs293(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 293, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs294(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 294, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs295(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 295, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs296(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 296, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs297(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 297, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs298(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 298, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs299(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 299, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs300(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 300, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs301(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 301, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs302(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 302, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs303(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 303, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs304(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 304, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs305(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 305, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs306(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 306, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs307(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 307, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs308(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 308, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs309(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 309, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs310(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 310, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs311(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 311, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs312(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 312, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs313(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 313, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs314(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 314, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs315(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 315, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs316(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 316, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs317(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 317, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs318(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 318, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs319(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 319, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs320(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 320, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs321(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 321, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs322(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 322, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs323(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 323, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs324(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 324, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs325(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 325, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs326(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 326, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs327(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 327, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs328(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 328, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs329(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 329, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs330(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 330, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs331(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 331, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs332(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 332, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs333(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 333, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs334(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 334, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs335(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 335, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs336(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 336, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs337(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 337, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs338(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 338, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs339(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 339, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs340(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 340, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs341(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 341, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs342(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 342, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs343(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 343, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs344(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 344, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs345(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 345, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs346(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 346, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs347(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 347, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs348(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 348, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs349(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 349, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs350(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 350, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs351(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 351, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs352(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 352, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs353(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 353, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs354(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 354, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs355(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 355, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs356(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 356, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs357(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 357, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs358(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 358, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs359(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 359, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs360(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 360, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs361(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 361, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs362(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 362, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs363(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 363, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs364(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 364, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs365(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 365, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs366(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 366, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs367(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 367, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs368(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 368, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs369(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 369, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs370(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 370, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs371(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 371, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs372(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 372, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs373(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 373, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs374(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 374, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs375(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 375, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs376(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 376, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs377(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 377, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs378(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 378, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs379(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 379, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs380(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 380, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs381(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 381, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs382(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 382, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs383(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 383, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs384(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 384, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs385(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 385, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs386(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 386, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs387(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 387, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs388(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 388, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs389(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 389, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs390(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 390, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs391(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 391, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs392(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 392, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs393(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 393, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs394(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 394, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs395(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 395, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs396(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 396, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs397(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 397, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs398(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 398, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs399(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 399, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs400(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 400, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs401(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 401, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs402(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 402, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs403(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 403, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs404(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 404, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs405(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 405, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs406(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 406, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs407(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 407, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs408(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 408, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs409(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 409, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs410(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 410, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs411(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 411, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs412(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 412, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs413(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 413, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs414(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 414, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs415(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 415, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs416(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 416, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs417(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 417, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs418(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 418, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs419(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 419, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs420(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 420, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs421(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 421, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs422(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 422, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs423(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 423, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs424(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 424, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs425(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 425, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs426(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 426, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs427(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 427, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs428(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 428, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs429(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 429, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs430(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 430, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs431(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 431, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs432(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 432, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs433(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 433, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs434(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 434, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs435(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 435, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs436(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 436, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs437(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 437, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs438(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 438, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs439(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 439, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs440(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 440, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs441(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 441, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs442(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 442, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs443(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 443, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs444(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 444, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs445(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 445, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs446(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 446, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs447(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 447, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs448(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 448, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs449(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 449, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs450(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 450, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs451(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 451, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs452(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 452, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs453(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 453, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs454(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 454, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs455(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 455, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs456(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 456, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs457(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 457, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs458(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 458, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs459(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 459, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs460(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 460, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs461(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 461, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs462(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 462, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs463(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 463, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs464(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 464, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs465(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 465, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs466(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 466, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs467(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 467, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs468(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 468, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs469(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 469, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs470(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 470, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs471(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 471, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs472(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 472, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs473(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 473, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs474(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 474, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs475(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 475, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs476(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 476, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs477(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 477, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs478(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 478, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs479(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 479, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs480(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 480, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs481(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 481, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs482(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 482, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs483(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 483, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs484(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 484, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs485(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 485, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs486(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 486, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs487(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 487, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs488(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 488, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs489(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 489, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs490(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 490, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs491(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 491, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs492(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 492, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs493(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 493, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs494(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 494, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs495(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 495, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs496(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 496, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs497(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 497, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs498(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 498, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs499(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 499, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs500(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 500, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs501(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 501, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs502(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 502, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs503(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 503, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs504(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 504, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs505(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 505, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs506(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 506, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs507(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 507, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs508(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 508, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs509(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 509, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs510(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 510, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs511(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 511, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 512, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs513(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 513, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs514(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 514, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs515(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 515, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs516(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 516, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs517(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 517, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs518(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 518, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs519(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 519, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs520(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 520, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs521(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 521, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs522(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 522, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs523(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 523, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs524(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 524, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs525(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 525, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs526(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 526, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs527(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 527, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs528(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 528, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs529(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 529, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs530(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 530, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs531(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 531, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs532(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 532, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs533(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 533, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs534(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 534, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs535(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 535, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs536(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 536, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs537(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 537, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs538(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 538, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs539(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 539, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs540(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 540, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs541(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 541, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs542(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 542, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs543(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 543, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs544(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 544, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs545(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 545, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs546(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 546, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs547(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 547, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs548(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 548, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs549(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 549, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs550(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 550, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs551(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 551, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs552(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 552, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs553(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 553, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs554(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 554, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs555(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 555, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs556(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 556, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs557(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 557, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs558(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 558, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs559(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 559, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs560(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 560, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs561(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 561, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs562(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 562, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs563(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 563, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs564(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 564, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs565(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 565, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs566(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 566, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs567(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 567, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs568(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 568, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs569(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 569, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs570(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 570, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs571(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 571, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs572(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 572, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs573(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 573, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs574(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 574, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs575(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 575, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs576(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 576, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs577(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 577, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs578(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 578, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs579(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 579, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs580(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 580, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs581(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 581, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs582(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 582, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs583(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 583, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs584(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 584, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs585(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 585, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs586(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 586, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs587(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 587, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs588(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 588, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs589(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 589, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs590(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 590, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs591(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 591, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs592(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 592, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs593(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 593, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs594(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 594, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs595(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 595, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs596(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 596, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs597(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 597, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs598(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 598, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs599(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 599, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs600(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 600, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs601(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 601, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs602(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 602, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs603(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 603, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs604(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 604, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs605(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 605, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs606(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 606, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs607(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 607, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs608(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 608, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs609(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 609, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs610(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 610, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs611(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 611, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs612(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 612, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs613(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 613, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs614(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 614, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs615(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 615, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs616(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 616, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs617(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 617, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs618(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 618, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs619(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 619, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs620(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 620, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs621(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 621, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs622(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 622, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs623(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 623, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs624(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 624, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs625(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 625, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs626(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 626, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs627(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 627, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs628(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 628, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs629(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 629, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs630(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 630, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs631(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 631, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs632(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 632, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs633(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 633, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs634(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 634, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs635(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 635, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs636(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 636, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs637(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 637, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs638(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 638, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs639(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 639, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs640(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 640, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs641(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 641, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs642(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 642, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs643(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 643, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs644(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 644, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs645(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 645, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs646(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 646, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs647(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 647, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs648(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 648, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs649(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 649, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs650(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 650, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs651(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 651, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs652(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 652, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs653(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 653, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs654(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 654, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs655(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 655, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs656(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 656, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs657(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 657, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs658(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 658, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs659(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 659, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs660(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 660, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs661(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 661, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs662(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 662, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs663(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 663, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs664(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 664, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs665(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 665, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs666(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 666, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs667(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 667, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs668(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 668, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs669(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 669, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs670(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 670, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs671(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 671, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs672(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 672, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs673(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 673, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs674(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 674, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs675(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 675, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs676(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 676, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs677(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 677, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs678(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 678, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs679(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 679, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs680(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 680, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs681(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 681, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs682(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 682, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs683(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 683, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs684(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 684, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs685(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 685, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs686(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 686, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs687(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 687, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs688(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 688, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs689(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 689, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs690(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 690, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs691(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 691, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs692(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 692, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs693(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 693, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs694(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 694, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs695(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 695, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs696(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 696, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs697(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 697, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs698(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 698, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs699(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 699, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs700(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 700, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs701(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 701, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs702(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 702, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs703(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 703, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs704(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 704, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs705(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 705, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs706(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 706, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs707(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 707, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs708(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 708, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs709(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 709, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs710(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 710, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs711(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 711, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs712(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 712, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs713(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 713, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs714(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 714, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs715(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 715, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs716(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 716, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs717(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 717, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs718(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 718, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs719(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 719, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs720(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 720, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs721(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 721, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs722(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 722, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs723(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 723, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs724(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 724, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs725(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 725, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs726(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 726, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs727(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 727, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs728(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 728, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs729(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 729, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs730(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 730, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs731(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 731, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs732(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 732, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs733(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 733, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs734(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 734, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs735(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 735, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs736(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 736, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs737(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 737, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs738(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 738, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs739(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 739, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs740(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 740, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs741(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 741, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs742(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 742, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs743(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 743, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs744(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 744, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs745(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 745, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs746(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 746, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs747(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 747, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs748(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 748, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs749(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 749, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs750(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 750, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs751(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 751, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs752(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 752, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs753(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 753, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs754(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 754, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs755(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 755, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs756(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 756, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs757(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 757, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs758(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 758, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs759(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 759, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs760(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 760, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs761(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 761, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs762(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 762, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs763(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 763, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs764(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 764, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs765(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 765, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs766(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 766, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs767(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 767, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs768(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 768, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs769(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 769, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs770(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 770, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs771(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 771, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs772(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 772, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs773(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 773, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs774(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 774, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs775(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 775, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs776(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 776, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs777(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 777, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs778(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 778, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs779(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 779, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs780(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 780, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs781(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 781, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs782(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 782, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs783(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 783, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs784(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 784, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs785(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 785, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs786(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 786, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs787(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 787, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs788(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 788, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs789(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 789, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs790(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 790, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs791(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 791, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs792(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 792, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs793(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 793, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs794(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 794, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs795(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 795, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs796(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 796, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs797(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 797, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs798(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 798, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs799(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 799, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs800(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 800, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs801(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 801, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs802(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 802, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs803(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 803, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs804(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 804, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs805(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 805, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs806(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 806, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs807(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 807, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs808(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 808, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs809(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 809, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs810(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 810, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs811(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 811, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs812(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 812, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs813(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 813, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs814(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 814, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs815(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 815, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs816(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 816, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs817(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 817, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs818(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 818, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs819(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 819, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs820(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 820, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs821(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 821, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs822(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 822, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs823(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 823, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs824(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 824, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs825(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 825, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs826(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 826, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs827(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 827, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs828(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 828, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs829(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 829, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs830(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 830, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs831(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 831, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs832(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 832, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs833(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 833, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs834(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 834, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs835(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 835, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs836(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 836, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs837(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 837, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs838(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 838, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs839(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 839, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs840(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 840, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs841(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 841, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs842(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 842, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs843(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 843, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs844(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 844, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs845(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 845, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs846(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 846, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs847(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 847, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs848(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 848, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs849(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 849, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs850(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 850, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs851(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 851, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs852(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 852, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs853(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 853, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs854(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 854, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs855(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 855, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs856(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 856, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs857(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 857, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs858(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 858, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs859(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 859, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs860(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 860, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs861(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 861, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs862(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 862, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs863(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 863, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs864(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 864, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs865(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 865, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs866(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 866, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs867(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 867, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs868(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 868, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs869(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 869, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs870(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 870, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs871(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 871, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs872(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 872, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs873(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 873, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs874(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 874, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs875(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 875, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs876(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 876, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs877(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 877, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs878(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 878, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs879(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 879, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs880(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 880, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs881(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 881, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs882(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 882, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs883(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 883, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs884(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 884, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs885(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 885, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs886(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 886, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs887(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 887, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs888(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 888, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs889(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 889, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs890(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 890, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs891(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 891, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs892(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 892, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs893(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 893, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs894(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 894, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs895(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 895, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs896(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 896, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs897(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 897, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs898(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 898, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs899(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 899, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs900(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 900, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs901(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 901, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs902(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 902, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs903(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 903, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs904(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 904, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs905(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 905, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs906(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 906, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs907(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 907, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs908(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 908, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs909(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 909, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs910(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 910, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs911(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 911, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs912(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 912, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs913(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 913, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs914(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 914, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs915(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 915, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs916(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 916, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs917(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 917, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs918(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 918, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs919(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 919, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs920(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 920, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs921(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 921, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs922(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 922, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs923(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 923, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs924(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 924, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs925(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 925, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs926(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 926, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs927(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 927, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs928(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 928, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs929(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 929, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs930(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 930, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs931(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 931, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs932(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 932, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs933(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 933, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs934(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 934, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs935(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 935, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs936(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 936, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs937(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 937, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs938(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 938, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs939(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 939, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs940(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 940, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs941(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 941, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs942(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 942, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs943(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 943, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs944(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 944, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs945(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 945, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs946(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 946, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs947(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 947, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs948(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 948, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs949(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 949, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs950(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 950, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs951(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 951, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs952(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 952, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs953(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 953, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs954(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 954, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs955(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 955, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs956(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 956, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs957(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 957, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs958(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 958, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs959(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 959, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs960(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 960, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs961(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 961, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs962(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 962, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs963(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 963, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs964(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 964, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs965(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 965, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs966(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 966, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs967(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 967, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs968(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 968, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs969(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 969, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs970(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 970, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs971(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 971, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs972(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 972, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs973(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 973, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs974(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 974, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs975(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 975, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs976(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 976, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs977(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 977, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs978(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 978, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs979(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 979, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs980(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 980, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs981(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 981, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs982(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 982, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs983(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 983, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs984(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 984, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs985(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 985, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs986(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 986, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs987(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 987, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs988(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 988, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs989(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 989, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs990(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 990, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs991(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 991, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs992(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 992, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs993(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 993, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs994(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 994, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs995(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 995, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs996(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 996, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs997(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 997, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs998(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 998, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs999(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 999, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1000(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1000, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1001(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1001, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1002(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1002, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1003(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1003, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1004(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1004, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1005(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1005, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1006(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1006, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1007(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1007, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1008(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1008, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1009(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1009, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1010(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1010, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1011(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1011, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1012(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1012, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1013(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1013, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1014(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1014, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1015(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1015, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1016(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1016, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1017(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1017, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1018(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1018, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1019(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1019, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1020(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1020, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1021(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1021, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1022(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1022, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1023(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1023, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}
void LaunchDequantAffineBits7F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_affine_int_gs<7, 1024, half, half><<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, n_words, n_groups);
}

