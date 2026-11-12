/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T ${texel_load_component_type(DTYPE, "buffer")}

// out [M, N] = mat1 [M, K] x dequant(W)^T for 4-bit weights in groups of `group_size` along K: linear_lds with the
// weight tile dequantized while it is stored to shared memory, fp32 accumulation. W is the W_4X8 nc-buffer prepack
// (pack_q4_linear_weight__w_4x8.glsl): one ivec4 per 4K x 8N block at block index (k4 * N8 + n8), each byte of a
// component holding an (even N, odd N) nibble pair at one k, codes stored + 8. Scales: [K / group_size, N].
// A workgroup of 8 x 8 threads computes a 64 x 64 output tile, 8 x 8 per thread; the K tile is 16 (faster than 32 on a
// Samsung Xclipse 940), so half the threads load and dequantize one weight block each, and every tile lies inside one
// scale group. Needs K % 16 == 0, N % 64 == 0 and group_size % 16 == 0; M is bounds-checked.
#define ACC_VEC4 vec4
#define TX 8
#define TY 8
#define TM 8
#define TN4 2
#define BK4 ${BK4}
#define BM (TY * TM)
#define BN4 (TX * TN4)
#define NTHREADS (TX * TY)
#define A_PER (BM * BK4 / NTHREADS)

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_weight", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "group_size", "32")}

shared VEC4_T As[BK4][BM + 1];
shared VEC4_T Bs[BK4 * 4][BN4 + 1];

// Weight block (k4, n8) into 4 k rows x 2 texels of 4 n, scaled.
void store_weight_block(const ivec4 block, const VEC4_T sa, const VEC4_T sb, const int k4l, const int n8l) {
  [[unroll]] for (int ki = 0; ki < 4; ++ki) {
    const ivec4 bytes = (block >> (8 * ki)) & 0xFF;
    // In fp32, converting once: fp16 arithmetic is slow where the driver must honour round-to-nearest.
    const vec4 wa = (vec4(bytes.x & 0xF, bytes.x >> 4, bytes.y & 0xF, bytes.y >> 4) - 8.0) * vec4(sa);
    const vec4 wb = (vec4(bytes.z & 0xF, bytes.z >> 4, bytes.w & 0xF, bytes.w >> 4) - 8.0) * vec4(sb);
    Bs[k4l * 4 + ki][n8l * 2] = VEC4_T(wa);
    Bs[k4l * 4 + ki][n8l * 2 + 1] = VEC4_T(wb);
  }
}

void main() {
  const int tx = int(gl_LocalInvocationID.x);
  const int ty = int(gl_LocalInvocationID.y);
  const int tid = ty * TX + tx;
  const int n4_0 = int(gl_WorkGroupID.x) * BN4;
  const int m_0 = int(gl_WorkGroupID.y) * BM;

  const int K4 = div_up_4(mat1_sizes.x);
  const int M = mat1_sizes.y;
  const int N4 = div_up_4(out_sizes.x);
  const int N8 = N4 / 2;

  // This thread's weight block in each tile, and its 8 output columns.
  const int k4l = tid / 8;
  const int n8l = tid % 8;
  const int n8 = n4_0 / 2 + n8l;

  ACC_VEC4 acc[TM][TN4];
  [[unroll]] for (int i = 0; i < TM; ++i) {
    [[unroll]] for (int j = 0; j < TN4; ++j) {
      acc[i][j] = ACC_VEC4(0);
    }
  }

  VEC4_T ra[A_PER];
  [[unroll]] for (int p = 0; p < A_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    const int m = m_0 + idx / BK4;
    ra[p] = m < M ? t_mat1[m * K4 + idx % BK4] : VEC4_T(0);
  }
  const bool loads_b = tid < BK4 * 8;
  ivec4 rb = ivec4(0);
  VEC4_T rsa = VEC4_T(0);
  VEC4_T rsb = VEC4_T(0);
  if (loads_b) {
    rb = t_packed_weight[k4l * N8 + n8];
    rsa = t_scales[n8 * 2];
    rsb = t_scales[n8 * 2 + 1];
  }

  for (int k4_0 = 0; k4_0 < K4; k4_0 += BK4) {
    [[unroll]] for (int p = 0; p < A_PER; ++p) {
      const int idx = tid + p * NTHREADS;
      As[idx % BK4][idx / BK4] = ra[p];
    }
    if (loads_b) {
      store_weight_block(rb, rsa, rsb, k4l, n8l);
    }
    memoryBarrierShared();
    barrier();

    if (k4_0 + BK4 < K4) {
      const int next = k4_0 + BK4;
      [[unroll]] for (int p = 0; p < A_PER; ++p) {
        const int idx = tid + p * NTHREADS;
        const int m = m_0 + idx / BK4;
        ra[p] = m < M ? t_mat1[m * K4 + next + idx % BK4] : VEC4_T(0);
      }
      if (loads_b) {
        rb = t_packed_weight[(next + k4l) * N8 + n8];
        const int group = (next * 4) / group_size;
        rsa = t_scales[group * N4 + n8 * 2];
        rsb = t_scales[group * N4 + n8 * 2 + 1];
      }
    }

    [[unroll]] for (int kk = 0; kk < BK4; ++kk) {
      VEC4_T a[TM];
      [[unroll]] for (int i = 0; i < TM; ++i) {
        a[i] = As[kk][ty + TY * i];
      }
      [[unroll]] for (int dk = 0; dk < 4; ++dk) {
        [[unroll]] for (int j = 0; j < TN4; ++j) {
          const ACC_VEC4 b = ACC_VEC4(Bs[kk * 4 + dk][tx + TX * j]);
          [[unroll]] for (int i = 0; i < TM; ++i) {
            acc[i][j] = fma(ACC_VEC4(a[i][dk]), b, acc[i][j]);
          }
        }
      }
    }
    barrier();
  }

  [[unroll]] for (int i = 0; i < TM; ++i) {
    const int m = m_0 + ty + TY * i;
    if (m < M) {
      [[unroll]] for (int j = 0; j < TN4; ++j) {
        t_output[m * N4 + n4_0 + tx + TX * j] = VEC4_T(acc[i][j]);
      }
    }
  }
}
