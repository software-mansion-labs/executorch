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

// fp32 accumulation, as fast as fp16 on the GPUs measured and far more accurate over K = 1024-2048.
#define ACC_VEC4 vec4

// out [M, N] = mat1 [M, K] x W^T, with W prepacked by pack_fp_linear_weight into buffer texels
// ((k4 * N4 + n4) * 4 + dk) = W^T[4 k4 + dk][4 n4 .. 4 n4 + 3]. A workgroup of TX x TY threads computes a BM x BN
// output tile (BM = TY * TM, BN = 4 * TX * TN4) from shared-memory tiles of the input (BM x BK) and the weight
// (BK x BN), BK = 4 * BK4, loading the next tiles into registers while it multiplies the current ones. Thread (tx, ty)
// owns rows ty + TY * i and column groups tx + TX * j. Needs K % BK == 0 and N % BN == 0; M is bounds-checked.
#define TX ${TX}
#define TY ${TY}
#define TM ${TM}
#define TN4 ${TN4}
#define BK4 ${BK4}
#define BM (TY * TM)
#define BN4 (TX * TN4)
#define NTHREADS (TX * TY)

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "has_bias", "0")}

// As[k4][m]: 4 consecutive k of one input row. Bs[k][n4]: 4 consecutive output columns of one k. One padding
// column each, so the strided writes spread over the banks.
shared VEC4_T As[BK4][BM + 1];
shared VEC4_T Bs[BK4 * 4][BN4 + 1];

#define A_PER (BM * BK4 / NTHREADS)
#define B_PER (BK4 * 4 * BN4 / NTHREADS)

void main() {
  const int tx = int(gl_LocalInvocationID.x);
  const int ty = int(gl_LocalInvocationID.y);
  const int tid = ty * TX + tx;
  const int n4_0 = int(gl_WorkGroupID.x) * BN4;
  const int m_0 = int(gl_WorkGroupID.y) * BM;

  const int K4 = div_up_4(mat1_sizes.x);
  const int M = mat1_sizes.y;
  const int N4 = div_up_4(out_sizes.x);

  ACC_VEC4 acc[TM][TN4];
  [[unroll]] for (int i = 0; i < TM; ++i) {
    [[unroll]] for (int j = 0; j < TN4; ++j) {
      acc[i][j] = ACC_VEC4(0);
    }
  }

  // The next tiles, held in registers while the current ones are multiplied.
  VEC4_T ra[A_PER];
  VEC4_T rb[B_PER];
  [[unroll]] for (int p = 0; p < A_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    const int m = m_0 + idx / BK4;
    ra[p] = m < M ? t_mat1[m * K4 + idx % BK4] : VEC4_T(0);
  }
  // Weight texels in memory order (dk fastest, then n4, then k4): one contiguous run per k4.
  [[unroll]] for (int p = 0; p < B_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    rb[p] = t_weight_packed[((idx / (4 * BN4)) * N4 + n4_0) * 4 + idx % (4 * BN4)];
  }

  for (int k4_0 = 0; k4_0 < K4; k4_0 += BK4) {
    [[unroll]] for (int p = 0; p < A_PER; ++p) {
      const int idx = tid + p * NTHREADS;
      As[idx % BK4][idx / BK4] = ra[p];
    }
    [[unroll]] for (int p = 0; p < B_PER; ++p) {
      const int idx = tid + p * NTHREADS;
      Bs[(idx / (4 * BN4)) * 4 + idx % 4][(idx / 4) % BN4] = rb[p];
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
      [[unroll]] for (int p = 0; p < B_PER; ++p) {
        const int idx = tid + p * NTHREADS;
        rb[p] = t_weight_packed[((next + idx / (4 * BN4)) * N4 + n4_0) * 4 + idx % (4 * BN4)];
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
        vec4 v = acc[i][j];
        if (has_bias != 0) {
          v += vec4(t_bias[n4_0 + tx + TX * j]);
        }
        t_output[m * N4 + n4_0 + tx + TX * j] = VEC4_T(v);
      }
    }
  }
}
