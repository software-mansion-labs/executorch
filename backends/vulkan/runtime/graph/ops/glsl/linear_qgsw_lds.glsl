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

// out [M, N] = mat1 [M, K] x dequant(W)^T (+ bias) for int8 or int4 weights with one fp32 scale per output column and
// group of `group_size` consecutive k: linear_lds's tiling (a 64 x 64 output tile per workgroup of 8 x 8 threads, K
// tile 16, fp32 accumulation, the next tiles prefetched) with the weight dequantized as it is loaded. W is W^T packed
// along N into int32 words: int8, 4 columns per word ([K, N / 4], byte i = column 4 n4 + i); int4, 8 per word
// ([K, N / 8], nibble i = column 8 n8 + i, stored + 8). Scales: [K / group_size, N]. mat1 is [M, K] or [1, M, K].
// Needs K % 16 == 0 and N % 64 == 0; M is bounds-checked.

#define ACC_VEC4 vec4
#define TX 8
#define TY 8
#define TM 8
#define TN4 2
#define BK4 4
#define BM (TY * TM)
#define BN4 (TX * TN4)
#define NTHREADS (TX * TY)
#define A_PER (BM * BK4 / NTHREADS)
#define B_PER (BK4 * 4 * BN4 / NTHREADS)

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_scales", "float", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "group_size", "32")}
${layout_declare_spec_const(C, "int", "has_bias", "0")}

shared VEC4_T As[BK4][BM + 1];
shared VEC4_T Bs[BK4 * 4][BN4 + 1];

// This thread's weight texels of the K tile from row k0, dequantized.
void load_weight(out VEC4_T rb[B_PER], const int tid, const int k0, const int n4_0, const int N4) {
$if BITS == 8:
  [[unroll]] for (int p = 0; p < B_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    const int k = k0 + idx / BN4;
    const int n4 = n4_0 + idx % BN4;
    const int w = t_weight[k * N4 + n4];
    const ivec4 q = ivec4(w << 24, w << 16, w << 8, w) >> 24;
    rb[p] = VEC4_T(vec4(q) * t_scales[(k / group_size) * N4 + n4]);
  }
$else:
  [[unroll]] for (int p = 0; p < B_PER / 2; ++p) {
    const int idx = tid + p * NTHREADS;
    const int k = k0 + idx / (BN4 / 2);
    const int n8 = n4_0 / 2 + idx % (BN4 / 2);
    const int w = t_weight[k * (N4 / 2) + n8];
    const vec4 lo = vec4(w & 0xF, (w >> 4) & 0xF, (w >> 8) & 0xF, (w >> 12) & 0xF) - 8.0;
    const vec4 hi = vec4((w >> 16) & 0xF, (w >> 20) & 0xF, (w >> 24) & 0xF, (w >> 28) & 0xF) - 8.0;
    const int s = (k / group_size) * N4 + 2 * n8;
    rb[2 * p] = VEC4_T(lo * t_scales[s]);
    rb[2 * p + 1] = VEC4_T(hi * t_scales[s + 1]);
  }
}

void store_weight(const VEC4_T rb[B_PER], const int tid) {
$if BITS == 8:
  [[unroll]] for (int p = 0; p < B_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    Bs[idx / BN4][idx % BN4] = rb[p];
  }
$else:
  [[unroll]] for (int p = 0; p < B_PER / 2; ++p) {
    const int idx = tid + p * NTHREADS;
    Bs[idx / (BN4 / 2)][2 * (idx % (BN4 / 2))] = rb[2 * p];
    Bs[idx / (BN4 / 2)][2 * (idx % (BN4 / 2)) + 1] = rb[2 * p + 1];
  }
}

void main() {
  const int tx = int(gl_LocalInvocationID.x);
  const int ty = int(gl_LocalInvocationID.y);
  const int tid = ty * TX + tx;
  const int n4_0 = int(gl_WorkGroupID.x) * BN4;
  const int m_0 = int(gl_WorkGroupID.y) * BM;

  const int K4 = div_up_4(mat1_sizes.x);
  const int M = mat1_sizes.y * mat1_sizes.z;
  const int N4 = div_up_4(out_sizes.x);

  ACC_VEC4 acc[TM][TN4];
  [[unroll]] for (int i = 0; i < TM; ++i) {
    [[unroll]] for (int j = 0; j < TN4; ++j) {
      acc[i][j] = ACC_VEC4(0);
    }
  }

  VEC4_T ra[A_PER];
  VEC4_T rb[B_PER];
  [[unroll]] for (int p = 0; p < A_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    const int m = m_0 + idx / BK4;
    ra[p] = m < M ? t_mat1[m * K4 + idx % BK4] : VEC4_T(0);
  }
  load_weight(rb, tid, 0, n4_0, N4);

  for (int k4_0 = 0; k4_0 < K4; k4_0 += BK4) {
    [[unroll]] for (int p = 0; p < A_PER; ++p) {
      const int idx = tid + p * NTHREADS;
      As[idx % BK4][idx / BK4] = ra[p];
    }
    store_weight(rb, tid);
    memoryBarrierShared();
    barrier();

    if (k4_0 + BK4 < K4) {
      const int next = k4_0 + BK4;
      [[unroll]] for (int p = 0; p < A_PER; ++p) {
        const int idx = tid + p * NTHREADS;
        const int m = m_0 + idx / BK4;
        ra[p] = m < M ? t_mat1[m * K4 + next + idx % BK4] : VEC4_T(0);
      }
      load_weight(rb, tid, next * 4, n4_0, N4);
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
