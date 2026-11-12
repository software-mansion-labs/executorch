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

// Attention scores with the LLM SDPA's 2D additive mask, as
// sdpa_compute_attn_weights_tiled_mask2d computes them: scores[h, s, c] =
// q[s, h] . k[c, h / (H / KV_H)] * inv_scale + mask[s, c], for buffer-backed q
// [1, S, H, D] and k [1, C, KV_H, D] (DHSB) and mask [S, C]; scores are texels
// of 4 c at (h * align_up_4(S) + s) * div_up_4(C) + c4. The linear_lds tiling:
// a workgroup of 8 x 8 threads computes a 64 x 64 tile of (s, c) of one head, 8
// x 8 per thread, from q and k tiles staged in shared memory (k transposed to
// texels of 4 c), with fp32 accumulation. Needs D % 16 == 0.
#define TX 8
#define TY 8
#define TM 8
#define TN4 2
#define BK4 4
#define BM (TY * TM)
#define BN4 (TX * TN4)
#define NTHREADS (TX * TY)
#define A_PER (BM * BK4 / NTHREADS)

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_q", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mask", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "ivec4", "k_cache_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}
${layout_declare_ubo(B, "ivec4", "mask_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}

shared VEC4_T As[BK4][BM + 1];
shared VEC4_T Bs[BK4 * 4][BN4 + 1];

void main() {
  const int tx = int(gl_LocalInvocationID.x);
  const int ty = int(gl_LocalInvocationID.y);
  const int tid = ty * TX + tx;
  const int n4_0 = int(gl_WorkGroupID.x) * BN4;
  const int m_0 = int(gl_WorkGroupID.y) * BM;
  const int q_h = int(gl_WorkGroupID.z);

  const int D4 = div_up_4(q_projected_sizes.x);
  const int Q_H = q_projected_sizes.y;
  const int S = q_projected_sizes.z;
  const int KV_H = k_cache_sizes.y;
  const int kv_h = q_h / (Q_H / KV_H);
  const int context_len = input_pos + S;
  const int C4 = div_up_4(context_len);
  // Uniform over the workgroup, so before any barrier.
  if (m_0 >= S || n4_0 >= C4 || q_h >= Q_H) {
    return;
  }

  // Each thread stages a texel of d for 4 consecutive c of the k tile.
  const int n4l = tid % BN4;
  const int d4l = tid / BN4;
  const int c_b = (n4_0 + n4l) * 4;

  vec4 acc[TM][TN4];
  [[unroll]] for (int i = 0; i < TM; ++i) {
    [[unroll]] for (int j = 0; j < TN4; ++j) {
      acc[i][j] = vec4(0);
    }
  }

  VEC4_T ra[A_PER];
  VEC4_T rb[4];
  [[unroll]] for (int p = 0; p < A_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    const int m = m_0 + idx / BK4;
    ra[p] = m < S ? t_q[(m * Q_H + q_h) * D4 + idx % BK4] : VEC4_T(0);
  }
  [[unroll]] for (int i = 0; i < 4; ++i) {
    rb[i] = c_b + i < context_len ? t_k[((c_b + i) * KV_H + kv_h) * D4 + d4l] : VEC4_T(0);
  }

  for (int k4_0 = 0; k4_0 < D4; k4_0 += BK4) {
    [[unroll]] for (int p = 0; p < A_PER; ++p) {
      const int idx = tid + p * NTHREADS;
      As[idx % BK4][idx / BK4] = ra[p];
    }
    [[unroll]] for (int e = 0; e < 4; ++e) {
      Bs[d4l * 4 + e][n4l] = VEC4_T(rb[0][e], rb[1][e], rb[2][e], rb[3][e]);
    }
    memoryBarrierShared();
    barrier();

    if (k4_0 + BK4 < D4) {
      const int next = k4_0 + BK4;
      [[unroll]] for (int p = 0; p < A_PER; ++p) {
        const int idx = tid + p * NTHREADS;
        const int m = m_0 + idx / BK4;
        ra[p] = m < S ? t_q[(m * Q_H + q_h) * D4 + next + idx % BK4] : VEC4_T(0);
      }
      [[unroll]] for (int i = 0; i < 4; ++i) {
        rb[i] = c_b + i < context_len ? t_k[((c_b + i) * KV_H + kv_h) * D4 + next + d4l] : VEC4_T(0);
      }
    }

    [[unroll]] for (int kk = 0; kk < BK4; ++kk) {
      VEC4_T a[TM];
      [[unroll]] for (int i = 0; i < TM; ++i) {
        a[i] = As[kk][ty + TY * i];
      }
      [[unroll]] for (int dk = 0; dk < 4; ++dk) {
        [[unroll]] for (int j = 0; j < TN4; ++j) {
          const vec4 b = vec4(Bs[kk * 4 + dk][tx + TX * j]);
          [[unroll]] for (int i = 0; i < TM; ++i) {
            acc[i][j] = fma(vec4(a[i][dk]), b, acc[i][j]);
          }
        }
      }
    }
    barrier();
  }

  const int S_aligned = align_up_4(S);
  const int mask_C = mask_sizes.x;
  [[unroll]] for (int i = 0; i < TM; ++i) {
    const int m = m_0 + ty + TY * i;
    if (m < S) {
      [[unroll]] for (int j = 0; j < TN4; ++j) {
        const int n4 = n4_0 + tx + TX * j;
        if (n4 < C4) {
          vec4 v = acc[i][j] * inv_scale;
          [[unroll]] for (int e = 0; e < 4; ++e) {
            const int c = n4 * 4 + e;
            if (c < mask_C) {
              v[e] += float(t_mask[m * mask_C + c]);
            }
          }
          t_attn_weights[(q_h * S_aligned + m) * C4 + n4] = VEC4_T(v);
        }
      }
    }
  }
}
