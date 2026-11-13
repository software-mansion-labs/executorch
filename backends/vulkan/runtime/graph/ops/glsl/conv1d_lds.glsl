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

// 1-D convolution of a buffer-backed input [1, Cin, L_in] (width packed), in `groups` groups of Cin / groups inputs
// and M / groups rows: y[m, q] = bias[m] + sum over its group's ci and k of w[m, ci, k] *
// in[ci, q * stride + k * dilation - padding], zero outside the input, for the weight's M rows and
// N = (L_in + 2 padding - dilation (kernel_size - 1) - 1) / stride + 1 columns. Computed per group (workgroup z) as a
// matmul over (Cin / groups) * kernel_size terms on the linear_lds tiling: a workgroup of TX x TY threads computes a
// (8 TY) x (4 TX TN4) tile of (m, q), 8 x 4 TN4 per thread, with the weight tile (texels of 4 terms) and the input
// tile (gathered from the shifted windows into texels of 4 q) staged in shared memory, the next tiles prefetched, fp32
// accumulation. Needs (Cin / groups * kernel_size) % 16 == 0. Fused around it (flags): Snake on the input as it is
// gathered (bit 0, in_snake [3, Cin]); then per output Snake (bit 1, out_snake [3, C]), a residual add (bit 2,
// [1, C, L]) and a clamp to [out_min, out_max] (bit 3, et_vk.conv_with_clamp). With phases > 1 the rows are `phases`
// interleaved phases of C = M / phases channels (a transposed convolution in polyphase form): y[r C + c, q] lands at
// out[c, q phases + r - crop], kept inside [0, L) for out [1, C, L].
// The q8 variants take int8 weights, 4 per int32 word ([M, terms / 4], terms in (input, tap) order), and fp32 scales,
// one per `group_size` consecutive terms of a row ([M, terms / group_size]), dequantized as the tile is loaded.
#define TX ${TX}
#define TY ${TY}
#define TM 8
#define TN4 ${TN4}
#define BK4 4
#define BM (TY * TM)
#define BN4 (TX * TN4)
#define NTHREADS (TX * TY)
#define A_PER (BM * BK4 / NTHREADS)
// Input tile: 16 terms x 4 BN4 columns, COLS consecutive columns of one term per thread.
#define COLS (BN4 * 4 * 16 / NTHREADS)
#define GROUPS (BN4 * 4 / COLS)

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer", is_scalar_array=True)}
$if QUANT == "q8":
  ${layout_declare_tensor(B, "r", "t_weight", "int", "buffer", is_scalar_array=True)}
  ${layout_declare_tensor(B, "r", "t_scales", "float", "buffer", is_scalar_array=True)}
$else:
  ${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_in_snake", "float", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_out_snake", "float", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_residual", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "kernel_size", "1")}
${layout_declare_spec_const(C, "int", "dilation", "1")}
${layout_declare_spec_const(C, "int", "padding", "0")}
${layout_declare_spec_const(C, "int", "phases", "1")}
${layout_declare_spec_const(C, "int", "crop", "0")}
${layout_declare_spec_const(C, "int", "flags", "0")}
${layout_declare_spec_const(C, "int", "stride", "1")}
${layout_declare_spec_const(C, "int", "groups", "1")}
${layout_declare_spec_const(C, "int", "group_size", "4")}
${layout_declare_spec_const(C, "float", "out_min", "0.0")}
${layout_declare_spec_const(C, "float", "out_max", "0.0")}

shared VEC4_T As[BK4][BM + 1];
shared VEC4_T Bs[BK4 * 4][BN4 + 1];

// Texel k4 (terms 4 k4 .. 4 k4 + 3) of weight row `row`.
VEC4_T load_weight(const int row, const int k4, const int K4) {
$if QUANT == "q8":
  const int w = t_weight[row * K4 + k4];
  const ivec4 q = ivec4(w << 24, w << 16, w << 8, w) >> 24;
  return VEC4_T(vec4(q) * t_scales[row * (K4 * 4 / group_size) + k4 * 4 / group_size]);
$else:
  return t_weight[row * K4 + k4];
}

// COLS consecutive columns from n_b of input term `term` (channel cin0 + term / kernel_size), zero outside the input,
// with Snake applied if fused (Snake(0) = 0, so the zero padding is unchanged).
void gather(out T rb[COLS], const int term, const int n_b, const int L_in, const int Cin, const int cin0) {
  const int ci = cin0 + term / kernel_size;
  const int base = ci * L_in + (term % kernel_size) * dilation - padding + n_b * stride;
  const int lo = ci * L_in;
  const int hi = lo + L_in;
  float alpha = 0.0, rsqrt = 0.0, sign = 0.0;
  if ((flags & 1) != 0) {
    alpha = t_in_snake[ci];
    rsqrt = t_in_snake[Cin + ci];
    sign = t_in_snake[2 * Cin + ci];
  }
  [[unroll]] for (int c = 0; c < COLS; ++c) {
    const int src = base + c * stride;
    T v = src >= lo && src < hi ? t_in[src] : T(0);
    if ((flags & 1) != 0) {
      const float x = float(v);
      const float s = sin(alpha * x) * rsqrt;
      v = T(x + sign * s * s);
    }
    rb[c] = v;
  }
}

void main() {
  const int tx = int(gl_LocalInvocationID.x);
  const int ty = int(gl_LocalInvocationID.y);
  const int tid = ty * TX + tx;
  const int n4_0 = int(gl_WorkGroupID.x) * BN4;
  const int m_0 = int(gl_WorkGroupID.y) * BM;

  const int L_in = in_sizes.x;
  const int Cin = in_sizes.y;
  const int L = out_sizes.x;
  const int C = out_sizes.y;
  // Rows and input channels of this workgroup's group.
  const int M = C * phases / groups;
  const int row0 = int(gl_WorkGroupID.z) * M;
  const int cin0 = int(gl_WorkGroupID.z) * (Cin / groups);
  const int N = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
  const int K4 = div_up_4(Cin / groups * kernel_size);
  // Uniform over the workgroup, so before any barrier.
  if (m_0 >= M || n4_0 * 4 >= N) {
    return;
  }

  const int kl = tid / GROUPS;
  const int cl0 = (tid % GROUPS) * COLS;
  const int n_b = n4_0 * 4 + cl0;

  vec4 acc[TM][TN4];
  [[unroll]] for (int i = 0; i < TM; ++i) {
    [[unroll]] for (int j = 0; j < TN4; ++j) {
      acc[i][j] = vec4(0);
    }
  }

  VEC4_T ra[A_PER];
  T rb[COLS];
  [[unroll]] for (int p = 0; p < A_PER; ++p) {
    const int idx = tid + p * NTHREADS;
    const int m = m_0 + idx / BK4;
    ra[p] = m < M ? load_weight(row0 + m, idx % BK4, K4) : VEC4_T(0);
  }
  gather(rb, kl, n_b, L_in, Cin, cin0);

  for (int k4_0 = 0; k4_0 < K4; k4_0 += BK4) {
    [[unroll]] for (int p = 0; p < A_PER; ++p) {
      const int idx = tid + p * NTHREADS;
      As[idx % BK4][idx / BK4] = ra[p];
    }
    [[unroll]] for (int j = 0; j < COLS / 4; ++j) {
      Bs[kl][(cl0 >> 2) + j] = VEC4_T(rb[4 * j], rb[4 * j + 1], rb[4 * j + 2], rb[4 * j + 3]);
    }
    memoryBarrierShared();
    barrier();

    if (k4_0 + BK4 < K4) {
      const int next = k4_0 + BK4;
      [[unroll]] for (int p = 0; p < A_PER; ++p) {
        const int idx = tid + p * NTHREADS;
        const int m = m_0 + idx / BK4;
        ra[p] = m < M ? load_weight(row0 + m, next + idx % BK4, K4) : VEC4_T(0);
      }
      gather(rb, next * 4 + kl, n_b, L_in, Cin, cin0);
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

  [[unroll]] for (int i = 0; i < TM; ++i) {
    const int m = m_0 + ty + TY * i;
    if (m < M) {
      const int row = row0 + m;
      const float bias = float(t_bias[row]);
      const int r = row / C;
      const int c = row - r * C;
      float alpha = 0.0, rsqrt = 0.0, sign = 0.0;
      if ((flags & 2) != 0) {
        alpha = t_out_snake[c];
        rsqrt = t_out_snake[C + c];
        sign = t_out_snake[2 * C + c];
      }
      [[unroll]] for (int j = 0; j < TN4; ++j) {
        const int q = (n4_0 + tx + TX * j) * 4;
        [[unroll]] for (int e = 0; e < 4; ++e) {
          const int t = (q + e) * phases + r - crop;
          if (q + e < N && t >= 0 && t < L) {
            float v = acc[i][j][e] + bias;
            if ((flags & 2) != 0) {
              const float s = sin(alpha * v) * rsqrt;
              v += sign * s * s;
            }
            if ((flags & 4) != 0) {
              v += float(t_residual[c * L + t]);
            }
            if ((flags & 8) != 0) {
              v = clamp(v, out_min, out_max);
            }
            t_out[c * L + t] = T(v);
          }
        }
      }
    }
  }
}
