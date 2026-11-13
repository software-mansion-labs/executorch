/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define T ${texel_load_component_type(DTYPE, "buffer")}

// conv1d_lds (same bindings and spec constants; Snake on the input and the clamp, flags bits 0 and 3, are the only
// fusions) where its matmul tiles would not pay: fewer than 16 terms per output or terms not a multiple of 16 (a
// 1-channel input, depthwise), or at most 4 rows. Each invocation computes 4 consecutive outputs of one channel from a
// window of its inputs held in registers (WIN >= 3 stride + dilation (kernel_size - 1) + 1).
#define WIN 64

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=True)}
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

void main() {
  const int L_in = in_sizes.x;
  const int Cin = in_sizes.y;
  const int L = out_sizes.x;
  const int C = out_sizes.y;
  const int l0 = int(gl_GlobalInvocationID.x) * 4;
  const int co = int(gl_GlobalInvocationID.y);
  if (l0 >= L || co >= C) {
    return;
  }
  const int cin_g = Cin / groups;
  const int cin0 = (co / (C / groups)) * cin_g;
  const int span = 3 * stride + dilation * (kernel_size - 1) + 1;
  const int start = l0 * stride - padding;

  vec4 acc = vec4(0);
  for (int cl = 0; cl < cin_g; ++cl) {
    const int ci = cin0 + cl;
    float alpha = 0.0, rsqrt = 0.0, sign = 0.0;
    if ((flags & 1) != 0) {
      alpha = t_in_snake[ci];
      rsqrt = t_in_snake[Cin + ci];
      sign = t_in_snake[2 * Cin + ci];
    }
    float win[WIN];
    for (int w = 0; w < span; ++w) {
      const int src = start + w;
      float x = src >= 0 && src < L_in ? float(t_in[ci * L_in + src]) : 0.0;
      if ((flags & 1) != 0) {
        const float s = sin(alpha * x) * rsqrt;
        x += sign * s * s;
      }
      win[w] = x;
    }
    for (int k = 0; k < kernel_size; ++k) {
      const int o = k * dilation;
      const float w = float(t_weight[(co * cin_g + cl) * kernel_size + k]);
      acc = fma(vec4(w), vec4(win[o], win[o + stride], win[o + 2 * stride], win[o + 3 * stride]), acc);
    }
  }
  const float bias = float(t_bias[co]);
  [[unroll]] for (int e = 0; e < 4; ++e) {
    if (l0 + e < L) {
      float v = acc[e] + bias;
      if ((flags & 8) != 0) {
        v = clamp(v, out_min, out_max);
      }
      t_out[co * L + l0 + e] = T(v);
    }
  }
}
