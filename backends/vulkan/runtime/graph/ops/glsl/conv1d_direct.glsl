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

// conv1d_lds for at most 4 output channels (bindings and spec constants alike; Snake on the input is the only fusion
// used, flags bit 0), where the 64-row tiles would sit nearly empty: each invocation computes 4 consecutive outputs
// of every channel, from a window of the input held in registers (WIN >= 4 + dilation * (kernel_size - 1)).
#define WIN 32

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

void main() {
  const int L_in = in_sizes.x;
  const int Cin = in_sizes.y;
  const int L = out_sizes.x;
  const int C = out_sizes.y;
  const int l0 = int(gl_GlobalInvocationID.x) * 4;
  if (l0 >= L) {
    return;
  }
  const int span = 4 + dilation * (kernel_size - 1);

  vec4 acc[4];
  [[unroll]] for (int c = 0; c < 4; ++c) {
    acc[c] = vec4(0);
  }
  for (int ci = 0; ci < Cin; ++ci) {
    float alpha = 0.0, rsqrt = 0.0, sign = 0.0;
    if ((flags & 1) != 0) {
      alpha = t_in_snake[ci];
      rsqrt = t_in_snake[Cin + ci];
      sign = t_in_snake[2 * Cin + ci];
    }
    float win[WIN];
    for (int w = 0; w < span; ++w) {
      const int src = l0 - padding + w;
      float x = src >= 0 && src < L_in ? float(t_in[ci * L_in + src]) : 0.0;
      if ((flags & 1) != 0) {
        const float s = sin(alpha * x) * rsqrt;
        x += sign * s * s;
      }
      win[w] = x;
    }
    for (int k = 0; k < kernel_size; ++k) {
      const int o = k * dilation;
      const vec4 xs = vec4(win[o], win[o + 1], win[o + 2], win[o + 3]);
      for (int c = 0; c < C; ++c) {
        acc[c] = fma(vec4(float(t_weight[(c * Cin + ci) * kernel_size + k])), xs, acc[c]);
      }
    }
  }
  for (int c = 0; c < C; ++c) {
    const float bias = float(t_bias[c]);
    [[unroll]] for (int e = 0; e < 4; ++e) {
      if (l0 + e < L) {
        t_out[c * L + l0 + e] = T(acc[c][e] + bias);
      }
    }
  }
}
