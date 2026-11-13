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

// DAC's Snake activation over a buffer-backed [1, C, L] (width packed), in one pass and in fp32:
// out = x + sign * (sin(alpha * x) * rsqrt)^2, with params [3, C] = (alpha, 1 / sqrt|alpha|, sign of alpha).
${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_params", "float", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int L = in_sizes.x;
  const int C = in_sizes.y;
  const int e0 = int(gl_GlobalInvocationID.x) * 4;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    const int e = e0 + i;
    if (e < L * C) {
      const int c = e / L;
      const float x = float(t_in[e]);
      const float r = sin(t_params[c] * x) * t_params[C + c];
      t_out[e] = T(x + t_params[2 * C + c] * r * r);
    }
  }
}
