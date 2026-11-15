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

// Each row (one channel) of a buffer-backed [N, C, L] (width packed) normalized over its length, then scaled and
// shifted by its channel's weight and bias: GroupNorm with one group per channel. One workgroup per row; the
// statistics in fp32, in two passes.
${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}

layout(push_constant) uniform PRECISION restrict Block {
  float epsilon;
};

#define NUM_WORKERS 256

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared float partial[NUM_WORKERS];

float workgroup_sum(const uint worker, const float v) {
  partial[worker] = v;
  memoryBarrierShared();
  barrier();
  [[unroll]] for (uint stride = NUM_WORKERS / 2; stride > 0; stride >>= 1) {
    if (worker < stride) {
      partial[worker] += partial[worker + stride];
    }
    memoryBarrierShared();
    barrier();
  }
  const float total = partial[0];
  barrier();
  return total;
}

void main() {
  const int L = in_sizes.x;
  const int C = in_sizes.y;
  const uint worker = gl_LocalInvocationID.x;
  const int row = int(gl_WorkGroupID.y);
  const int base = row * L;

  float s = 0.0;
  for (int x = int(worker); x < L; x += NUM_WORKERS) {
    s += float(t_in[base + x]);
  }
  const float mean = workgroup_sum(worker, s) / float(L);

  float v = 0.0;
  for (int x = int(worker); x < L; x += NUM_WORKERS) {
    const float d = float(t_in[base + x]) - mean;
    v += d * d;
  }
  const float rstd = inversesqrt(workgroup_sum(worker, v) / float(L) + epsilon);

  const int c = row % C;
  const float scale = float(t_weight[c]) * rstd;
  const float shift = float(t_bias[c]) - mean * scale;
  for (int x = int(worker); x < L; x += NUM_WORKERS) {
    t_out[base + x] = T(float(t_in[base + x]) * scale + shift);
  }
}
