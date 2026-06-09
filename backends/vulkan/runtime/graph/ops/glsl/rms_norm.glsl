/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

$if NO_WEIGHT:
  #define NO_WEIGHT

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"
#include "convert.glslh"

#define NUM_WORKERS_PER_ROW 64

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}
$if not NO_WEIGHT:
  ${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform PRECISION restrict Block {
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Accumulate in fp32 regardless of DTYPE — fp16 sum-of-squares overflows
// (RMSNorm classic numerical hazard). Only the loads/stores are DTYPE.
shared float shared_sumsq[NUM_WORKERS_PER_ROW];

void main() {
  // One workgroup per output row. The row index iterates over numel/width
  // (i.e. B*S*... rows of `hidden_size` elements each).
  const uint row_id = gl_GlobalInvocationID.y;

  // Each row of the input has `width(inp)` elements (= hidden_size).
  const uint row_width = width(inp);

  // Number of (B*S*...) rows = numel/width.
  if (out_of_bounds(row_id * row_width, inp)) {
    return;
  }

  const uint worker_id = gl_LocalInvocationID.x;
  const uint row_base = row_id * row_width;

  // -------- Pass 1: sum(x*x) in fp32 across the row. --------
  float local_sumsq = 0.0;
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS_PER_ROW) {
    const float v = float(t_in[row_base + x]);
    local_sumsq += v * v;
  }

  shared_sumsq[worker_id] = local_sumsq;

  memoryBarrierShared();
  barrier();

  for (uint stride = NUM_WORKERS_PER_ROW / 2; stride > 0; stride >>= 1) {
    if (worker_id < stride) {
      shared_sumsq[worker_id] += shared_sumsq[worker_id + stride];
    }
    memoryBarrierShared();
    barrier();
  }

  // Broadcast the final mean(x*x) + eps -> rstd to every lane.
  // shared_sumsq[0] currently holds the full row sum-of-squares.
  const float mean_sq = shared_sumsq[0] / float(row_width);
  const float rstd = inversesqrt(mean_sq + epsilon);

  // -------- Pass 2: out[i] = x[i] * rstd * weight[i] --------
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS_PER_ROW) {
    const float v = float(t_in[row_base + x]);
#ifdef NO_WEIGHT
    const float y = v * rstd;
#else
    const float w = float(t_weight[x]);
    const float y = v * rstd * w;
#endif
    t_out[row_base + x] = convert_to_T(y);
  }
}
