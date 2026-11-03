/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

$DTYPE = IN_DTYPE

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}

#define NUM_WORKERS_PER_WG 64

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights_softmax", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Iter136 fp32-attn-acc: keep storage at DTYPE (fp16 when DTYPE=half), but
// run the row-max + exp_sum reductions in fp32 to avoid fp16 cumulative
// drift over long context lengths. Storage of the softmax output stays at
// DTYPE so downstream (V*attn-weights) is unchanged.
//
// Iter 88 fix: the softmax subtracts the row max before exp(). Without it,
// fp16 exp() saturates to inf for any QK > ~11.09 and underflows to 0 for
// QK < ~-11, and either case yields NaN in the final divide.
#define SOFTMAX_IN_VEC4_T VEC4_T
#define SOFTMAX_ACC_T float

VEC4_T load_attn_weights_c4(
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  return t_attn_weights[(q_h * S * C4) + (s * C4) + c4];
#else
  return texelFetch(t_attn_weights, ivec3(c4, s, q_h), 0);
#endif
}

void store_attn_weights_softmax_c4(
    const VEC4_T out_texel,
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  t_attn_weights_softmax[(q_h * S * C4) + (s * C4) + c4] = out_texel;
#else
  imageStore(t_attn_weights_softmax, ivec3(c4, s, q_h), out_texel);
#endif
}

#include "sdpa_attn_weights_softmax_row.glslh"

void main() {
  const int worker_id = int(gl_LocalInvocationID.x);

  // Index along attention weight's sequence_len dim
  const int s = int(gl_GlobalInvocationID.y);
  // idx along attention weight's num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // number of Q heads
  const int Q_H = q_projected_sizes.y;
  // sequence length
  const int S = q_projected_sizes.z;
  const int S_aligned = align_up_4(S);
  // manually determine size of the context_len dim of the attention weight.
  // The "actual" tensor sizes may have been aligned to a multiple of 4 to allow
  // memory loads to be aligned to texel boundaries.
  const int context_len = input_pos + S;

  if (s >= S || q_h >= Q_H) {
    return;
  }

  softmax_attn_weights_row(worker_id, s, q_h, context_len, S_aligned, Q_H);
}
