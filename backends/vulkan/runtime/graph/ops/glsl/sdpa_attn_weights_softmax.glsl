/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

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
// Shared memory for cooperative exp sum finding (fp32).
shared float shared_exp_sum[NUM_WORKERS_PER_WG];
// Iter 88 fix: shared memory for cooperative row-max finding. Without
// max-subtraction, fp16 exp() saturates to inf for any QK > ~11.09 and
// underflows to 0 for QK < ~-11. Either case yields NaN through 0/inf
// or 0/0 in the final divide. The reproducer shows 20/40 (pos,head)
// slots NaN for random gemma-shape weights with QK in [-45, +39]. With
// max-subtraction the exponent is clamped to [-large, 0] so exp() never
// overflows; small exp values still underflow to 0 but at least one term
// (where x == max) contributes exp(0)=1 to the sum so we never divide by 0.
shared float shared_row_max[NUM_WORKERS_PER_WG];

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
  const int context_texel_len = div_up_4(context_len);

  if (s >= S || q_h >= Q_H) {
    return;
  }

  const int context_len_aligned_down = context_len - mod_4(context_len);
  const int C4_limit = div_4(context_len_aligned_down);

  // Iter 88 fix: numerically-stable softmax via max-subtraction.
  // Iter136 fp32-attn-acc: max + exp_sum reductions run in fp32 even when
  // DTYPE=half. Cast to float on read, cast back to VEC4_T on store.
  // Pass 1: compute row max across this thread's slice.
  float local_max = -1.0 / 0.0; // -inf
  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_WG) {
    VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, S_aligned, Q_H);
    [[unroll]] for (int comp = 0; comp < 4; comp++) {
      local_max = max(local_max, float(in_texel[comp]));
    }
  }
  if (worker_id == 0) {
    for (int c4 = C4_limit; c4 < context_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, S_aligned, Q_H);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < context_len) {
          local_max = max(local_max, float(in_texel[comp]));
        }
      }
    }
  }

  // Reduce max across the workgroup.
  shared_row_max[worker_id] = local_max;
  memoryBarrierShared();
  barrier();
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_row_max[worker_id] =
          max(shared_row_max[worker_id], shared_row_max[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }
  float row_max = shared_row_max[0];

  // Pass 2: compute sum of exp(x - row_max) in fp32.
  float local_exp_sum = 0.0;
  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_WG) {
    VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, S_aligned, Q_H);

    for (int comp = 0; comp < 4; comp++) {
      local_exp_sum += exp(float(in_texel[comp]) - row_max);
    }
  }
  if (worker_id == 0) {
    for (int c4 = C4_limit; c4 < context_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, S_aligned, Q_H);

      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < context_len) {
          local_exp_sum += exp(float(in_texel[comp]) - row_max);
        }
      }
    }
  }

  // Reduce sum.
  shared_exp_sum[worker_id] = local_exp_sum;
  memoryBarrierShared();
  barrier();
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_exp_sum[worker_id] = shared_exp_sum[worker_id] +
          shared_exp_sum[worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }

  local_exp_sum = shared_exp_sum[0];
  // Pass 3: write exp(x - row_max) / sum (fp32 division, cast to VEC4_T on store).
  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_WG) {
    VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, S_aligned, Q_H);

    vec4 out_fp32 = exp(vec4(in_texel) - vec4(row_max)) / local_exp_sum;
    VEC4_T out_texel = VEC4_T(out_fp32);
    store_attn_weights_softmax_c4(
        out_texel, c4, s, q_h, context_texel_len, S_aligned, Q_H);
  }
  if (worker_id == 0) {
    for (int c4 = C4_limit; c4 < context_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, S_aligned, Q_H);

      vec4 out_fp32 = vec4(0);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < context_len) {
          out_fp32[comp] = exp(float(in_texel[comp]) - row_max) / local_exp_sum;
        }
      }
      VEC4_T out_texel = VEC4_T(out_fp32);
      store_attn_weights_softmax_c4(
          out_texel, c4, s, q_h, context_texel_len, S_aligned, Q_H);
    }
  }
}
