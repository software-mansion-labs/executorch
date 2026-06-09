/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Fused decode-path attn-weights compute + softmax.
 *
 * Standard llama coop attn-weights shader (this file's no-softmax sibling)
 * dispatches one workgroup per (c4, q_h) and tree-reduces across head_dim.
 * Each WG emits one VEC4 to t_attn_weights, then a separate shader does
 * softmax across the c axis in a 3-pass max-subtract algorithm.
 *
 * This fused variant restructures the dispatch: one workgroup per (s, q_h),
 * with 64 workers cooperatively iterating ALL c4 values for that row. After
 * computing the row, the WG reduces in-shared-memory to find row_max and
 * exp_sum, then writes normalized softmax values to t_attn_weights directly.
 *
 * Removes one full dispatch (sdpa_attn_weights_softmax) plus the read/write
 * round-trip on t_attn_weights between the matmul and softmax steps.
 *
 * Trade-off: drops c-axis parallelism (was 50–512 WGs per q_h, now 1).
 * Per-WG work scales accordingly. Acceptable at decode S=1 where the
 * dispatch overhead per WG dominates wall time.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if K_CACHE_STORAGE == "buffer":
  #define K_CACHE_BUFFER

$if HAS_MASK == 1:
  #define HAS_MASK
$if HAS_MASK == 1 and IO_STORAGE == "buffer":
  #define MASK_BUFFER

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M 1
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

#define NUM_WORKERS_PER_OUT 64

// Shared mem cap for the in-shader attn_weights row. Must exceed the largest
// expected context_texel_len at decode (context_len / 4 + 1). For Gemma4 with
// max_context_len = 2048, this is 512 VEC4 entries. Sized to 1024 to leave
// headroom for longer contexts.
#define MAX_CONTEXT_TEXEL_LEN 1024

${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_q_projected", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k_cache", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}
$if HAS_MASK == 1:
  ${layout_declare_tensor(B, "r", "t_attn_mask", DTYPE, IO_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "ivec4", "k_cache_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}

#include "sdpa_fp_q_projected_tile_load.glslh"
#include "sdpa_fp_k_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_attn_weight_tile_store.glslh"

// Shared mem: row of attn_weights (after scale + mask), plus per-worker
// scratch for max + exp_sum tree-reductions.
shared T row_attn_w_x[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_y[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_z[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_w[MAX_CONTEXT_TEXEL_LEN];
shared T shared_max[NUM_WORKERS_PER_OUT];
shared T shared_sum[NUM_WORKERS_PER_OUT];

void main() {
  const int worker_id = int(gl_LocalInvocationID.x);

  // idx along output seq_len dim (always 1 at decode)
  const int s = int(gl_GlobalInvocationID.y);
  // idx along output num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // texel size of head_dim, over which the dot product is accumulated
  const int D4 = div_up_4(q_projected_sizes.x);
  // number of Q heads
  const int Q_H = q_projected_sizes.y;
  // sequence length
  const int S = q_projected_sizes.z;
  const int S_aligned = align_up_4(S);

  // number of K/V heads
  const int KV_H = k_cache_sizes.y;
  // Max context length
  const int C = k_cache_sizes.z;
  const int C4 = div_up_4(C);

  int kv_h = q_h;
  if (KV_H < Q_H) {
    kv_h = q_h / (Q_H / KV_H);
  }

  // current context length
  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  if (s >= S || q_h >= Q_H) {
    return;
  }

  const VEC4_T inv_scale_vec = VEC4_T(inv_scale);
  T local_max = T(-1.0 / 0.0); // -inf

  // Pass 1: compute attn_weight for each (c4, s, q_h) this worker owns;
  // apply scale + mask; stash to shared mem; track per-worker local max.
  FPInputTile q_tile;
  FPWeightTile w_tile;
  FPOutTile out_tile;

  // Each worker owns c4 = worker_id, worker_id+64, worker_id+128, ...
  for (int c4 = worker_id; c4 < context_texel_len; c4 += NUM_WORKERS_PER_OUT) {
    const int c = mul_4(c4);

    initialize(out_tile);

#ifdef HAS_MASK
    bool tile_in_mask_region = false;
#else
    bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
#endif

    if (tile_in_mask_region) {
      const VEC4_T negative_infinity_vec = VEC4_T(negative_infinity_val);
      set_out_tile_to_vec(out_tile, negative_infinity_vec);
    } else {
      for (int d4 = 0; d4 < D4; ++d4) {
        load_q_projected_tile_no_checks(q_tile, d4, s, q_h, D4, Q_H, S);
        load_k_cache_tile_with_checks(w_tile, d4, c, kv_h, D4, context_len, C, KV_H);
        fp_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
      }

#ifdef HAS_MASK
      apply_scale_and_mask_with_mask(out_tile, inv_scale_vec, input_pos, c, s, q_h, S, C4);
#else
      apply_scale_and_mask(out_tile, inv_scale_vec, input_pos, c, s);
#endif
    }

    // out_tile is TILE_M=1 x TILE_N4=1, so the value is out_tile.data[0][0].
    VEC4_T raw = out_tile.data[0][0];

    // Mask out OOB c-positions within the last texel — they must not
    // contribute to row_max. Set to -inf so exp() yields 0 below.
    const int c_base = c;
    if (c_base + 0 >= context_len) raw.x = negative_infinity_val;
    if (c_base + 1 >= context_len) raw.y = negative_infinity_val;
    if (c_base + 2 >= context_len) raw.z = negative_infinity_val;
    if (c_base + 3 >= context_len) raw.w = negative_infinity_val;

    row_attn_w_x[c4] = raw.x;
    row_attn_w_y[c4] = raw.y;
    row_attn_w_z[c4] = raw.z;
    row_attn_w_w[c4] = raw.w;

    local_max = max(local_max, max(max(raw.x, raw.y), max(raw.z, raw.w)));
  }

  // Reduce row_max across workers.
  shared_max[worker_id] = local_max;
  memoryBarrierShared();
  barrier();
  for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_max[worker_id] = max(shared_max[worker_id], shared_max[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }
  T row_max = shared_max[0];

  // Pass 2: per-worker sum of exp(s - row_max) over its slice.
  T local_sum = T(0);
  for (int c4 = worker_id; c4 < context_texel_len; c4 += NUM_WORKERS_PER_OUT) {
    local_sum += exp(row_attn_w_x[c4] - row_max);
    local_sum += exp(row_attn_w_y[c4] - row_max);
    local_sum += exp(row_attn_w_z[c4] - row_max);
    local_sum += exp(row_attn_w_w[c4] - row_max);
  }

  shared_sum[worker_id] = local_sum;
  memoryBarrierShared();
  barrier();
  for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_sum[worker_id] = shared_sum[worker_id] + shared_sum[worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }
  T row_sum = shared_sum[0];

  // Pass 3: write exp(s - row_max) / row_sum to t_attn_weights.
  // OOB c-positions in the last texel were set to -inf above, so exp(-inf-row_max)=0
  // contributes 0 to the consumer matmul (this matches consumer load-with-checks zeroing).
  for (int c4 = worker_id; c4 < context_texel_len; c4 += NUM_WORKERS_PER_OUT) {
    VEC4_T raw;
    raw.x = row_attn_w_x[c4];
    raw.y = row_attn_w_y[c4];
    raw.z = row_attn_w_z[c4];
    raw.w = row_attn_w_w[c4];

    VEC4_T out_texel = exp(raw - VEC4_T(row_max)) / row_sum;

    FPOutTile out_tile_w;
    out_tile_w.data[0][0] = out_texel;
    store_attn_weight_tile_with_checks(
        out_tile_w, c4, s, q_h, context_texel_len, S_aligned, Q_H);
  }
}
