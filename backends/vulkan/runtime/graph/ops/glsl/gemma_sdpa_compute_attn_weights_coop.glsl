/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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
$if MASK_PER_HEAD == 1:
  #define MASK_PER_HEAD
$if HAS_SOFTCAP == 1:
  #define HAS_SOFTCAP

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M 1
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

#define NUM_WORKERS_PER_OUT 64

${define_required_extensions(IO_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_q", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}
$if HAS_MASK == 1:
  ${layout_declare_tensor(B, "r", "t_attn_mask", DTYPE, IO_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "ivec4", "k_cache_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

$if HAS_SOFTCAP == 1:
  layout(push_constant) uniform restrict Block {
    float softcap;
    float inv_softcap;
  };

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Gemma4 exports Q/K in DHSB layout; helpers default to DSHB.
#define Q_LAYOUT DHSB
#define K_LAYOUT DHSB
#include "sdpa_fp_q_projected_tile_load.glslh"
#include "sdpa_fp_k_cache_tile_load.glslh"
// iter136 fp32-attn-acc: fp32 accumulator + fp32 partial-sums for the
// cooperative tree reduction (head_dim ~ 512 partials per worker, 64
// workers per output → trees of 64 fp16 sums was a major drift point).
#include "gemma_sdpa_fp32_attn_weight_tile_store.glslh"

shared SDPAFPOutTileFP32 partial_sums[NUM_WORKERS_PER_OUT];

/*
 * Gemma4-flavored cooperative variant. See the tiled variant for the
 * Gemma-vs-llama diff. This variant runs only when seq_len == 1 (decode).
 */

void main() {
  const int worker_id = int(gl_LocalInvocationID.y);

  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  const int q_h = int(gl_GlobalInvocationID.z);

  const int c = tile_idx_x * TILE_N;
  const int c4 = div_4(c);

  const int s = 0;

  const int D = q_projected_sizes.x;
  const int D4 = div_up_4(q_projected_sizes.x);
  const int Q_H = q_projected_sizes.y;
  const int S = q_projected_sizes.z;
  const int S_aligned = align_up_4(S);

  const int KV_H = k_cache_sizes.y;
  const int C = k_cache_sizes.z;
  const int C4 = div_up_4(C);

  int kv_h = q_h;
  if (KV_H < Q_H) {
    kv_h = q_h / (Q_H / KV_H);
  }

  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  if (c >= context_len || s >= S || q_h >= Q_H) {
    return;
  }

  SDPAFPOutTileFP32 out_tile;
  fp32_initialize(out_tile);

  FPInputTile q_tile;
  FPWeightTile w_tile;

#ifdef HAS_MASK
  // With a caller mask, always run the matmul.
  for (int d4 = worker_id; d4 < D4; d4 += NUM_WORKERS_PER_OUT) {
    load_q_projected_tile_with_checks(q_tile, d4, s, q_h, D4, D, S, Q_H);
    load_k_cache_tile_with_checks(w_tile, d4, c, kv_h, D4, D, context_len, C, KV_H);
    fp32_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
  }
#else
  bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
  if (tile_in_mask_region) {
    fp32_set_out_tile_to_vec(out_tile, vec4(fp32_negative_infinity_val));
  } else {
    for (int d4 = worker_id; d4 < D4; d4 += NUM_WORKERS_PER_OUT) {
      load_q_projected_tile_with_checks(q_tile, d4, s, q_h, D4, D, S, Q_H);
      load_k_cache_tile_with_checks(w_tile, d4, c, kv_h, D4, D, context_len, C, KV_H);
      fp32_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
    }
  }
#endif

  partial_sums[worker_id] = out_tile;

  memoryBarrierShared();
  barrier();

  for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i /= 2) {
    if (worker_id < i) {
      fp32_accumulate_out_tile_with_out_tile(
          partial_sums[worker_id], partial_sums[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }

  if (worker_id == 0) {
    out_tile = partial_sums[0];
#ifdef HAS_MASK
    fp32_apply_mask_only_with_mask(out_tile, input_pos, c, s, q_h, S, C4);
#ifdef HAS_SOFTCAP
    // Audio extension (Phase 2B): tanh(x/cap) * cap, AFTER mask add.
    fp32_apply_softcap_tile(out_tile, softcap, inv_softcap);
#endif
#else
    bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
    if (!tile_in_mask_region) {
      fp32_apply_mask_only_causal(out_tile, input_pos, c, s);
    }
#endif

    fp32_store_attn_weight_tile_with_checks(
      out_tile, c4, s, q_h, context_texel_len, S_aligned, Q_H);
  }
}
